"""
Perdidas para Gabor splatting de audio en el dominio del tiempo.

Para audio, comparar la waveform sample-a-sample (L1/MSE) es perceptualmente
malo: dos senales que suenan identicas pueden diferir mucho muestra a muestra
por desfases minusculos. El estandar moderno (Parallel WaveGAN, HiFi-GAN, DDSP)
es la perdida multi-resolution STFT (MR-STFT): comparar la MAGNITUD del
espectrograma a varias resoluciones.

loss_total = lambda_wave * L1(x_hat, x)
           + lambda_mrstft * MR-STFT(x_hat, x)

MR-STFT a cada resolucion combina:
    - spectral convergence:  ||S - S_hat||_F / ||S||_F
    - log-magnitude L1:      mean(|log(S+eps) - log(S_hat+eps)|)
"""
import torch


_EPS = 1e-7


def _stft_mag(x, n_fft, hop, win):
    """x: [T] -> magnitud STFT [F, frames]."""
    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )
    return X.abs()


def _stft_loss_una_resolucion(x_hat, x, n_fft, hop, win):
    S = _stft_mag(x, n_fft, hop, win)
    S_hat = _stft_mag(x_hat, n_fft, hop, win)

    # spectral convergence
    num = torch.linalg.norm(S - S_hat)
    den = torch.linalg.norm(S).clamp_min(_EPS)
    sc = num / den

    # log-magnitude L1
    mag = torch.mean(torch.abs(torch.log(S + _EPS) - torch.log(S_hat + _EPS)))

    return sc + mag


def mrstft_loss(x_hat, x, ffts=(512, 1024, 2048), device=None):
    """
    Multi-resolution STFT loss. ffts: lista de tamanos de ventana.
    hop = n_fft // 4 para cada resolucion.
    """
    if device is None:
        device = x_hat.device

    total = x_hat.new_tensor(0.0)
    for n_fft in ffts:
        n_fft = int(n_fft)
        # Si la senal es mas corta que la ventana, saltar esa resolucion.
        if x_hat.shape[-1] < n_fft:
            continue
        hop = max(1, n_fft // 4)
        win = torch.hann_window(n_fft, device=device, dtype=x_hat.dtype)
        total = total + _stft_loss_una_resolucion(x_hat, x, n_fft, hop, win)

    return total


def si_sdr_loss(x_hat, x):
    """
    SI-SDR como termino de perdida (DIFERENCIABLE), a diferencia de la metrica
    si_sdr_db que es no_grad. Como SI-SDR es algo a MAXIMIZAR (dB altos = mejor),
    la perdida devuelve el NEGATIVO en dB, que es lo que se minimiza.
    """
    x = x - x.mean()
    x_hat = x_hat - x_hat.mean()
    alpha = torch.sum(x_hat * x) / torch.sum(x * x).clamp_min(_EPS)
    s_target = alpha * x
    e = x_hat - s_target
    pot_t = torch.sum(s_target * s_target).clamp_min(_EPS)
    pot_e = torch.sum(e * e).clamp_min(_EPS)
    sisdr = 10.0 * torch.log10(pot_t / pot_e)
    return -sisdr


def loss_gabor(x_hat, x, config):
    """
    Perdida combinada para Gabor audio.

    config:
        lambda_wave   : peso del L1 en waveform (default 0.0)
        lambda_mrstft : peso de la MR-STFT (default 1.0)
        lambda_sisdr  : peso del termino -SI-SDR (default 0.0)
        mrstft_ffts   : lista de tamanos de ventana (default [512,1024,2048])
    """
    lambda_wave = float(config.get("lambda_wave", 0.0))
    lambda_mrstft = float(config.get("lambda_mrstft", 1.0))
    lambda_sisdr = float(config.get("lambda_sisdr", 0.0))
    ffts = config.get("mrstft_ffts", [512, 1024, 2048])

    loss = x_hat.new_tensor(0.0)
    partes = {}

    if lambda_wave > 0.0:
        l_wave = torch.mean(torch.abs(x_hat - x))
        loss = loss + lambda_wave * l_wave
        partes["l_wave"] = float(l_wave.detach())

    if lambda_mrstft > 0.0:
        l_mr = mrstft_loss(x_hat, x, ffts=ffts, device=x_hat.device)
        loss = loss + lambda_mrstft * l_mr
        partes["l_mrstft"] = float(l_mr.detach())

    if lambda_sisdr > 0.0:
        l_si = si_sdr_loss(x_hat, x)
        loss = loss + lambda_sisdr * l_si
        partes["l_sisdr"] = float(l_si.detach())

    return loss, partes


@torch.no_grad()
def si_sdr_db(x_hat, x):
    """
    Scale-Invariant Signal-to-Distortion Ratio (Le Roux et al., 2019).

    A diferencia del SNR, es invariante a un reescalado global de x_hat: primero
    proyecta la estimacion sobre el target y luego mide el residuo ortogonal.
    Esto evita inflar el numero con un simple factor de ganancia.

        s_target = (<x_hat, x> / <x, x>) * x
        SI-SDR   = 10 log10( ||s_target||^2 / ||x_hat - s_target||^2 )

    Se resta la media (zero-mean) a ambas senales, como en la definicion canonica.
    """
    x = x.float()
    x_hat = x_hat.float()
    x = x - x.mean()
    x_hat = x_hat - x_hat.mean()

    alpha = torch.sum(x_hat * x) / torch.sum(x * x).clamp_min(1e-12)
    s_target = alpha * x
    e = x_hat - s_target
    pot_t = torch.sum(s_target * s_target).clamp_min(1e-12)
    pot_e = torch.sum(e * e).clamp_min(1e-12)
    return float((10.0 * torch.log10(pot_t / pot_e)).detach().cpu())


@torch.no_grad()
def lsd_db(x_hat, x, n_fft=1024, hop=256):
    """
    Log-Spectral Distance (en dB). Error de la magnitud espectral, en el mismo
    terreno que el PSNR_logmag del enfoque por espectrograma.

        LSD = mean_t  sqrt( mean_f ( 10 log10(P) - 10 log10(P_hat) )^2 )

    con P = |STFT|^2. Menor es mejor. Devuelve None si la senal es mas corta
    que la ventana.
    """
    if x_hat.shape[-1] < n_fft:
        return None
    win = torch.hann_window(n_fft, device=x_hat.device, dtype=x_hat.float().dtype)
    S = _stft_mag(x.float(), n_fft, hop, win)
    S_hat = _stft_mag(x_hat.float(), n_fft, hop, win)
    log_p = 10.0 * torch.log10(S * S + _EPS)
    log_p_hat = 10.0 * torch.log10(S_hat * S_hat + _EPS)
    diff2 = (log_p - log_p_hat) ** 2           # [F, frames]
    por_frame = torch.sqrt(torch.mean(diff2, dim=0))   # [frames]
    return float(torch.mean(por_frame).detach().cpu())


# ----------------------------------------------------------------------
# mel filterbank (implementado a mano para no depender de torchaudio)
# ----------------------------------------------------------------------
_MEL_CACHE = {}


def _hz_a_mel(f):
    # formula HTK
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def _mel_a_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(sr, n_fft, n_mels, fmin, fmax, device, dtype):
    """Devuelve [n_mels, n_freqs] con triangulos mel. Cacheado."""
    fmax = float(fmax) if fmax is not None else sr / 2.0
    key = (int(sr), int(n_fft), int(n_mels), float(fmin), fmax, str(device), str(dtype))
    if key in _MEL_CACHE:
        return _MEL_CACHE[key]

    n_freqs = n_fft // 2 + 1
    m_min = _hz_a_mel(torch.tensor(float(fmin)))
    m_max = _hz_a_mel(torch.tensor(fmax))
    puntos_mel = torch.linspace(float(m_min), float(m_max), n_mels + 2)
    puntos_hz = _mel_a_hz(puntos_mel)                      # [n_mels+2]
    fft_hz = torch.linspace(0.0, sr / 2.0, n_freqs)       # [n_freqs]

    fb = torch.zeros(n_mels, n_freqs)
    for m in range(1, n_mels + 1):
        f_izq, f_cen, f_der = puntos_hz[m - 1], puntos_hz[m], puntos_hz[m + 1]
        if f_cen > f_izq:
            rampa_sub = (fft_hz - f_izq) / (f_cen - f_izq)
        else:
            rampa_sub = torch.zeros_like(fft_hz)
        if f_der > f_cen:
            rampa_baj = (f_der - fft_hz) / (f_der - f_cen)
        else:
            rampa_baj = torch.zeros_like(fft_hz)
        fb[m - 1] = torch.clamp(torch.minimum(rampa_sub, rampa_baj), min=0.0)

    fb = fb.to(device=device, dtype=dtype)
    _MEL_CACHE[key] = fb
    return fb


@torch.no_grad()
def mel_l1(x_hat, x, sr, n_fft=2048, hop=512, n_mels=128, fmin=0.0, fmax=None):
    """
    L1 sobre el log-mel-espectrograma (la "mel loss" estandar de HiFi-GAN).
    Devuelve None si la senal es mas corta que la ventana.
    """
    if x_hat.shape[-1] < n_fft:
        return None
    dtype = x_hat.float().dtype
    win = torch.hann_window(n_fft, device=x_hat.device, dtype=dtype)
    P = _stft_mag(x.float(), n_fft, hop, win) ** 2
    P_hat = _stft_mag(x_hat.float(), n_fft, hop, win) ** 2
    fb = _mel_filterbank(sr, n_fft, n_mels, fmin, fmax, x_hat.device, dtype)
    mel = fb @ P
    mel_hat = fb @ P_hat
    lm = torch.log(mel + _EPS)
    lm_hat = torch.log(mel_hat + _EPS)
    return float(torch.mean(torch.abs(lm - lm_hat)).cpu())


@torch.no_grad()
def mrstft_metric(x_hat, x, ffts=(512, 1024, 2048)):
    """MR-STFT final como METRICA (mismo computo que la loss, reportado aparte)."""
    return float(mrstft_loss(x_hat, x, ffts=ffts).detach().cpu())


@torch.no_grad()
def metricas_audio(x_hat, x):
    """
    Metricas de reconstruccion en el dominio del tiempo y espectral.

    SNR_dB = 10 log10( ||x||^2 / ||x - x_hat||^2 )
    """
    x = x.float()
    x_hat = x_hat.float()

    err = x - x_hat
    pot_senal = torch.sum(x * x).clamp_min(1e-12)
    pot_error = torch.sum(err * err).clamp_min(1e-12)

    snr = 10.0 * torch.log10(pot_senal / pot_error)

    mse = torch.mean(err * err).clamp_min(1e-12)
    # PSNR asumiendo rango de senal [-1, 1] (amplitud pico 1, rango 2).
    psnr = 10.0 * torch.log10((2.0 ** 2) / mse)

    return {
        "snr_db": float(snr.detach().cpu()),
        "psnr_db": float(psnr.detach().cpu()),
        "mse_wave": float(mse.detach().cpu()),
        "si_sdr_db": si_sdr_db(x_hat, x),
        "lsd_db": lsd_db(x_hat, x),
    }
