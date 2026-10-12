"""
Graficas comparativas original vs reconstruido para un experimento Gabor.

Detecta automaticamente si el experimento es mono (target.wav / recon.wav) o
estereo (target_stereo.wav / recon_stereo.wav, usa el downmix para las figuras).

Genera en la carpeta del experimento:
    vis_waveform.png      <- forma de onda completa + zoom (original vs recon)
    vis_espectro.png      <- espectrogramas original / recon / |diferencia|
    vis_error.png         <- error absoluto en el tiempo + histograma del error

Uso (en la Mac, no necesita CUDA):
    python scripts/audio/visualizar.py --exp gabor_rock_30s_mono_f4v2_a07_N48k
    python scripts/audio/visualizar.py --exp <ruta_carpeta> --zoom-ms 30
"""
import argparse
import wave
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[2]


def leer_wav(ruta):
    """Devuelve (sr, x_mono_float). Si es estereo, downmix a mono."""
    w = wave.open(str(ruta), "rb")
    sr = w.getframerate()
    n = w.getnframes()
    ch = w.getnchannels()
    x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float64) / 32768.0
    w.close()
    if ch == 2:
        x = x.reshape(-1, 2).mean(axis=1)
    return sr, x


def localizar(carpeta):
    """Devuelve (target, recon, etiqueta) segun mono/estereo."""
    if (carpeta / "recon_stereo.wav").exists():
        return carpeta / "target_stereo.wav", carpeta / "recon_stereo.wav", "estereo (downmix)"
    if (carpeta / "recon.wav").exists():
        return carpeta / "target.wav", carpeta / "recon.wav", "mono"
    raise FileNotFoundError(f"No encuentro recon(.wav|_stereo.wav) en {carpeta}")


def snr_db(x, y):
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    e = x - y
    return 10 * np.log10(max(np.sum(x * x), 1e-12) / max(np.sum(e * e), 1e-12))


def fig_waveform(sr, t, r, ruta, zoom_ms, zoom_ini_s, titulo):
    n = min(len(t), len(r)); t, r = t[:n], r[:n]
    tiempo = np.arange(n) / sr
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), dpi=120)

    ax[0].plot(tiempo, t, lw=0.5, label="original", color="#1f77b4", alpha=0.8)
    ax[0].plot(tiempo, r, lw=0.5, label="reconstruido", color="#d62728", alpha=0.6)
    ax[0].set_title(f"Forma de onda completa — {titulo}  (SNR={snr_db(t, r):.2f} dB)")
    ax[0].set_xlabel("tiempo (s)"); ax[0].set_ylabel("amplitud")
    ax[0].legend(loc="upper right"); ax[0].grid(alpha=0.3)

    # zoom
    if zoom_ini_s is None:
        zoom_ini_s = (n / sr) / 2.0
    i0 = int(zoom_ini_s * sr)
    i1 = min(n, i0 + int(zoom_ms / 1000.0 * sr))
    tz = np.arange(i0, i1) / sr
    ax[1].plot(tz, t[i0:i1], lw=1.0, marker=".", ms=2, label="original", color="#1f77b4")
    ax[1].plot(tz, r[i0:i1], lw=1.0, marker=".", ms=2, label="reconstruido", color="#d62728", alpha=0.8)
    ax[1].set_title(f"Zoom {zoom_ms} ms (desde {zoom_ini_s:.1f}s) — detalle de la oscilacion")
    ax[1].set_xlabel("tiempo (s)"); ax[1].set_ylabel("amplitud")
    ax[1].legend(loc="upper right"); ax[1].grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(ruta); plt.close(fig)


def _stft_mag_db(x, n_fft=1024, hop=256):
    win = np.hanning(n_fft)
    n = 1 + (len(x) - n_fft) // hop
    S = np.empty((n_fft // 2 + 1, max(n, 1)), dtype=np.float64)
    for i in range(n):
        seg = x[i * hop: i * hop + n_fft] * win
        S[:, i] = np.abs(np.fft.rfft(seg))
    return 20 * np.log10(S + 1e-6)


def fig_espectro(sr, t, r, ruta, titulo):
    n = min(len(t), len(r)); t, r = t[:n], r[:n]
    St = _stft_mag_db(t); Sr = _stft_mag_db(r)
    m = min(St.shape[1], Sr.shape[1]); St, Sr = St[:, :m], Sr[:, :m]
    diff = np.abs(St - Sr)
    ext = [0, n / sr, 0, sr / 2000.0]  # x: s, y: kHz
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), dpi=120)
    vmin, vmax = -60, St.max()
    for a, S, ti in [(ax[0], St, "Original"), (ax[1], Sr, "Reconstruido")]:
        im = a.imshow(S, origin="lower", aspect="auto", extent=ext, vmin=vmin, vmax=vmax, cmap="magma")
        a.set_title(f"Espectrograma {ti} — {titulo}"); a.set_ylabel("kHz")
        fig.colorbar(im, ax=a, format="%+d dB", pad=0.01)
    im = ax[2].imshow(diff, origin="lower", aspect="auto", extent=ext, vmin=0, vmax=30, cmap="viridis")
    ax[2].set_title("|diferencia| log-magnitud (mas claro = mas error)")
    ax[2].set_xlabel("tiempo (s)"); ax[2].set_ylabel("kHz")
    fig.colorbar(im, ax=ax[2], format="%d dB", pad=0.01)
    fig.tight_layout(); fig.savefig(ruta); plt.close(fig)


def fig_error(sr, t, r, ruta, titulo):
    n = min(len(t), len(r)); t, r = t[:n], r[:n]
    e = t - r
    tiempo = np.arange(n) / sr
    fig, ax = plt.subplots(1, 2, figsize=(13, 4), dpi=120, gridspec_kw={"width_ratios": [3, 1]})
    ax[0].plot(tiempo, np.abs(e), lw=0.4, color="#7f0000")
    ax[0].set_title(f"Error absoluto |original - recon| — {titulo}")
    ax[0].set_xlabel("tiempo (s)"); ax[0].set_ylabel("|error|"); ax[0].grid(alpha=0.3)
    ax[1].hist(e, bins=120, color="#7f0000", alpha=0.8)
    ax[1].set_title("Histograma del error"); ax[1].set_xlabel("error"); ax[1].set_yscale("log")
    fig.tight_layout(); fig.savefig(ruta); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="nombre o ruta de la carpeta del experimento")
    ap.add_argument("--zoom-ms", type=float, default=30.0)
    ap.add_argument("--zoom-inicio-s", type=float, default=None)
    args = ap.parse_args()

    carpeta = Path(args.exp)
    if not carpeta.exists():
        carpeta = RAIZ / "outputs" / "gabor" / args.exp
    if not carpeta.exists():
        raise FileNotFoundError(f"No existe la carpeta: {args.exp}")

    target, recon, etiqueta = localizar(carpeta)
    sr, t = leer_wav(target)
    _, r = leer_wav(recon)
    titulo = carpeta.name

    fig_waveform(sr, t, r, carpeta / "vis_waveform.png", args.zoom_ms, args.zoom_inicio_s, titulo)
    fig_espectro(sr, t, r, carpeta / "vis_espectro.png", titulo)
    fig_error(sr, t, r, carpeta / "vis_error.png", titulo)
    print(f"[{etiqueta}] {titulo}  SNR={snr_db(t, r):.2f} dB")
    print(f"  guardadas: vis_waveform.png, vis_espectro.png, vis_error.png en {carpeta}")


if __name__ == "__main__":
    main()