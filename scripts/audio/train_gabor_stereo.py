"""
Entrenamiento ESTEREO de Gabor splatting de audio (waveform directa).

Dos dominios de entrenamiento (config "dominio"):
  - "LR" : se entrenan dos modelos independientes para L y R (como el companero).
  - "MS" : se entrenan en Mid = (L+R)/2 y Side = (L-R)/2, y se reconstruye
           L = M+S, R = M-S. El Mid concentra casi toda la energia y el Side es
           de baja energia, asi que se le pueden dar MENOS atomos al Side
           (n_atomos_side) -> mejor compresion y preserva la imagen estereo por
           construccion.

Cada canal (sea L/R o M/S) es exactamente el problema mono de entrenar_canal.

Metricas (mismas definiciones que el companero, mas extras):
    snr_left/right/promedio_lr/stereo, mse_wave_stereo, psnr_stereo
    si_sdr_*  (scale-invariant SDR)         lsd_*  (log-spectral distance)
    snr_mid/side, si_sdr_mid/side           <- detectan dano a la imagen estereo
    bloque "cuantizacion"                   <- fp32 vs fp16mix vs fp16full

Uso:
    python scripts/audio/train_gabor_stereo.py \
        --config configs/audio/gabor/gabor_rock_30s_stereo_LR_N48k.json
"""
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parents[2]

from gs2d_gabor.core.entrenamiento import entrenar_canal
from gs2d_gabor.core.perdidas_gabor import (
    si_sdr_db, lsd_db, metricas_audio, mel_l1, mrstft_metric,
)
from gs2d_gabor.core import cuantizacion as cuant


# ======================================================================
# IO de audio (estereo)
# ======================================================================

def cargar_wav_estereo(ruta, sr_objetivo=None, max_segundos=None):
    """Carga WAV estereo -> (sr, xL, xR) float32 en [-1, 1]."""
    from scipy.io import wavfile
    from scipy import signal

    sr, data = wavfile.read(str(ruta))

    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(
            f"El audio no es estereo (shape={getattr(data, 'shape', None)}). "
            f"Usa train_gabor.py para mono."
        )

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    data = np.nan_to_num(data)
    data = np.clip(data, -1.0, 1.0)

    xL = np.ascontiguousarray(data[:, 0])
    xR = np.ascontiguousarray(data[:, 1])

    if sr_objetivo is not None and int(sr_objetivo) != int(sr):
        gcd = np.gcd(int(sr), int(sr_objetivo))
        up = int(sr_objetivo) // gcd
        down = int(sr) // gcd
        xL = signal.resample_poly(xL, up, down).astype(np.float32)
        xR = signal.resample_poly(xR, up, down).astype(np.float32)
        sr = int(sr_objetivo)

    if max_segundos is not None:
        n = int(float(max_segundos) * sr)
        xL = xL[:n]
        xR = xR[:n]

    n = min(len(xL), len(xR))
    xL, xR = xL[:n], xR[:n]
    if n < 1024:
        raise RuntimeError("audio demasiado corto")

    return int(sr), xL, xR


def guardar_wav_estereo(ruta, sr, xL, xR):
    """Guarda [T, 2] int16. Normaliza por el maximo GLOBAL (preserva balance)."""
    from scipy.io import wavfile

    xL = np.nan_to_num(np.asarray(xL, dtype=np.float32))
    xR = np.nan_to_num(np.asarray(xR, dtype=np.float32))
    max_abs = max(float(np.max(np.abs(xL))), float(np.max(np.abs(xR))), 1e-12)
    if max_abs > 1.0:
        xL = xL / max_abs
        xR = xR / max_abs
    stereo = np.stack([np.clip(xL, -1, 1), np.clip(xR, -1, 1)], axis=1)
    wavfile.write(str(ruta), int(sr), (stereo * 32767.0).astype(np.int16))


def guardar_curva(valores, titulo, ylabel, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(valores)
    ax.set_xlabel("epoch"); ax.set_ylabel(ylabel); ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(ruta); plt.close(fig)


def guardar_log_csv(ruta, historial_loss, tiempos):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss", "tiempo_s"])
        for i, (lo, ti) in enumerate(zip(historial_loss, tiempos)):
            w.writerow([i + 1, f"{lo:.6f}", f"{ti:.4f}"])


# ======================================================================
# conversiones y metricas
# ======================================================================

def lr_a_ms(xL, xR):
    return 0.5 * (xL + xR), 0.5 * (xL - xR)


def ms_a_lr(xM, xS):
    return xM + xS, xM - xS


def _snr(x, x_hat):
    err = x - x_hat
    ps = torch.sum(x * x).clamp_min(1e-12)
    pe = torch.sum(err * err).clamp_min(1e-12)
    return float((10.0 * torch.log10(ps / pe)).cpu())


def metricas_estereo(xL, xR, xL_hat, xR_hat):
    """Metricas completas a partir de L/R original y reconstruido."""
    xL, xR = xL.float(), xR.float()
    xL_hat, xR_hat = xL_hat.float(), xR_hat.float()

    snr_L, snr_R = _snr(xL, xL_hat), _snr(xR, xR_hat)

    eL, eR = xL - xL_hat, xR - xR_hat
    ps = (torch.sum(xL * xL) + torch.sum(xR * xR)).clamp_min(1e-12)
    pe = (torch.sum(eL * eL) + torch.sum(eR * eR)).clamp_min(1e-12)
    snr_st = float((10.0 * torch.log10(ps / pe)).cpu())

    n_total = xL.numel() + xR.numel()
    mse_st = float(((torch.sum(eL * eL) + torch.sum(eR * eR)) / n_total).cpu())
    psnr_st = float((10.0 * torch.log10(torch.tensor(4.0) / max(mse_st, 1e-12))).cpu())

    # Mid / Side: detectan si el entrenamiento dano la imagen estereo.
    xM, xS = lr_a_ms(xL, xR)
    xM_hat, xS_hat = lr_a_ms(xL_hat, xR_hat)

    x_cat = torch.cat([xL, xR]); x_hat_cat = torch.cat([xL_hat, xR_hat])
    si_sdr_st = si_sdr_db(x_hat_cat, x_cat)

    lsd_L, lsd_R = lsd_db(xL_hat, xL), lsd_db(xR_hat, xR)
    lsd_prom = (0.5 * (lsd_L + lsd_R)) if (lsd_L is not None and lsd_R is not None) else None

    return {
        "snr_left_db": snr_L,
        "snr_right_db": snr_R,
        "snr_promedio_lr_db": 0.5 * (snr_L + snr_R),
        "snr_stereo_db": snr_st,
        "psnr_stereo_db": psnr_st,
        "mse_wave_stereo": mse_st,
        "si_sdr_left_db": si_sdr_db(xL_hat, xL),
        "si_sdr_right_db": si_sdr_db(xR_hat, xR),
        "si_sdr_stereo_db": si_sdr_st,
        "lsd_left_db": lsd_L,
        "lsd_right_db": lsd_R,
        "lsd_promedio_db": lsd_prom,
        "snr_mid_db": _snr(xM, xM_hat),
        "snr_side_db": _snr(xS, xS_hat),
        "si_sdr_mid_db": si_sdr_db(xM_hat, xM),
        "si_sdr_side_db": si_sdr_db(xS_hat, xS),
    }


# ======================================================================
# main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--nombre-experimento", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    if args.nombre_experimento is not None:
        config["nombre_experimento"] = args.nombre_experimento
    if not config.get("nombre_experimento"):
        from datetime import datetime
        config["nombre_experimento"] = "gabor_stereo_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    dominio = str(config.get("dominio", "LR")).upper()
    if dominio not in ("LR", "MS"):
        raise ValueError(f"dominio debe ser 'LR' o 'MS', no {dominio}")

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device(
        "cuda" if (config.get("device", "cuda") == "cuda" and torch.cuda.is_available())
        else "cpu"
    )
    print(f"device: {device}  dominio: {dominio}", flush=True)

    # === audio ===
    ruta_audio = Path(config["audio"])
    if not ruta_audio.is_absolute():
        ruta_audio = RAIZ / ruta_audio
    if not ruta_audio.exists():
        raise FileNotFoundError(f"No existe el audio: {ruta_audio}")

    sr, xL_np, xR_np = cargar_wav_estereo(
        ruta_audio, sr_objetivo=config.get("sr"), max_segundos=config.get("max_segundos"),
    )
    T = xL_np.shape[0]
    xL = torch.from_numpy(xL_np).to(device=device, dtype=torch.float32)
    xR = torch.from_numpy(xR_np).to(device=device, dtype=torch.float32)
    print(f"audio: {ruta_audio.name}  sr={sr}  samples/canal={T}  ({T/sr:.2f}s)  ESTEREO", flush=True)

    # === salida ===
    nombre_exp = config["nombre_experimento"]
    salida = RAIZ / "outputs" / "gabor" / nombre_exp
    sobreescribir = bool(config.get("sobreescribir_salida", False))
    if salida.exists() and not sobreescribir:
        raise FileExistsError(f"La carpeta ya existe: {salida}. Usa sobreescribir_salida=true.")

    if salida.exists() and sobreescribir and bool(config.get("limpiar_salida", False)):
        print(f"[audio] limpiando salida anterior: {salida}", flush=True)
        shutil.rmtree(salida)

    (salida / "checkpoints").mkdir(parents=True, exist_ok=True)

    with open(salida / "config_usada.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(salida / "info_audio.json", "w", encoding="utf-8") as f:
        json.dump({"audio": str(ruta_audio), "sr": sr, "samples": T,
                   "duracion_s": T / sr, "canales": "stereo",
                   "dominio": dominio, "seed": seed}, f, indent=2)

    guardar_wav_estereo(salida / "target_stereo.wav", sr, xL_np, xR_np)

    # === preparar los dos canales segun dominio ===
    if dominio == "LR":
        x1, x2 = xL, xR
        nom1, nom2 = "L", "R"
        n1 = n2 = int(config["n_atomos"])
    else:  # MS
        xM, xS = lr_a_ms(xL, xR)
        x1, x2 = xM, xS
        nom1, nom2 = "M", "S"
        n1 = int(config.get("n_atomos_mid", config["n_atomos"]))
        n2 = int(config.get("n_atomos_side", config["n_atomos"]))

    t_global = time.time()

    cfg1 = dict(config); cfg1["n_atomos"] = n1
    cfg2 = dict(config); cfg2["n_atomos"] = n2

    print(f"\n===== CANAL {nom1} (N={n1}) =====", flush=True)
    res1 = entrenar_canal(x1, sr, cfg1, device, semilla=seed, etiqueta=f"[{nom1}]")
    print(f"\n===== CANAL {nom2} (N={n2}) =====", flush=True)
    res2 = entrenar_canal(x2, sr, cfg2, device, semilla=seed + 1, etiqueta=f"[{nom2}]")

    # === reconstruccion -> L/R ===
    if dominio == "LR":
        xL_hat, xR_hat = res1["x_hat"], res2["x_hat"]
    else:
        xL_hat, xR_hat = ms_a_lr(res1["x_hat"], res2["x_hat"])

    guardar_wav_estereo(salida / "recon_stereo.wav", sr,
                        xL_hat.cpu().numpy(), xR_hat.cpu().numpy())

    # === metricas (sin cuantizar) ===
    met_st = metricas_estereo(xL, xR, xL_hat, xR_hat)

    # Mel-L1 y MR-STFT final (promedio L/R), misma bateria que el companero.
    ffts = config.get("mrstft_ffts", [512, 1024, 2048])
    mel_L, mel_R = mel_l1(xL_hat, xL, sr), mel_l1(xR_hat, xR, sr)
    met_st["mel_l1_promedio"] = (0.5 * (mel_L + mel_R)) if (mel_L is not None and mel_R is not None) else None
    met_st["mr_stft_promedio"] = 0.5 * (mrstft_metric(xL_hat, xL, ffts) + mrstft_metric(xR_hat, xR, ffts))

    # === cuantizacion: por cada esquema, re-render degradado y re-medir ===
    bloque_cuant = {}
    for nombre_esq, esquema in cuant.ESQUEMAS.items():
        x1q = cuant.render_cuantizado(res1["modelo"], esquema)
        x2q = cuant.render_cuantizado(res2["modelo"], esquema)
        if dominio == "LR":
            xLq, xRq = x1q, x2q
        else:
            xLq, xRq = ms_a_lr(x1q, x2q)
        mq = metricas_estereo(xL, xR, xLq, xRq)
        bytes_modelo = (n1 + n2) * cuant.bytes_por_atomo(esquema) + 8  # +4 gain/canal
        bytes_wav = T * 2 * 2
        bloque_cuant[nombre_esq] = {
            "bytes_por_atomo": cuant.bytes_por_atomo(esquema),
            "bytes_modelo": bytes_modelo,
            "ratio_compresion_vs_wav": bytes_wav / max(1, bytes_modelo),
            "snr_stereo_db": mq["snr_stereo_db"],
            "si_sdr_stereo_db": mq["si_sdr_stereo_db"],
            "mse_wave_stereo": mq["mse_wave_stereo"],
        }

    bytes_wav = T * 2 * 2
    bytes_modelo_fp32 = (n1 + n2) * cuant.bytes_por_atomo(cuant.ESQUEMAS["fp32"]) + 8
    metricas = {
        "dominio": dominio,
        **met_st,
        "n_atomos_canal1": n1, "n_atomos_canal2": n2, "n_atomos_total": n1 + n2,
        "samples_por_canal": T, "sr": sr, "duracion_s": T / sr,
        "bytes_modelo": bytes_modelo_fp32,
        "bytes_wav_int16_stereo": bytes_wav,
        "ratio_compresion_vs_wav": bytes_wav / max(1, bytes_modelo_fp32),
        "bitrate_kbps": bytes_modelo_fp32 * 8 / (T / sr) / 1000.0,
        "cuantizacion": bloque_cuant,
        "loss_final_c1": res1["historial_loss"][-1] if res1["historial_loss"] else None,
        "loss_final_c2": res2["historial_loss"][-1] if res2["historial_loss"] else None,
        "tiempo_total_s": time.time() - t_global,
    }
    with open(salida / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2)

    print("\n=== resultado ESTEREO ===", flush=True)
    print(f"  SNR_L={met_st['snr_left_db']:.2f}  SNR_R={met_st['snr_right_db']:.2f}  "
          f"SNR_ST={met_st['snr_stereo_db']:.2f} dB", flush=True)
    print(f"  SI-SDR_ST={met_st['si_sdr_stereo_db']:.2f} dB  "
          f"SNR_Mid={met_st['snr_mid_db']:.2f}  SNR_Side={met_st['snr_side_db']:.2f} dB", flush=True)
    _lp = met_st["lsd_promedio_db"]
    print(f"  LSD_prom={(_lp if _lp is not None else float('nan')):.3f} dB  "
          f"MSE_ST={met_st['mse_wave_stereo']:.6f}", flush=True)
    _melp = met_st["mel_l1_promedio"]
    print(f"  Mel-L1_prom={(_melp if _melp is not None else float('nan')):.4f}  "
          f"MR-STFT_prom={met_st['mr_stft_promedio']:.4f}", flush=True)
    print(f"  [fp32 ] bytes={bytes_modelo_fp32}  ratio={metricas['ratio_compresion_vs_wav']:.2f}x  "
          f"bitrate={metricas['bitrate_kbps']:.1f} kbps", flush=True)
    for esq in ("fp16mix", "fp16full"):
        b = bloque_cuant[esq]
        print(f"  [{esq}] ratio={b['ratio_compresion_vs_wav']:.2f}x  "
              f"SNR_ST={b['snr_stereo_db']:.2f}  SI-SDR_ST={b['si_sdr_stereo_db']:.2f} dB", flush=True)

    # === curvas + logs + checkpoints ===
    guardar_curva(res1["historial_loss"], f"loss {nom1} - {nombre_exp}", "loss",
                  salida / f"loss_curve_{nom1}.png")
    guardar_curva(res2["historial_loss"], f"loss {nom2} - {nombre_exp}", "loss",
                  salida / f"loss_curve_{nom2}.png")
    guardar_log_csv(salida / f"log_entrenamiento_{nom1}.csv",
                    res1["historial_loss"], res1["tiempos"])
    guardar_log_csv(salida / f"log_entrenamiento_{nom2}.csv",
                    res2["historial_loss"], res2["tiempos"])

    for nom, res in ((nom1, res1), (nom2, res2)):
        torch.save({"state_dict_coefs": res["modelo"].state_dict_coefs(),
                    "config": config, "metricas_canal": res["metricas"],
                    "canal": nom}, salida / "checkpoints" / f"checkpoint_final_{nom}.pt")

    print(f"\nlisto. resultados en: {salida}", flush=True)


if __name__ == "__main__":
    main()
