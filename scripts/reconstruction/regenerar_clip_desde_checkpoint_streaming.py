"""
Regenera todos los frames desde un checkpoint, en modo streaming.

Objetivo:
- No apila todos los renders en GPU.
- Guarda cada frame PNG apenas se renderiza.
- Opcionalmente genera comparaciones original | render | diff.
- Opcionalmente llama a scripts/data/frames_a_video.py para crear MP4.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

RAIZ = Path(__file__).resolve().parents[2]

from gs2d_video.io.checkpoints import cargar_modelo_desde_checkpoint
from gs2d_video.render.renderer import render_frame


def listar_frames_clip(carpeta_clip):
    if not carpeta_clip.is_dir():
        return []
    return sorted(
        p for p in carpeta_clip.iterdir()
        if p.name.startswith("frame_") and p.suffix.lower() == ".png"
    )


def imagen_render_a_uint8(render):
    render = torch.nan_to_num(render, nan=0.0, posinf=1.0, neginf=0.0)
    render = render.clamp(0, 1)
    return (render.detach().cpu().numpy() * 255).astype(np.uint8)


def guardar_comparacion(render_uint8, ruta_original, ruta_salida, factor_diff=5.0):
    target = Image.open(ruta_original).convert("RGB")
    target_np = np.asarray(target, dtype=np.float32) / 255.0
    pred_np = render_uint8.astype(np.float32) / 255.0
    if target_np.shape != pred_np.shape:
        raise RuntimeError(
            f"Shape original y render no coinciden: original={target_np.shape}, render={pred_np.shape}"
        )
    diff = np.clip(np.abs(target_np - pred_np) * float(factor_diff), 0, 1)
    concat = np.concatenate([target_np, pred_np, diff], axis=1)
    Image.fromarray((concat * 255).astype(np.uint8)).save(ruta_salida)


def calcular_metricas_simples(render_uint8, ruta_original):
    target = Image.open(ruta_original).convert("RGB")
    target_np = np.asarray(target, dtype=np.float32) / 255.0
    pred_np = render_uint8.astype(np.float32) / 255.0
    if target_np.shape != pred_np.shape:
        return None
    mse = float(np.mean((pred_np - target_np) ** 2))
    mae = float(np.mean(np.abs(pred_np - target_np)))
    psnr = float("inf") if mse <= 0 else float(-10.0 * np.log10(mse))
    return mse, mae, psnr


@torch.no_grad()
def regenerar_frames_streaming(
    checkpoint,
    salida,
    clip_override=None,
    device_str=None,
    inicio=0,
    fin=None,
    guardar_comparaciones=False,
    factor_diff=5.0,
    crear_video=False,
    fps=27,
    limpiar_cache_cada=25,
):
    checkpoint = Path(checkpoint).resolve()
    salida = Path(salida).resolve()
    salida.mkdir(parents=True, exist_ok=True)

    device_str = device_str or "cuda"

    modelo, config, matrices_base, info = cargar_modelo_desde_checkpoint(
        checkpoint,
        device=device_str,
        clip_override=clip_override,
    )

    device = modelo.mu_a0.device
    grados = info["grados"]
    n_frames_ckpt = int(info["n_frames"])
    H = int(info["H"])
    W = int(info["W"])
    n_gaussianas = int(info["N"])
    base_temporal = info["base_temporal"]

    clip = config["clip"]
    carpeta_clip = RAIZ / "data" / "clips" / clip
    frames_originales = listar_frames_clip(carpeta_clip)

    if frames_originales:
        n_frames_clip = len(frames_originales)
        n_frames = min(n_frames_ckpt, n_frames_clip)
    else:
        n_frames_clip = 0
        n_frames = n_frames_ckpt

    inicio = int(inicio)
    fin = n_frames if fin is None else min(int(fin), n_frames)
    if inicio < 0 or inicio >= n_frames:
        raise ValueError(f"inicio fuera de rango: {inicio}, n_frames={n_frames}")
    if fin <= inicio:
        raise ValueError(f"fin debe ser mayor que inicio. inicio={inicio}, fin={fin}")

    print("=== regenerar desde checkpoint ===", flush=True)
    print(f"checkpoint : {checkpoint}", flush=True)
    print(f"salida     : {salida}", flush=True)
    print(f"clip       : {clip}", flush=True)
    print(f"carpeta clip: {carpeta_clip}", flush=True)
    print(f"device     : {device}", flush=True)
    print(f"N          : {n_gaussianas}", flush=True)
    print(f"frames ckpt: {n_frames_ckpt}", flush=True)
    print(f"frames clip: {n_frames_clip}", flush=True)
    print(f"render     : {inicio}..{fin - 1}", flush=True)
    print(f"resolucion : {H}x{W}", flush=True)
    print(f"grados     : {grados}", flush=True)
    print(f"base       : {base_temporal}", flush=True)


    carpeta_comp = salida.parent / "comparaciones_streaming"
    if guardar_comparaciones:
        carpeta_comp.mkdir(parents=True, exist_ok=True)

    ruta_csv = salida.parent / "metricas_streaming_simples.csv"
    psnr_vals, mae_vals, mse_vals = [], [], []

    with open(ruta_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["frame", "mse", "mae", "psnr"])

        for j in range(inicio, fin):
            params_j = modelo.evaluar_en_frame(j, matrices_base)
            render = render_frame(params_j, H, W, config)
            img = imagen_render_a_uint8(render)

            ruta_frame = salida / f"frame_{j:04d}.png"
            Image.fromarray(img).save(ruta_frame)

            if frames_originales and j < len(frames_originales):
                met = calcular_metricas_simples(img, frames_originales[j])
                if met is not None:
                    mse, mae, psnr = met
                    mse_vals.append(mse)
                    mae_vals.append(mae)
                    psnr_vals.append(psnr)
                    writer.writerow([j, f"{mse:.8f}", f"{mae:.8f}", f"{psnr:.4f}"])
                else:
                    writer.writerow([j, "", "", ""])

                if guardar_comparaciones:
                    ruta_comp = carpeta_comp / f"comparacion_{j:04d}.png"
                    guardar_comparacion(img, frames_originales[j], ruta_comp, factor_diff=factor_diff)
            else:
                writer.writerow([j, "", "", ""])

            del params_j, render
            if device.type == "cuda" and limpiar_cache_cada > 0 and ((j + 1) % limpiar_cache_cada == 0):
                torch.cuda.empty_cache()

            if (j == inicio) or ((j + 1 - inicio) % 10 == 0) or (j == fin - 1):
                msg = f"render {j + 1}/{fin}"
                if psnr_vals:
                    msg += f"  PSNR_prom={float(np.mean(psnr_vals)):.2f}"
                print(msg, flush=True)

    print("=== listo frames ===", flush=True)
    print(f"frames guardados en: {salida}", flush=True)
    print(f"metricas simples en: {ruta_csv}", flush=True)
    if psnr_vals:
        print(
            f"PSNR promedio simple={float(np.mean(psnr_vals)):.2f}  "
            f"MAE promedio={float(np.mean(mae_vals)):.6f}",
            flush=True,
        )

    if crear_video:
        ruta_video = salida.parent / "video_reconstruido.mp4"
        script_video = RAIZ / "scripts" / "data" / "frames_a_video.py"
        cmd = [
            sys.executable,
            str(script_video),
            "--frames", str(salida),
            "--salida", str(ruta_video),
            "--fps", str(int(fps)),
        ]
        print("=== creando video ===", flush=True)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(RAIZ), check=True)
        print(f"video guardado en: {ruta_video}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--salida", required=True)
    parser.add_argument("--clip", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--inicio", type=int, default=0)
    parser.add_argument("--fin", type=int, default=None)
    parser.add_argument("--crear_video", action="store_true")
    parser.add_argument("--fps", type=int, default=27)
    parser.add_argument("--comparaciones", action="store_true")
    parser.add_argument("--factor_diff", type=float, default=5.0)
    parser.add_argument("--limpiar_cache_cada", type=int, default=25)
    args = parser.parse_args()

    regenerar_frames_streaming(
        checkpoint=args.checkpoint,
        salida=args.salida,
        clip_override=args.clip,
        device_str=args.device,
        inicio=args.inicio,
        fin=args.fin,
        guardar_comparaciones=args.comparaciones,
        factor_diff=args.factor_diff,
        crear_video=args.crear_video,
        fps=args.fps,
        limpiar_cache_cada=args.limpiar_cache_cada,
    )


if __name__ == "__main__":
    main()
