"""
Renderiza un checkpoint usando SOLO ciertas gaussianas o EXCLUYENDO ciertas gaussianas.

Sirve para:
  1) Ver que aportan las gaussianas estaticas/no-op.
  2) Ver que pasa si las quitas.
  3) Renderizar top_movimiento/top_color/top_opacidad.
  4) Probar pruning sin reentrenar.

Requiere el paquete instalado (pip install -e .).

Ejemplos Windows:

# A) Renderizar SOLO las gaussianas estaticas visibles del CSV top
python scripts/visualization/viz_render_subset.py ^
  --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt ^
  --ids_csv outputs\EXP\viz_stats\top_estaticas_visibles.csv ^
  --modo only ^
  --salida outputs\EXP\ablacion\solo_estaticas_top ^
  --crear_video --fps 30 --fin 120

# B) Renderizar quitando esas gaussianas
python scripts/visualization/viz_render_subset.py ^
  --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt ^
  --ids_csv outputs\EXP\viz_stats\top_estaticas_visibles.csv ^
  --modo exclude ^
  --salida outputs\EXP\ablacion\sin_estaticas_top ^
  --crear_video --fps 30 --fin 120

# C) Usar gaussian_stats.csv y quitar TODAS las gaussianas casi no-op
python scripts/visualization/viz_render_subset.py ^
  --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt ^
  --stats_csv outputs\EXP\viz_stats\gaussian_stats.csv ^
  --modo exclude_static ^
  --salida outputs\EXP\ablacion\sin_static_rule ^
  --static_path_max 1.0 ^
  --static_color_max 0.01 ^
  --static_opstd_max 0.005 ^
  --static_scalestd_max 0.01 ^
  --crear_video --fps 30

Notas:
  - --modo only: renderiza solo ids seleccionados.
  - --modo exclude: renderiza todo menos ids seleccionados.
  - --modo only_static: selecciona no-op por regla desde gaussian_stats.csv y renderiza solo esas.
  - --modo exclude_static: selecciona no-op por regla desde gaussian_stats.csv y las quita.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image

RAIZ = Path(__file__).resolve().parents[2]

from gs2d_video.io.checkpoints import cargar_modelo_desde_checkpoint, cargar_frame_fondo
from gs2d_video.render.renderer import render_frame


def leer_ids_csv(path, max_ids=None):
    ids = []
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "id" not in (r.fieldnames or []):
            raise RuntimeError(f"El CSV no tiene columna id: {path}")
        for row in r:
            ids.append(int(row["id"]))
            if max_ids is not None and len(ids) >= int(max_ids):
                break
    return ids


def leer_static_ids_desde_stats(path, args):
    ids = []
    total = 0
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        needed = ["id", "path_length_px", "color_path", "op_std", "scale_std", "op_mean"]
        faltan = [c for c in needed if c not in (r.fieldnames or [])]
        if faltan:
            raise RuntimeError(f"Faltan columnas en stats_csv: {faltan}")

        for row in r:
            total += 1
            op_mean = float(row["op_mean"])
            path = float(row["path_length_px"])
            color = float(row["color_path"])
            opstd = float(row["op_std"])
            scalestd = float(row["scale_std"])

            es_static = (
                op_mean >= float(args.static_opmean_min)
                and path <= float(args.static_path_max)
                and color <= float(args.static_color_max)
                and opstd <= float(args.static_opstd_max)
                and scalestd <= float(args.static_scalestd_max)
            )
            if es_static:
                ids.append(int(row["id"]))

    if args.max_ids is not None:
        ids = ids[:int(args.max_ids)]

    print(f"[static rule] stats leidas={total}  ids_static={len(ids)}", flush=True)
    return ids


def filtrar_modelo_inplace(modelo, indices_keep):
    indices_keep = torch.as_tensor(indices_keep, device=modelo.mu_a0.device, dtype=torch.long)
    for nombre in ["mu", "opacity", "color", "scale", "theta", "depth"]:
        for sufijo in ["_a0", "_high"]:
            attr = nombre + sufijo
            t = getattr(modelo, attr).data
            setattr(modelo, attr, nn.Parameter(t.index_select(0, indices_keep).clone()))
    modelo.N = int(indices_keep.numel())


def construir_indices(N, ids, modo):
    ids_set = set(int(i) for i in ids if 0 <= int(i) < N)
    if modo in ("only", "only_static"):
        keep = sorted(ids_set)
    elif modo in ("exclude", "exclude_static"):
        keep = [i for i in range(N) if i not in ids_set]
    else:
        raise ValueError(f"modo no soportado: {modo}")
    return keep, sorted(ids_set)


def imagen_render_a_uint8(render):
    render = torch.nan_to_num(render, nan=0.0, posinf=1.0, neginf=0.0)
    render = render.clamp(0, 1)
    return (render.detach().cpu().numpy() * 255).astype(np.uint8)


def calcular_metricas_simples(img_u8, gt_tensor):
    if gt_tensor is None:
        return None
    gt = gt_tensor.numpy().astype(np.float32)
    pred = img_u8.astype(np.float32) / 255.0
    if pred.shape != gt.shape:
        return None
    mse = float(np.mean((pred - gt) ** 2))
    mae = float(np.mean(np.abs(pred - gt)))
    psnr = float("inf") if mse <= 0 else float(-10.0 * np.log10(mse))
    return mse, mae, psnr


def crear_video(frames_dir, salida_mp4, fps):
    script_video = RAIZ / "scripts" / "data" / "frames_a_video.py"
    cmd = [
        sys.executable,
        str(script_video),
        "--frames", str(frames_dir),
        "--salida", str(salida_mp4),
        "--fps", str(int(fps)),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(RAIZ), check=True)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--modo", choices=["only", "exclude", "only_static", "exclude_static"], required=True)

    ap.add_argument("--ids_csv", default=None, help="CSV con columna id")
    ap.add_argument("--stats_csv", default=None, help="gaussian_stats.csv para modos *_static")
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--max_ids", type=int, default=None)

    ap.add_argument("--inicio", type=int, default=0)
    ap.add_argument("--fin", type=int, default=None)
    ap.add_argument("--paso", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--crear_video", action="store_true")
    ap.add_argument("--fps", type=int, default=30)

    # Regla static/no-op.
    ap.add_argument("--static_opmean_min", type=float, default=0.05)
    ap.add_argument("--static_path_max", type=float, default=1.0)
    ap.add_argument("--static_color_max", type=float, default=0.01)
    ap.add_argument("--static_opstd_max", type=float, default=0.005)
    ap.add_argument("--static_scalestd_max", type=float, default=0.01)

    args = ap.parse_args()

    salida = Path(args.salida).resolve()
    frames_dir = salida / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    modelo, config, matrices_base, info = cargar_modelo_desde_checkpoint(args.checkpoint, device=args.device)
    N = info["N"]
    T = info["n_frames"]
    H = info["H"]
    W = info["W"]

    # Leer IDs.
    if args.modo in ("only_static", "exclude_static"):
        if not args.stats_csv:
            raise RuntimeError("Los modos only_static/exclude_static requieren --stats_csv gaussian_stats.csv")
        ids = leer_static_ids_desde_stats(args.stats_csv, args)
    else:
        ids = []
        if args.ids_csv:
            ids.extend(leer_ids_csv(args.ids_csv, max_ids=args.max_ids))
        if args.ids:
            ids.extend(args.ids)
        if not ids:
            raise RuntimeError("Necesitas --ids_csv o --ids para modo only/exclude")

    keep, selected = construir_indices(N, ids, args.modo)
    if not keep:
        raise RuntimeError("La seleccion dejo 0 gaussianas para renderizar.")

    print("=== viz_render_subset ===", flush=True)
    print(f"checkpoint       : {Path(args.checkpoint).resolve()}", flush=True)
    print(f"salida           : {salida}", flush=True)
    print(f"modo             : {args.modo}", flush=True)
    print(f"N original       : {N}", flush=True)
    print(f"ids seleccionados: {len(selected)}", flush=True)
    print(f"N render         : {len(keep)}", flush=True)
    print(f"T                : {T}", flush=True)

    # Guardar ids usados.
    with open(salida / "ids_seleccionados.txt", "w", encoding="utf-8") as f:
        for i in selected:
            f.write(str(i) + "\n")
    with open(salida / "ids_keep.txt", "w", encoding="utf-8") as f:
        for i in keep:
            f.write(str(i) + "\n")

    filtrar_modelo_inplace(modelo, keep)
    print(f"modelo filtrado  : N={modelo.N}", flush=True)

    inicio = max(0, int(args.inicio))
    fin = T if args.fin is None else min(int(args.fin), T)
    paso = max(1, int(args.paso))

    ruta_csv = salida / "metricas_subset.csv"
    psnrs = []
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame", "mse", "mae", "psnr"])

        out_idx = 0
        for j in range(inicio, fin, paso):
            params_j = modelo.evaluar_en_frame(j, matrices_base)
            render = render_frame(params_j, H, W, config)
            img = imagen_render_a_uint8(render)

            # Numeracion compacta para video.
            ruta_png = frames_dir / f"frame_{out_idx:04d}.png"
            Image.fromarray(img).save(ruta_png)

            gt = cargar_frame_fondo(config, j, H=H, W=W)
            met = calcular_metricas_simples(img, gt)
            if met is not None:
                mse, mae, psnr = met
                psnrs.append(psnr)
                wcsv.writerow([j, f"{mse:.8f}", f"{mae:.8f}", f"{psnr:.4f}"])
            else:
                wcsv.writerow([j, "", "", ""])

            if out_idx == 0 or (out_idx + 1) % 25 == 0:
                msg = f"  render frame_original={j} frame_out={out_idx}"
                if psnrs:
                    msg += f" PSNR_prom={float(np.mean(psnrs)):.2f}"
                print(msg, flush=True)

            del params_j, render
            if str(args.device).startswith("cuda") and (out_idx + 1) % 25 == 0:
                torch.cuda.empty_cache()

            out_idx += 1

    with open(salida / "resumen_subset.txt", "w", encoding="utf-8") as f:
        f.write("viz_render_subset\n")
        f.write("=================\n")
        f.write(f"modo={args.modo}\n")
        f.write(f"N_original={N}\n")
        f.write(f"ids_seleccionados={len(selected)}\n")
        f.write(f"N_render={len(keep)}\n")
        f.write(f"inicio={inicio}\nfin={fin}\npaso={paso}\n")
        if psnrs:
            f.write(f"PSNR_prom={float(np.mean(psnrs)):.4f}\n")
            f.write(f"PSNR_min={float(np.min(psnrs)):.4f}\n")

    print(f"metricas: {ruta_csv}", flush=True)
    if psnrs:
        print(f"PSNR_prom={float(np.mean(psnrs)):.2f}", flush=True)

    if args.crear_video:
        salida_mp4 = salida / "video_subset.mp4"
        crear_video(frames_dir, salida_mp4, args.fps)
        print(f"video: {salida_mp4}", flush=True)

    print("=== listo ===", flush=True)


if __name__ == "__main__":
    main()
