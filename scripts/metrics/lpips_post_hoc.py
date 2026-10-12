"""
Calcula LPIPS post-hoc de un experimento que ya termino pero donde LPIPS quedo en None
(tipico cuando el nodo de compute no tenia internet para bajar AlexNet).

Lee:
  - <exp>/frames_renderizados/frame_NNNN.png    (renders del modelo)
  - data/clips/<clip>/frame_NNNN.png            (originales, segun config_usada.json)

Actualiza in-place:
  - <exp>/metricas.json (campos lpips_* y lpips_por_frame en post_pruning)
  - <exp>/metricas_por_frame.csv (columna lpips)

Uso:
    python scripts/metrics/lpips_post_hoc.py --exp outputs_khipu/fase1_motion
    python scripts/metrics/lpips_post_hoc.py --exp outputs_khipu/fase1_motion \\
                                     --renders /ruta/a/frames_renderizados \\
                                     --gt data/clips/video_clips

Por defecto usa CPU (en Mac). Si tienes CUDA, agrega --device cuda.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def calcular_lpips_por_frame(fn_lpips, renders_paths, gt_paths, device):
    """Devuelve lista de LPIPS por frame, mismo largo que renders_paths."""
    valores = []
    n = len(renders_paths)
    for j, (rp, gp) in enumerate(zip(renders_paths, gt_paths)):
        r = np.asarray(Image.open(rp).convert("RGB"), dtype=np.float32) / 255.0
        g = np.asarray(Image.open(gp).convert("RGB"), dtype=np.float32) / 255.0

        if r.shape != g.shape:
            raise RuntimeError(
                f"shape mismatch frame {j}: render={r.shape} gt={g.shape}\n"
                f"  render: {rp}\n  gt: {gp}"
            )

        # LPIPS espera tensores (1, 3, H, W) en [-1, 1].
        rt = torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        gt = torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

        with torch.no_grad():
            v = float(fn_lpips(rt, gt).item())

        valores.append(v)

        if (j == 0) or ((j + 1) % 50 == 0) or (j == n - 1):
            print(f"  frame {j + 1:4d}/{n}  lpips={v:.4f}", flush=True)

    return valores


def agregados_lpips(valores):
    if not valores:
        return {k: None for k in ("lpips_promedio", "lpips_min", "lpips_max", "lpips_p5", "lpips_std")}
    arr = np.asarray([v for v in valores if v is not None], dtype=np.float64)
    return {
        "lpips_promedio": float(arr.mean()),
        "lpips_min":      float(arr.min()),
        "lpips_max":      float(arr.max()),
        "lpips_p5":       float(np.percentile(arr, 5.0)),
        "lpips_std":      float(arr.std()),
    }


def actualizar_metricas_json(ruta, lpips_vals):
    """Inyecta lpips_por_frame y agregados en post_pruning. Hace backup."""
    with open(ruta, encoding="utf-8") as f:
        m = json.load(f)

    backup = ruta.with_name(ruta.stem + ".pre_lpips.json")
    if not backup.exists():
        with open(backup, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2, default=str)
        print(f"  backup guardado: {backup}", flush=True)

    post = m.setdefault("post_pruning", {})
    agg = post.setdefault("agregados", {})

    post["lpips_por_frame"] = lpips_vals
    agg.update(agregados_lpips(lpips_vals))

    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, default=str)
    print(f"  metricas.json actualizado: {ruta}", flush=True)


def actualizar_csv_por_frame(ruta_csv, lpips_vals):
    """Reescribe metricas_por_frame.csv inyectando la columna lpips actualizada."""
    if not ruta_csv.exists():
        print(f"  (no existe {ruta_csv}, omito)", flush=True)
        return

    with open(ruta_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        filas = list(reader)
        campos = reader.fieldnames or ["frame_idx", "psnr", "ssim", "lpips", "psnr_temporal"]

    if "lpips" not in campos:
        campos = list(campos) + ["lpips"]

    for j, fila in enumerate(filas):
        if j < len(lpips_vals) and lpips_vals[j] is not None:
            fila["lpips"] = f"{lpips_vals[j]:.6f}"
        else:
            fila["lpips"] = ""

    backup = ruta_csv.with_name(ruta_csv.stem + ".pre_lpips.csv")
    if not backup.exists():
        backup.write_text(ruta_csv.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  backup guardado: {backup}", flush=True)

    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for fila in filas:
            writer.writerow(fila)
    print(f"  metricas_por_frame.csv actualizado: {ruta_csv}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True,
                    help="ruta a la carpeta del experimento (p.ej. outputs_khipu/fase1_motion)")
    ap.add_argument("--renders", default=None,
                    help="override de la carpeta de frames renderizados "
                         "(default: <exp>/frames_renderizados)")
    ap.add_argument("--gt", default=None,
                    help="override de la carpeta de frames GT "
                         "(default: data/clips/<clip de config_usada.json>)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                    help="device para LPIPS (default cpu; cuda si tienes GPU)")
    ap.add_argument("--max_frames", type=int, default=None,
                    help="cap manual de frames a procesar (default: lo que diga config_usada.json)")
    args = ap.parse_args()

    raiz = Path(__file__).resolve().parents[2]
    exp = Path(args.exp).resolve()
    if not exp.is_dir():
        print(f"ERROR: no existe {exp}", file=sys.stderr)
        sys.exit(2)

    cfg_path = exp / "config_usada.json"
    if not cfg_path.is_file():
        print(f"ERROR: no existe {cfg_path}", file=sys.stderr)
        sys.exit(2)

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    clip = cfg["clip"]
    max_frames = int(args.max_frames or cfg.get("max_frames") or 0)

    renders_dir = Path(args.renders) if args.renders else (exp / "frames_renderizados")
    gt_dir = Path(args.gt) if args.gt else (raiz / "data" / "clips" / clip)

    if not renders_dir.is_dir():
        print(
            f"ERROR: no existe la carpeta de renders: {renders_dir}\n"
            f"  Si los moviste a Drive, descomprime/baja a esta ruta y vuelve a correr,\n"
            f"  o pasa --renders /ruta/donde/los/dejaste",
            file=sys.stderr,
        )
        sys.exit(3)
    if not gt_dir.is_dir():
        print(f"ERROR: no existe la carpeta de GT: {gt_dir}", file=sys.stderr)
        sys.exit(3)

    renders_paths = sorted(renders_dir.glob("frame_*.png"))
    gt_paths_all = sorted(gt_dir.glob("frame_*.png"))

    if max_frames > 0:
        renders_paths = renders_paths[:max_frames]
        gt_paths_all = gt_paths_all[:max_frames]

    if len(renders_paths) == 0:
        print(f"ERROR: 0 renders en {renders_dir}", file=sys.stderr)
        sys.exit(4)
    if len(renders_paths) != len(gt_paths_all):
        print(
            f"ERROR: cantidades distintas: {len(renders_paths)} renders vs {len(gt_paths_all)} GTs",
            file=sys.stderr,
        )
        sys.exit(4)

    print(f"=== LPIPS post-hoc ===", flush=True)
    print(f"  exp        : {exp}", flush=True)
    print(f"  clip       : {clip}", flush=True)
    print(f"  renders_dir: {renders_dir}", flush=True)
    print(f"  gt_dir     : {gt_dir}", flush=True)
    print(f"  n_frames   : {len(renders_paths)}", flush=True)
    print(f"  device     : {args.device}", flush=True)

    try:
        import lpips
    except Exception as e:
        print(f"ERROR: importar lpips fallo: {e}\n  Instala con: pip install lpips", file=sys.stderr)
        sys.exit(5)

    device = torch.device(args.device)
    fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    valores = calcular_lpips_por_frame(fn, renders_paths, gt_paths_all, device)

    actualizar_metricas_json(exp / "metricas.json", valores)
    actualizar_csv_por_frame(exp / "metricas_por_frame.csv", valores)

    agg = agregados_lpips(valores)
    print(f"\nLPIPS post-hoc:")
    for k, v in agg.items():
        print(f"  {k:20s} {v:.4f}" if v is not None else f"  {k:20s} None")


if __name__ == "__main__":
    main()
