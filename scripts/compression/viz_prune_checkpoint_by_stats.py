"""
Crea un checkpoint nuevo quitando gaussianas sin reentrenar.

Puede quitar:
  - IDs desde un CSV con columna id.
  - Gaussianas casi no-op detectadas desde gaussian_stats.csv.

Ejemplo: quitar top_estaticas_visibles.csv
python scripts/compression/viz_prune_checkpoint_by_stats.py ^
  --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt ^
  --ids_csv outputs\EXP\viz_stats\top_estaticas_visibles.csv ^
  --modo remove_ids ^
  --salida outputs\EXP\checkpoints\checkpoint_sin_estaticas_top.pt

Ejemplo: quitar TODAS las no-op por regla
python scripts/compression/viz_prune_checkpoint_by_stats.py ^
  --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt ^
  --stats_csv outputs\EXP\viz_stats\gaussian_stats.csv ^
  --modo remove_static ^
  --static_path_max 1.0 ^
  --static_color_max 0.01 ^
  --static_opstd_max 0.005 ^
  --static_scalestd_max 0.01 ^
  --salida outputs\EXP\checkpoints\checkpoint_sin_static_rule.pt
"""

import argparse
import csv
from pathlib import Path

import torch


def torch_load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


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
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            op_mean = float(row["op_mean"])
            path_len = float(row["path_length_px"])
            color = float(row["color_path"])
            opstd = float(row["op_std"])
            scalestd = float(row["scale_std"])

            if (
                op_mean >= float(args.static_opmean_min)
                and path_len <= float(args.static_path_max)
                and color <= float(args.static_color_max)
                and opstd <= float(args.static_opstd_max)
                and scalestd <= float(args.static_scalestd_max)
            ):
                ids.append(int(row["id"]))

    if args.max_ids is not None:
        ids = ids[:int(args.max_ids)]
    return ids


def filtrar_state_dict(sd, keep):
    keep_t = torch.as_tensor(keep, dtype=torch.long)
    out = {}

    for k, v in sd.items():
        if torch.is_tensor(v) and v.ndim >= 1 and k.endswith(("_a0", "_high")):
            out[k] = v.index_select(0, keep_t).clone()
        else:
            out[k] = v

    out["N"] = len(keep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--modo", choices=["remove_ids", "keep_ids", "remove_static", "keep_static"], required=True)

    ap.add_argument("--ids_csv", default=None)
    ap.add_argument("--stats_csv", default=None)
    ap.add_argument("--max_ids", type=int, default=None)

    ap.add_argument("--static_opmean_min", type=float, default=0.05)
    ap.add_argument("--static_path_max", type=float, default=1.0)
    ap.add_argument("--static_color_max", type=float, default=0.01)
    ap.add_argument("--static_opstd_max", type=float, default=0.005)
    ap.add_argument("--static_scalestd_max", type=float, default=0.01)

    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    ckpt = torch_load(ckpt_path)
    if "state_dict_coefs" not in ckpt:
        raise RuntimeError("Checkpoint sin state_dict_coefs")

    sd = ckpt["state_dict_coefs"]
    N = int(sd.get("N", sd["mu_a0"].shape[0]))

    if args.modo in ("remove_static", "keep_static"):
        if not args.stats_csv:
            raise RuntimeError("remove_static/keep_static requiere --stats_csv")
        ids = leer_static_ids_desde_stats(args.stats_csv, args)
    else:
        if not args.ids_csv:
            raise RuntimeError("remove_ids/keep_ids requiere --ids_csv")
        ids = leer_ids_csv(args.ids_csv, max_ids=args.max_ids)

    ids_set = set(int(i) for i in ids if 0 <= int(i) < N)

    if args.modo in ("keep_ids", "keep_static"):
        keep = sorted(ids_set)
    else:
        keep = [i for i in range(N) if i not in ids_set]

    if not keep:
        raise RuntimeError("Resultado dejaria 0 gaussianas; revisa la seleccion.")

    new_sd = filtrar_state_dict(sd, keep)
    ckpt_out = dict(ckpt)
    ckpt_out["state_dict_coefs"] = new_sd

    config = dict(ckpt_out.get("config", {}))
    config["n_gaussianas_inicial"] = len(keep)
    config["prune_origen_checkpoint"] = str(ckpt_path)
    config["prune_modo"] = args.modo
    config["prune_ids_seleccionados"] = len(ids_set)
    config["prune_N_original"] = N
    config["prune_N_final"] = len(keep)
    ckpt_out["config"] = config

    salida = Path(args.salida).resolve()
    salida.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt_out, salida)

    txt = salida.with_suffix(".txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("checkpoint pruneado\n")
        f.write("===================\n")
        f.write(f"origen={ckpt_path}\n")
        f.write(f"salida={salida}\n")
        f.write(f"modo={args.modo}\n")
        f.write(f"N_original={N}\n")
        f.write(f"ids_seleccionados={len(ids_set)}\n")
        f.write(f"N_final={len(keep)}\n")

    print("=== listo ===", flush=True)
    print(f"N_original={N}", flush=True)
    print(f"ids_seleccionados={len(ids_set)}", flush=True)
    print(f"N_final={len(keep)}", flush=True)
    print(f"checkpoint: {salida}", flush=True)
    print(f"resumen   : {txt}", flush=True)


if __name__ == "__main__":
    main()
