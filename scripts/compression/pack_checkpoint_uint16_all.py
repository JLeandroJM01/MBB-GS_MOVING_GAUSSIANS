import argparse
from pathlib import Path
import torch


def quant_uint16_affine(x):
    x = x.detach().cpu().float().contiguous()
    mn = float(x.min().item())
    mx = float(x.max().item())

    if mx == mn:
        q = torch.zeros_like(x, dtype=torch.uint16)
        scale = 0.0
    else:
        scale = (mx - mn) / 65535.0
        q = torch.round((x - mn) / scale).clamp(0, 65535).to(torch.uint16)

    return {
        "q": q,
        "min": mn,
        "scale": scale,
        "shape": list(x.shape),
        "dtype_original": str(x.dtype),
        "bits": 16,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ckpt", required=True)
    ap.add_argument("--out_pkg", required=True)
    ap.add_argument("--omit_zero_depth_high", action="store_true")
    ap.add_argument("--zero_eps", type=float, default=1e-12)
    args = ap.parse_args()

    in_path = Path(args.in_ckpt)
    out_path = Path(args.out_pkg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(in_path, map_location="cpu")

    if "state_dict_coefs" not in ckpt:
        raise RuntimeError("El checkpoint no tiene state_dict_coefs.")

    sd = ckpt["state_dict_coefs"]

    skeleton_sd = {}
    quantized = {}
    omitted_zeros = {}

    print("=== PACK UINT16 ALL ===")
    print(f"entrada: {in_path}")
    print(f"salida : {out_path}")

    for k, v in sd.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            if args.omit_zero_depth_high and k == "depth_high":
                max_abs = float(v.detach().cpu().float().abs().max().item())
                if max_abs <= args.zero_eps:
                    omitted_zeros[k] = {
                        "shape": list(v.shape),
                        "dtype": str(v.dtype),
                    }
                    print(f"OMIT ZERO {k:15s} shape={tuple(v.shape)}")
                    continue

            quantized[k] = quant_uint16_affine(v)
            q = quantized[k]["q"]
            mb = q.numel() * q.element_size() / 1024 / 1024
            print(f"UINT16 {k:15s} shape={tuple(v.shape)} MB_q={mb:.4f}")

        else:
            skeleton_sd[k] = v

    pkg = {
        "format": "gs2d_uint16_all_affine_package_v1",
        "source_checkpoint": str(in_path),
        "ckpt_skeleton": {
            "state_dict_coefs": skeleton_sd,
            "config": ckpt.get("config", {}),
        },
        "quantized": quantized,
        "omitted_zeros": omitted_zeros,
    }

    torch.save(pkg, out_path)

    print("")
    print("=== package UINT16 ALL creado ===")
    print(f"MB entrada: {in_path.stat().st_size / 1024 / 1024:.2f}")
    print(f"MB salida : {out_path.stat().st_size / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
