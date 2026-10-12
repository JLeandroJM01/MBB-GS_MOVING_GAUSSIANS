import argparse
from pathlib import Path
import torch


DEFAULT_QUANT = ["mu_high", "color_high", "opacity_high", "scale_high"]


def find_sd(ckpt):
    if isinstance(ckpt, dict) and "state_dict_coefs" in ckpt:
        return ckpt["state_dict_coefs"]
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError("No encontre state_dict.")


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

    return q, mn, scale, list(x.shape)


def convert_other_floats(obj, mode):
    if torch.is_tensor(obj):
        if obj.is_floating_point() and mode == "fp16":
            return obj.half()
        return obj

    if isinstance(obj, dict):
        return {k: convert_other_floats(v, mode) for k, v in obj.items()}

    if isinstance(obj, list):
        return [convert_other_floats(v, mode) for v in obj]

    if isinstance(obj, tuple):
        return tuple(convert_other_floats(v, mode) for v in obj)

    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ckpt", required=True)
    ap.add_argument("--out_pkg", required=True)
    ap.add_argument("--other_float", choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--quant_tensors", nargs="*", default=DEFAULT_QUANT)
    ap.add_argument("--omit_zero_depth_high", action="store_true")
    ap.add_argument("--zero_eps", type=float, default=1e-12)
    args = ap.parse_args()

    in_path = Path(args.in_ckpt)
    out_path = Path(args.out_pkg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(in_path, map_location="cpu")
    sd = find_sd(ckpt)

    quantized = {}
    omitted_zeros = {}

    for name in args.quant_tensors:
        if name not in sd:
            print(f"[WARN] no existe tensor: {name}")
            continue

        if not torch.is_tensor(sd[name]):
            print(f"[WARN] no es tensor: {name}")
            continue

        if not sd[name].is_floating_point():
            print(f"[WARN] no es float: {name}")
            continue

        x = sd.pop(name)
        q, mn, scale, shape = quant_uint16_affine(x)

        quantized[name] = {
            "q": q,
            "min": mn,
            "scale": scale,
            "shape": shape,
            "original_dtype": str(x.dtype),
        }

        print(f"UINT16 {name:15s} shape={shape} min={mn:.8g} max={float(x.max().item()):.8g} scale={scale:.8g}")

    if args.omit_zero_depth_high and "depth_high" in sd and torch.is_tensor(sd["depth_high"]):
        x = sd["depth_high"].detach().cpu()
        max_abs = float(x.float().abs().max().item())

        if max_abs <= args.zero_eps:
            omitted_zeros["depth_high"] = {
                "shape": list(x.shape),
                "dtype": str(x.dtype),
            }
            sd.pop("depth_high")
            print(f"OMIT depth_high porque max_abs={max_abs:.3e}")
        else:
            print(f"[WARN] depth_high no es cero. max_abs={max_abs:.3e}. No se omite.")

    if args.other_float == "fp16":
        ckpt = convert_other_floats(ckpt, "fp16")

    pkg = {
        "format": "gs2d_uint16_affine_package_v1",
        "source_checkpoint": str(in_path),
        "other_float": args.other_float,
        "ckpt_skeleton": ckpt,
        "quantized": quantized,
        "omitted_zeros": omitted_zeros,
    }

    torch.save(pkg, out_path)

    print("")
    print("=== paquete UINT16 creado ===")
    print(f"entrada: {in_path}")
    print(f"salida : {out_path}")
    print(f"modo otros floats: {args.other_float}")
    print(f"MB entrada: {in_path.stat().st_size / 1024 / 1024:.2f}")
    print(f"MB paquete: {out_path.stat().st_size / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
