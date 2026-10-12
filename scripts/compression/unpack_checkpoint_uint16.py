import argparse
from pathlib import Path
import torch


def find_sd(ckpt):
    if isinstance(ckpt, dict) and "state_dict_coefs" in ckpt:
        return ckpt["state_dict_coefs"]
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError("No encontre state_dict.")


def dtype_from_string(s):
    s = str(s)
    if "float16" in s:
        return torch.float16
    if "float64" in s:
        return torch.float64
    return torch.float32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pkg", required=True)
    ap.add_argument("--out_ckpt", required=True)
    ap.add_argument("--out_float", choices=["fp32", "fp16", "original"], default="fp32")
    args = ap.parse_args()

    in_path = Path(args.in_pkg)
    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pkg = torch.load(in_path, map_location="cpu")

    if pkg.get("format") != "gs2d_uint16_affine_package_v1":
        raise RuntimeError("Formato no reconocido.")

    ckpt = pkg["ckpt_skeleton"]
    sd = find_sd(ckpt)

    for name, info in pkg["quantized"].items():
        q = info["q"].cpu()
        mn = float(info["min"])
        scale = float(info["scale"])

        x = q.float() * scale + mn

        if args.out_float == "fp16":
            x = x.half()
        elif args.out_float == "original":
            x = x.to(dtype_from_string(info.get("original_dtype", "torch.float32")))
        else:
            x = x.float()

        sd[name] = x.contiguous()
        print(f"DEQUANT {name:15s} shape={tuple(x.shape)} dtype={x.dtype}")

    for name, info in pkg.get("omitted_zeros", {}).items():
        dtype = torch.float32 if args.out_float == "fp32" else dtype_from_string(info.get("dtype", "torch.float32"))
        if args.out_float == "fp16":
            dtype = torch.float16

        sd[name] = torch.zeros(tuple(info["shape"]), dtype=dtype)
        print(f"RESTORE ZERO {name:15s} shape={tuple(info['shape'])} dtype={dtype}")

    torch.save(ckpt, out_path)

    print("")
    print("=== checkpoint dequantizado creado ===")
    print(f"entrada package: {in_path}")
    print(f"salida ckpt    : {out_path}")
    print(f"MB package     : {in_path.stat().st_size / 1024 / 1024:.2f}")
    print(f"MB ckpt salida : {out_path.stat().st_size / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
