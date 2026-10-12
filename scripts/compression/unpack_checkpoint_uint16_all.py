import argparse
from pathlib import Path
import torch


def dequant(info):
    q = info["q"].detach().cpu()
    mn = float(info["min"])
    scale = float(info["scale"])

    x = q.float() * scale + mn
    return x.float().contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pkg", required=True)
    ap.add_argument("--out_ckpt", required=True)
    args = ap.parse_args()

    in_path = Path(args.in_pkg)
    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pkg = torch.load(in_path, map_location="cpu")

    if pkg.get("format") != "gs2d_uint16_all_affine_package_v1":
        raise RuntimeError(f"Formato no reconocido: {pkg.get('format')}")

    ckpt_skeleton = pkg["ckpt_skeleton"]
    sd = dict(ckpt_skeleton["state_dict_coefs"])

    print("=== UNPACK UINT16 ALL ===")
    print(f"entrada: {in_path}")
    print(f"salida : {out_path}")

    for k, info in pkg["quantized"].items():
        sd[k] = dequant(info)
        print(f"DEQUANT {k:15s} shape={tuple(sd[k].shape)}")

    for k, info in pkg.get("omitted_zeros", {}).items():
        sd[k] = torch.zeros(tuple(info["shape"]), dtype=torch.float32)
        print(f"RESTORE ZERO {k:15s} shape={tuple(sd[k].shape)}")

    out_ckpt = {
        "config": ckpt_skeleton["config"],
        "state_dict_coefs": sd,
    }

    torch.save(out_ckpt, out_path)

    print("")
    print("=== checkpoint reconstruido ===")
    print(f"MB package     : {in_path.stat().st_size / 1024 / 1024:.2f}")
    print(f"MB ckpt salida : {out_path.stat().st_size / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
