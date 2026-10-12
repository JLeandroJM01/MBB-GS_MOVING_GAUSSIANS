"""
Carga centralizada de checkpoints generados por scripts/train.py:
    checkpoint_final.pt
    modelo_pruneado.pt

Reconstruye el objeto GaussianasPolinomial2D y las matrices de la base
temporal declarada en el checkpoint. No rasteriza.

Es la unica ruta para interpretar un checkpoint: los scripts de
reconstruccion, interpolacion, pruning y visualizacion la comparten, de modo
que la metadata (N, H, W, n_frames, grados, base temporal) se lee siempre
igual. Los checkpoints antiguos sin 'base_temporal' se interpretan como
Chebyshev.
"""

from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image

from gs2d_video.core.bases import construir_matriz_chebyshev, construir_matriz_monomial
from gs2d_video.core.modelo import GaussianasPolinomial2D


# src/gs2d_video/io/checkpoints.py -> io -> gs2d_video -> src -> raiz del repo.
# Se usa para resolver data/clips/<clip> al cargar frames de fondo.
RAIZ = Path(__file__).resolve().parents[3]


def _torch_load_seguro(path, map_location="cpu"):
    """
    PyTorch reciente puede quejarse con weights_only.
    Probamos con weights_only=False y hacemos fallback si la version no acepta.
    """
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def elegir_device(device_str="cpu"):
    if device_str is None:
        device_str = "cpu"

    device_str = str(device_str)

    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Pediste {device_str}, pero torch.cuda.is_available()==False"
        )

    return torch.device(device_str)


def inferir_n_gaussianas(sd, config):
    for key in ["mu_a0", "color_a0", "opacity_a0", "scale_a0", "theta_a0", "depth_a0"]:
        value = sd.get(key)
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.shape[0])
    return int(config["n_gaussianas_inicial"])


def resolver_base_temporal(config):
    base_temporal = str(
        config.get("base_temporal", "chebyshev")
    ).lower().strip()

    if base_temporal in ("chebyshev", "cheby", "cheb"):
        return "chebyshev", construir_matriz_chebyshev

    if base_temporal in ("monomial", "mono"):
        return "monomial", construir_matriz_monomial

    raise ValueError(
        f"base_temporal desconocida: {base_temporal!r}. "
        "Usa 'chebyshev' o 'monomial'."
    )


def _leer_info_clip_cercano(checkpoint):
    """
    Si el checkpoint esta en outputs/<exp>/checkpoints/checkpoint_final.pt,
    intenta leer outputs/<exp>/info_clip.json.
    """
    checkpoint = Path(checkpoint).resolve()
    candidatos = [
        checkpoint.parent.parent / "info_clip.json",
        checkpoint.parent / "info_clip.json",
    ]
    for p in candidatos:
        if p.is_file():
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def cargar_modelo_desde_checkpoint(checkpoint, device="cpu", clip_override=None):
    """
    Returns:
        modelo, config, matrices_base, info

    info contiene:
        N, H, W, n_frames, grados
    """
    checkpoint = Path(checkpoint).resolve()
    device = elegir_device(device)

    ckpt = _torch_load_seguro(checkpoint, map_location="cpu")
    if "state_dict_coefs" not in ckpt:
        raise RuntimeError(f"Checkpoint sin state_dict_coefs: {checkpoint}")

    config = dict(ckpt.get("config", {}))
    if clip_override is not None:
        config["clip"] = clip_override

    info_clip = _leer_info_clip_cercano(checkpoint)

    sd = ckpt["state_dict_coefs"]
    grados = dict(sd.get("grados", config.get("grados", {})))
    if not grados:
        raise RuntimeError("No pude inferir grados desde checkpoint/config.")

    n_gaussianas = int(sd.get("N", inferir_n_gaussianas(sd, config)))
    H = int(sd.get("H", info_clip.get("H", config.get("H", 0))))
    W = int(sd.get("W", info_clip.get("W", config.get("W", 0))))
    n_frames = int(sd.get("n_frames", info_clip.get("n_frames", config.get("max_frames", 0))))

    if H <= 0 or W <= 0 or n_frames <= 0:
        raise RuntimeError(
            f"No pude inferir H/W/n_frames. H={H}, W={W}, n_frames={n_frames}. "
            f"Revisa checkpoint o info_clip.json."
        )

    modelo = GaussianasPolinomial2D(
        n_gaussianas=n_gaussianas,
        n_frames=n_frames,
        grados=grados,
        H=H,
        W=W,
        device=device,
        escala_inicial_px=float(config.get("escala_inicial_px", 5.0)),
        frame_0_imagen=None,
        semilla=int(config.get("seed", 42)),
    )

    with torch.no_grad():
        for nombre in ["mu", "opacity", "color", "scale", "theta", "depth"]:
            getattr(modelo, f"{nombre}_a0").copy_(sd[f"{nombre}_a0"].to(device))
            getattr(modelo, f"{nombre}_high").copy_(sd[f"{nombre}_high"].to(device))

    modelo.eval()

    base_temporal, construir_matriz_base = resolver_base_temporal(config)
    config["base_temporal"] = base_temporal

    grados_distintos = sorted(set(grados.values()))
    matrices_base = {
        g: construir_matriz_base(
            n_frames,
            g,
            device=device,
            dtype=torch.float32,
        )
        for g in grados_distintos
    }

    info = {
        "N": n_gaussianas,
        "H": H,
        "W": W,
        "n_frames": n_frames,
        "grados": grados,
        "base_temporal": base_temporal,
        "checkpoint": str(checkpoint),
    }
    return modelo, config, matrices_base, info


def cargar_frame_fondo(config, indice, H=None, W=None):
    """
    Carga frame GT desde data/clips/<clip>/frame_NNNN.png si existe.
    Devuelve torch tensor float32 [H,W,3] en [0,1], o None.
    """
    clip = config.get("clip")
    if not clip:
        return None

    ruta = RAIZ / "data" / "clips" / clip / f"frame_{int(indice):04d}.png"
    if not ruta.is_file():
        return None

    img = Image.open(ruta).convert("RGB")
    if H is not None and W is not None and img.size != (int(W), int(H)):
        img = img.resize((int(W), int(H)))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def cargar_fondo_desde_dir_o_gt(fondo_dir, config, indice, H, W):
    """
    Prioridad:
      1. fondo_dir/frame_NNNN.png, normalmente frames_renderizados
      2. data/clips/<clip>/frame_NNNN.png
      3. imagen negra
    """
    if fondo_dir is not None:
        ruta = Path(fondo_dir) / f"frame_{int(indice):04d}.png"
        if ruta.is_file():
            img = Image.open(ruta).convert("RGB")
            if img.size != (int(W), int(H)):
                img = img.resize((int(W), int(H)))
            return np.asarray(img, dtype=np.float32) / 255.0

    gt = cargar_frame_fondo(config, indice, H=H, W=W)
    if gt is not None:
        return gt.numpy()

    return np.zeros((int(H), int(W), 3), dtype=np.float32)
