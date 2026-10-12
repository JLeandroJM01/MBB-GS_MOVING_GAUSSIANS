
"""
Pipeline audiovisual Gaussian2D + Gabor.

Flujo:
1) Extrae frames y audio del mismo segmento MP4.
2) Entrena video y audio.
3) Ejecuta pruning dinámico 10%, 15%, 20%... usando PSNR global.
4) Conserva el último pruning aprobado o el modelo original.
5) Genera UINT16 SAFE y UINT16 ALL, renderiza y mide PSNR.
6) Comprime packages con 7-Zip cuando está disponible.
7) Une video reconstruido + recon_stereo.wav.
8) Guarda resumen JSON/TXT.

PowerShell:
python scripts\pipeline\run_pipeline_video_audio.py `
  --config configs\audiovisual\rockyourbody_10s_1ep\pipeline.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
SCRIPTS = RAIZ / "scripts"
PYTHON = Path(sys.executable)


def run(cmd, capture=False, log_path=None):
    cmd = [str(x) for x in cmd]
    print("\n>", subprocess.list2cmdline(cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=RAIZ,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=True,
    )
    text = result.stdout or ""
    if capture:
        print(text, flush=True)
        if log_path:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(text, encoding="utf-8")
    return text


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def resolve(base_dir, value):
    p = Path(value)
    if p.is_absolute():
        return p
    local = (Path(base_dir) / p).resolve()
    if local.exists():
        return local
    return (RAIZ / p).resolve()


def file_mb(path):
    path = Path(path) if path else None
    if not path or not path.exists():
        return None
    return path.stat().st_size / (1024 ** 2)


def count_frames(path):
    path = Path(path)
    return len(list(path.glob("frame_*.png"))) if path.exists() else 0


def require_program(name):
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"No se encontró {name} en PATH")
    return found


def probe_command(cmd):
    try:
        result = subprocess.run(
            [str(x) for x in cmd],
            cwd=RAIZ,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    text = (result.stdout or "").strip()
    return text or None


def collect_environment(config_path):
    config_path = Path(config_path)

    git_commit = probe_command(["git", "rev-parse", "HEAD"])
    git_status = probe_command(["git", "status", "--porcelain"])

    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": bool(git_status) if git_status is not None else None,
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "pipeline_config": str(config_path.resolve()),
        "pipeline_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "cuda_home": str(
            Path(shutil.which("nvcc")).resolve().parents[1]
        ) if shutil.which("nvcc") else None,
        "nvcc": None,
        "ffmpeg": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": False,
        "gpu": None,
    }

    nvcc = shutil.which("nvcc")
    if nvcc:
        nvcc_text = probe_command([nvcc, "--version"])
        if nvcc_text:
            for line in nvcc_text.splitlines():
                if "release" in line.lower():
                    info["nvcc"] = line.strip()
                    break
            if info["nvcc"] is None:
                info["nvcc"] = nvcc_text.splitlines()[0]

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        ffmpeg_text = probe_command([ffmpeg, "-version"])
        if ffmpeg_text:
            info["ffmpeg"] = ffmpeg_text.splitlines()[0]

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda_version"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())

        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["torch_error"] = str(exc)

    return info


def save_environment(pipeline_out, environment):
    save_json(pipeline_out / "environment.json", environment)

    lines = [
        "MBB-GS ENVIRONMENT",
        "=" * 64,
        f"Timestamp UTC: {environment.get('timestamp_utc')}",
        f"Git commit: {environment.get('git_commit')}",
        f"Git dirty: {environment.get('git_dirty')}",
        f"Python: {environment.get('python_version')}",
        f"Python executable: {environment.get('python_executable')}",
        f"PyTorch: {environment.get('torch_version')}",
        f"PyTorch CUDA: {environment.get('torch_cuda_version')}",
        f"CUDA available: {environment.get('cuda_available')}",
        f"GPU: {environment.get('gpu')}",
        f"CUDA_HOME: {environment.get('cuda_home')}",
        f"NVCC: {environment.get('nvcc')}",
        f"FFmpeg: {environment.get('ffmpeg')}",
        f"Platform: {environment.get('platform')}",
        f"Pipeline config: {environment.get('pipeline_config')}",
        f"Config SHA256: {environment.get('pipeline_config_sha256')}",
    ]

    (pipeline_out / "environment.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def ffprobe_info(ffprobe, video):
    text = run([
        ffprobe, "-v", "error",
        "-show_entries", "format=duration:stream=codec_type,channels,sample_rate,r_frame_rate",
        "-of", "json", video,
    ], capture=True)
    return json.loads(text)


def read_pruning_winner(summary_csv):
    summary_csv = Path(summary_csv)
    if not summary_csv.exists():
        return None
    rows = []
    with summary_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            ok = str(row.get("aprobado", "")).strip().lower() in {"true", "1", "yes", "si", "sí"}
            if ok:
                row["_pct"] = float(row["porcentaje"])
                rows.append(row)
    return max(rows, key=lambda r: r["_pct"]) if rows else None


def parse_compare(text):
    labels = {
        "psnr_promedio": "PSNR promedio",
        "psnr_min": "PSNR min",
        "psnr_p5": "PSNR p5",
        "mse_promedio": "MSE promedio",
        "mae_promedio": "MAE promedio",
    }
    out = {}
    for key, label in labels.items():
        m = re.search(rf"{re.escape(label)}\s*:\s*([0-9.+\-eE]+|inf|nan)", text, re.I)
        if not m:
            out[key] = None
        elif m.group(1).lower() == "inf":
            out[key] = float("inf")
        elif m.group(1).lower() == "nan":
            out[key] = float("nan")
        else:
            out[key] = float(m.group(1))
    mse = out.get("mse_promedio")
    if mse is None:
        out["psnr_global"] = None
    elif mse == 0:
        out["psnr_global"] = float("inf")
    elif mse > 0:
        out["psnr_global"] = -10.0 * math.log10(mse)
    else:
        out["psnr_global"] = None
    return out


def compare_frames(baseline, test, out_csv, out_txt):
    text = run([
        PYTHON, SCRIPTS / "metrics" / "comparar_frames_psnr.py",
        "--a", baseline, "--b", test, "--out", out_csv,
    ], capture=True, log_path=out_txt)
    return parse_compare(text)


def find_7z():
    for name in ("7z", "7za"):
        p = shutil.which(name)
        if p:
            return p
    common = Path(r"C:\Program Files\7-Zip\7z.exe")
    return str(common) if common.exists() else None


def compress_7z(exe, source):
    source = Path(source)
    if not exe or not source.exists():
        return None
    archive = Path(str(source) + ".7z")
    if archive.exists():
        archive.unlink()
    run([exe, "a", "-t7z", archive, source, "-mx=9"])
    return archive


def render_checkpoint(checkpoint, out_dir, n_frames, fps, device):
    out_dir = Path(out_dir)
    frames = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    run([
        PYTHON, SCRIPTS / "reconstruction" / "regenerar_clip_desde_checkpoint_streaming.py",
        "--checkpoint", checkpoint,
        "--salida", frames,
        "--device", device,
        "--inicio", "0",
        "--fin", str(n_frames),
        "--crear_video",
        "--fps", str(int(fps)),
    ])
    return frames, out_dir / "video_reconstruido.mp4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg_arg = Path(args.config)
    config_path = cfg_arg.resolve() if cfg_arg.is_absolute() else (Path.cwd() / cfg_arg).resolve()
    if not config_path.exists():
        config_path = (RAIZ / cfg_arg).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    master = load_json(config_path)
    master_dir = config_path.parent
    ffmpeg = require_program("ffmpeg")
    ffprobe = require_program("ffprobe")

    nombre = str(master["nombre_pipeline"])
    mp4 = resolve(master_dir, master["mp4_original"])
    video_cfg_path = resolve(master_dir, master["config_video"])
    audio_cfg_path = resolve(master_dir, master["config_audio"])
    if not mp4.exists():
        raise FileNotFoundError(mp4)

    probe = ffprobe_info(ffprobe, mp4)
    streams = probe.get("streams", [])
    if not any(s.get("codec_type") == "audio" for s in streams):
        raise RuntimeError(f"El MP4 no tiene audio: {mp4}")
    if not any(s.get("codec_type") == "video" for s in streams):
        raise RuntimeError(f"El archivo no tiene video: {mp4}")

    total = float(probe.get("format", {}).get("duration", 0) or 0)
    inicio = float(master.get("inicio_segundos", 0))
    solicitada = float(master.get("duracion_segundos", 10))
    duracion = min(solicitada, max(0.0, total - inicio)) if total > 0 else solicitada
    if duracion <= 0:
        raise RuntimeError("Segmento sin duración disponible")

    fps = int(master.get("fps", 30))
    H, W = [int(x) for x in master.get("resolucion", [720, 1280])]
    device = str(master.get("device", "cuda"))
    n_objetivo = int(round(duracion * fps))
    clip = str(master.get("nombre_clip", f"{nombre}_clip"))

    pipeline_out = RAIZ / "outputs" / "AV_PIPELINE" / nombre
    limpiar_pipeline = bool(master.get("limpiar_salida_pipeline", True))

    if limpiar_pipeline and pipeline_out.exists():
        print(f"[pipeline] limpiando salida anterior: {pipeline_out}")
        shutil.rmtree(pipeline_out)

    runtime_dir = pipeline_out / "runtime_configs"
    logs_dir = pipeline_out / "logs"
    quant_dir = pipeline_out / "video_quant"
    final_dir = pipeline_out / "final"
    for d in (runtime_dir, logs_dir, quant_dir, final_dir):
        d.mkdir(parents=True, exist_ok=True)

    shutil.copy2(config_path, runtime_dir / "pipeline_master.json")

    environment = collect_environment(config_path)
    save_environment(pipeline_out, environment)

    print("=" * 64)
    print("PIPELINE AUDIOVISUAL")
    print("=" * 64)
    print(f"MP4: {mp4}")
    print(f"Segmento: {inicio:.3f}s, duración={duracion:.3f}s")
    print(f"Video: {n_objetivo} frames, {fps} fps, {H}x{W}")

    # 1) Frames
    cmd_extract = [
        PYTHON, SCRIPTS / "data" / "extraer_clips_720p.py",
        "--video", mp4,
        "--nombre_clip", clip,
        "--inicio_seg", str(inicio),
        "--duracion_seg", str(duracion),
        "--max_frames", str(n_objetivo),
        "--fps", str(fps),
        "--H", str(H), "--W", str(W),
    ]
    if bool(master.get("forzar_extraccion", True)):
        cmd_extract.append("--forzar")
    run(cmd_extract)
    clip_dir = RAIZ / "data" / "clips" / clip
    n_frames = count_frames(clip_dir)
    if n_frames <= 0:
        raise RuntimeError("No se extrajeron frames")

    # 2) Audio WAV estéreo
    audio_template = load_json(audio_cfg_path)
    sr = int(audio_template.get("sr", 44100))
    audio_wav = RAIZ / "data" / "audio_pipeline" / f"{nombre}.wav"
    audio_wav.parent.mkdir(parents=True, exist_ok=True)
    run([
        ffmpeg, "-y", "-ss", str(inicio), "-i", mp4,
        "-t", str(duracion), "-map", "0:a:0", "-vn",
        "-ac", "2", "-ar", str(sr), "-c:a", "pcm_s16le", audio_wav,
    ])

    # 3) Runtime video config
    video_cfg = load_json(video_cfg_path)
    video_exp = str(video_cfg.get("nombre_experimento", f"{nombre}_video"))
    video_cfg.update({
        "nombre_experimento": video_exp,
        "clip": clip,
        "video_mp4": None,
        "max_frames": n_frames,
        "n_frames_extraer": None,
        "fps_extraccion": None,
        "resolucion_extraccion": [H, W],
        "device": device,
        "ejecutar_pruning_post": False,
        "guardar_frames_rasterizados": True,
        "sobreescribir_salida": True,
        "limpiar_salida": True,
    })
    runtime_video = runtime_dir / "video_runtime.json"
    save_json(runtime_video, video_cfg)

    # 4) Runtime audio config
    audio_exp = str(audio_template.get("nombre_experimento", f"{nombre}_audio"))
    audio_template.update({
        "nombre_experimento": audio_exp,
        "audio": audio_wav.relative_to(RAIZ).as_posix(),
        "max_segundos": duracion,
        "sr": sr,
        "device": device,
        "sobreescribir_salida": True,
        "limpiar_salida": True,
    })
    runtime_audio = runtime_dir / "audio_runtime.json"
    save_json(runtime_audio, audio_template)

    # 5) Entrenar video
    run([PYTHON, SCRIPTS / "train.py", "--config", runtime_video])
    video_dir = RAIZ / "outputs" / video_exp
    original_ckpt = video_dir / "checkpoints" / "checkpoint_final.pt"
    baseline_frames = video_dir / "frames_renderizados"
    if not original_ckpt.exists():
        raise FileNotFoundError(original_ckpt)
    if count_frames(baseline_frames) != n_frames:
        raise RuntimeError(f"Baseline incompleto: {count_frames(baseline_frames)}/{n_frames}")

    # 6) Entrenar audio
    run([PYTHON, SCRIPTS / "audio" / "train_gabor_stereo.py", "--config", runtime_audio])
    audio_dir = RAIZ / "outputs" / "gabor" / audio_exp
    recon_audio = audio_dir / "recon_stereo.wav"
    if not recon_audio.exists():
        raise FileNotFoundError(recon_audio)

    # 7) Pruning dinámico
    pruning = dict(master.get("pruning", {}))
    run([
        PYTHON, SCRIPTS / "compression" / "run_binary_pruning_adaptativo.py",
        "--exp", video_dir,
        "--checkpoint", original_ckpt,
        "--baseline_frames", baseline_frames,
        "--fps", str(fps), "--device", device,
        "--inicio_pct", str(pruning.get("inicio_pct", 10)),
        "--paso_pct", str(pruning.get("paso_pct", 5)),
        "--min_pct", str(pruning.get("min_pct", 10)),
        "--max_pct", str(pruning.get("max_pct", 50)),
        "--psnr_min", str(pruning.get("psnr_global_min", 65)),
        "--crear_video_ganador",
        "--force",
    ])

    pruning_dir = video_dir / "binary_pruning"
    winner = read_pruning_winner(pruning_dir / "resumen_busqueda_pruning.csv")
    if winner:
        label = str(winner["regla"])
        pct = float(winner["porcentaje"])
        selected_ckpt = pruning_dir / "checkpoints" / f"checkpoint_{label}_fp32.pt"
        restantes = int(float(winner["n_restantes"]))
        if not selected_ckpt.exists():
            raise FileNotFoundError(selected_ckpt)
    else:
        label, pct, restantes = "sin_pruning", 0.0, None
        selected_ckpt = original_ckpt

    # 8) UINT16 SAFE
    qcfg = dict(master.get("video_cuantizacion", {}))
    safe_pkg = quant_dir / "checkpoint_uint16_safe.pkg.pt"
    safe_ckpt = quant_dir / "checkpoint_uint16_safe_render.pt"
    safe_video = None
    safe_metrics = None
    if bool(qcfg.get("generar_uint16_safe", True)):
        run([
            PYTHON, SCRIPTS / "compression" / "pack_checkpoint_uint16.py",
            "--in_ckpt", selected_ckpt,
            "--out_pkg", safe_pkg,
            "--other_float", "fp32",
            "--quant_tensors", "mu_high", "color_high", "opacity_high", "scale_high",
            "--omit_zero_depth_high",
        ])
        run([
            PYTHON, SCRIPTS / "compression" / "unpack_checkpoint_uint16.py",
            "--in_pkg", safe_pkg,
            "--out_ckpt", safe_ckpt,
            "--out_float", "fp32",
        ])
        safe_frames, safe_video = render_checkpoint(
            safe_ckpt, quant_dir / "render_uint16_safe", n_frames, fps, device
        )
        safe_metrics = compare_frames(
            baseline_frames, safe_frames,
            quant_dir / "psnr_baseline_vs_uint16_safe.csv",
            logs_dir / "psnr_baseline_vs_uint16_safe.txt",
        )

    # 9) UINT16 ALL
    all_pkg = quant_dir / "checkpoint_uint16_all.pkg.pt"
    all_ckpt = quant_dir / "checkpoint_uint16_all_render.pt"
    all_video = None
    all_metrics = None
    if bool(qcfg.get("generar_uint16_all", True)):
        run([
            PYTHON, SCRIPTS / "compression" / "pack_checkpoint_uint16_all.py",
            "--in_ckpt", selected_ckpt,
            "--out_pkg", all_pkg,
            "--omit_zero_depth_high",
        ])
        run([
            PYTHON, SCRIPTS / "compression" / "unpack_checkpoint_uint16_all.py",
            "--in_pkg", all_pkg,
            "--out_ckpt", all_ckpt,
        ])
        all_frames, all_video = render_checkpoint(
            all_ckpt, quant_dir / "render_uint16_all", n_frames, fps, device
        )
        all_metrics = compare_frames(
            baseline_frames, all_frames,
            quant_dir / "psnr_baseline_vs_uint16_all.csv",
            logs_dir / "psnr_baseline_vs_uint16_all.txt",
        )

    # 10) 7-Zip
    seven = find_7z()
    safe_7z = compress_7z(seven, safe_pkg) if safe_pkg.exists() else None
    all_7z = compress_7z(seven, all_pkg) if all_pkg.exists() else None

    # 11) Video final
    mode = str(qcfg.get("usar_para_video_final", "uint16_safe")).lower()
    if mode == "uint16_safe" and safe_video and safe_video.exists():
        video_sin_audio = safe_video
    elif mode == "uint16_all" and all_video and all_video.exists():
        video_sin_audio = all_video
    else:
        _, video_sin_audio = render_checkpoint(
            selected_ckpt, quant_dir / "render_fp32_elegido", n_frames, fps, device
        )
        mode = "fp32"

    bitrate = int(master.get("video_final", {}).get("bitrate_audio_kbps", 192))
    final_video = final_dir / f"{nombre}_reconstruido_con_audio.mp4"
    run([
        ffmpeg, "-y", "-i", video_sin_audio, "-i", recon_audio,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", f"{bitrate}k",
        "-shortest", final_video,
    ])

    audio_metrics_path = audio_dir / "metricas.json"
    audio_metrics = load_json(audio_metrics_path) if audio_metrics_path.exists() else {}

    summary = {
        "pipeline": nombre,
        "environment": environment,
        "fuente_mp4": str(mp4),
        "inicio_segundos": inicio,
        "duracion_segundos": duracion,
        "fps": fps,
        "resolucion": [H, W],
        "n_frames": n_frames,
        "pruning": {
            "ganador": label,
            "porcentaje_eliminado": pct,
            "gaussianas_restantes": restantes,
            "checkpoint": str(selected_ckpt),
            "checkpoint_mb": file_mb(selected_ckpt),
        },
        "uint16_safe": {
            "package": str(safe_pkg) if safe_pkg.exists() else None,
            "package_mb": file_mb(safe_pkg),
            "archive_7z": str(safe_7z) if safe_7z else None,
            "archive_7z_mb": file_mb(safe_7z),
            "metricas_vs_baseline": safe_metrics,
        },
        "uint16_all": {
            "package": str(all_pkg) if all_pkg.exists() else None,
            "package_mb": file_mb(all_pkg),
            "archive_7z": str(all_7z) if all_7z else None,
            "archive_7z_mb": file_mb(all_7z),
            "metricas_vs_baseline": all_metrics,
        },
        "audio": {
            "recon_stereo_wav": str(recon_audio),
            "snr_stereo_db": audio_metrics.get("snr_stereo_db"),
            "si_sdr_stereo_db": audio_metrics.get("si_sdr_stereo_db"),
            "mse_wave_stereo": audio_metrics.get("mse_wave_stereo"),
            "bytes_modelo": audio_metrics.get("bytes_modelo"),
            "cuantizacion_medida": audio_metrics.get("cuantizacion"),
        },
        "video_final": {
            "modo": mode,
            "salida": str(final_video),
            "salida_mb": file_mb(final_video),
        },
    }
    save_json(pipeline_out / "resumen_pipeline.json", summary)

    lines = [
        "=" * 64,
        "RESUMEN PIPELINE AUDIOVISUAL",
        "=" * 64,
        f"Pipeline: {nombre}",
        f"Fuente: {mp4}",
        f"Git commit: {environment.get('git_commit')}",
        f"Git dirty: {environment.get('git_dirty')}",
        f"Python: {environment.get('python_version')}",
        f"PyTorch: {environment.get('torch_version')}",
        f"CUDA: {environment.get('torch_cuda_version')}",
        f"GPU: {environment.get('gpu')}",
        f"Frames: {n_frames} a {fps} fps, {H}x{W}",
        f"Pruning ganador: {label}",
        f"Porcentaje eliminado: {pct:.2f}%",
        f"UINT16 SAFE MB: {file_mb(safe_pkg)}",
        f"UINT16 SAFE PSNR global: {None if not safe_metrics else safe_metrics.get('psnr_global')}",
        f"UINT16 ALL MB: {file_mb(all_pkg)}",
        f"UINT16 ALL PSNR global: {None if not all_metrics else all_metrics.get('psnr_global')}",
        f"Audio SNR stereo: {audio_metrics.get('snr_stereo_db')}",
        f"Video final: {final_video}",
        f"Video final MB: {file_mb(final_video)}",
    ]
    (pipeline_out / "resumen_pipeline.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 64)
    print("PIPELINE COMPLETADO")
    print("=" * 64)
    print(f"MP4 final: {final_video}")
    print(f"Resumen: {pipeline_out / 'resumen_pipeline.txt'}")


if __name__ == "__main__":
    main()
