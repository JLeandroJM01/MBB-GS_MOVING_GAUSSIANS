import argparse
import csv
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch



METRICS = (
    "path_length_px",
    "color_path",
    "op_std",
    "scale_std",
)


def run(cmd, capture=False, log_path=None):
    cmd = [str(x) for x in cmd]

    print("")
    print(">", " ".join(cmd))

    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=True,
    )

    if capture:
        text = result.stdout or ""
        print(text)

        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(text, encoding="utf-8")

        return text

    return ""


def resolve_path(root, value):
    path = Path(value)

    if path.is_absolute():
        return path

    return root / path


def load_checkpoint(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_info(path):
    ckpt = load_checkpoint(path)

    if "state_dict_coefs" not in ckpt:
        raise RuntimeError(
            f"El checkpoint no tiene state_dict_coefs: {path}"
        )

    sd = ckpt["state_dict_coefs"]
    config = ckpt.get("config", {})

    n_frames = sd.get(
        "n_frames",
        config.get("max_frames"),
    )

    if n_frames is None:
        raise RuntimeError(
            "No se pudo determinar n_frames."
        )

    N = sd.get("N")

    if N is None:
        for value in sd.values():
            if (
                torch.is_tensor(value)
                and value.ndim >= 1
            ):
                N = value.shape[0]
                break

    if N is None:
        raise RuntimeError(
            "No se pudo determinar N."
        )

    return int(N), int(n_frames)


def count_frames(frames_dir):
    frames_dir = Path(frames_dir)

    if not frames_dir.exists():
        return 0

    return len(list(frames_dir.glob("frame_*.png")))


def file_mb(path):
    path = Path(path)

    if not path.exists():
        return None

    return path.stat().st_size / 1024 / 1024


def safe_float(row, key, default=0.0):
    try:
        value = float(row.get(key, default))

        if not math.isfinite(value):
            return default

        return value
    except (TypeError, ValueError):
        return default


def percentile_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    if n <= 1:
        return np.zeros(n, dtype=np.float64)

    order = np.argsort(
        values,
        kind="mergesort",
    )

    ranks = np.empty(n, dtype=np.float64)

    ranks[order] = (
        np.arange(n, dtype=np.float64)
        / float(n - 1)
    )

    return ranks


class AdaptiveSelector:
    def __init__(
        self,
        stats_csv,
        op_mean_min=0.05,
        transient_op_max=0.90,
        transient_active_frac=0.10,
        w_path=0.30,
        w_color=0.25,
        w_opstd=0.25,
        w_scale=0.20,
    ):
        self.stats_csv = Path(stats_csv)

        with self.stats_csv.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as fh:
            reader = csv.DictReader(fh)
            self.fieldnames = list(
                reader.fieldnames or []
            )
            self.rows = list(reader)

        if "id" not in self.fieldnames:
            raise RuntimeError(
                "gaussian_stats.csv no tiene columna id."
            )

        for metric in METRICS:
            if metric not in self.fieldnames:
                raise RuntimeError(
                    f"Falta columna requerida: {metric}"
                )

        self.n_total = len(self.rows)
        self.candidate_indices = []
        self.protected_indices = []

        for i, row in enumerate(self.rows):
            op_mean = safe_float(row, "op_mean")
            op_max = safe_float(row, "op_max")
            active_frac = safe_float(
                row,
                "active_frac",
                1.0,
            )

            transient = (
                op_max >= transient_op_max
                and active_frac <= transient_active_frac
            )

            if transient:
                self.protected_indices.append(i)
                continue

            if op_mean >= op_mean_min:
                self.candidate_indices.append(i)

        if not self.candidate_indices:
            raise RuntimeError(
                "No se encontraron gaussianas candidatas."
            )

        values = {}

        for metric in METRICS:
            values[metric] = percentile_ranks([
                safe_float(self.rows[i], metric)
                for i in self.candidate_indices
            ])

        self.scores = (
            w_path * values["path_length_px"]
            + w_color * values["color_path"]
            + w_opstd * values["op_std"]
            + w_scale * values["scale_std"]
        )

        self.sorted_local = np.argsort(
            self.scores,
            kind="mergesort",
        )

    def write_ids_csv(self, porcentaje, out_csv):
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        target = int(round(
            self.n_total
            * porcentaje
            / 100.0
        ))

        if target <= 0:
            raise RuntimeError(
                "El porcentaje genera cero gaussianas."
            )

        if target > len(self.candidate_indices):
            raise RuntimeError(
                "No hay suficientes candidatas para "
                f"eliminar {porcentaje}%."
            )

        selected_local = self.sorted_local[:target]
        selected_rows = []

        out_fields = self.fieldnames + [
            "adaptive_score",
            "adaptive_rank",
        ]

        for rank, local_idx in enumerate(selected_local):
            local_idx = int(local_idx)
            global_idx = self.candidate_indices[local_idx]

            row = dict(self.rows[global_idx])
            row["adaptive_score"] = (
                f"{float(self.scores[local_idx]):.12g}"
            )
            row["adaptive_rank"] = str(rank)

            selected_rows.append(row)

        with out_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=out_fields,
            )
            writer.writeheader()
            writer.writerows(selected_rows)

        return {
            "n_original": self.n_total,
            "n_eliminadas": target,
            "n_restantes": self.n_total - target,
            "porcentaje": porcentaje,
            "cutoff_score": float(
                self.scores[int(selected_local[-1])]
            ),
            "transitorias_protegidas": len(
                self.protected_indices
            ),
        }


def metric_from_text(text, label):
    pattern = (
        rf"{re.escape(label)}\s*:\s*"
        r"([0-9.+\-eE]+|inf|nan)"
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    value = match.group(1).lower()

    if value == "inf":
        return float("inf")

    if value == "nan":
        return float("nan")

    return float(value)


def parse_metrics(text):
    return {
        "psnr_mean": metric_from_text(
            text,
            "PSNR promedio",
        ),
        "psnr_min": metric_from_text(
            text,
            "PSNR min",
        ),
        "psnr_p5": metric_from_text(
            text,
            "PSNR p5",
        ),
        "psnr_max": metric_from_text(
            text,
            "PSNR max",
        ),
        "psnr_std": metric_from_text(
            text,
            "PSNR std",
        ),
        "mse_mean": metric_from_text(
            text,
            "MSE promedio",
        ),
        "mae_mean": metric_from_text(
            text,
            "MAE promedio",
        ),
    }


def render_baseline(
    python_exe,
    scripts_dir,
    checkpoint,
    frames_dir,
    n_frames,
    device,
    fps,
    force=False,
):
    frames_dir = Path(frames_dir)

    if (
        not force
        and count_frames(frames_dir) == n_frames
    ):
        print(
            f"[SKIP] Baseline completo: {frames_dir}"
        )
        return

    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    frames_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run([
        python_exe,
        scripts_dir
        / "reconstruction"
        / "regenerar_clip_desde_checkpoint_streaming.py",
        "--checkpoint",
        checkpoint,
        "--salida",
        frames_dir,
        "--device",
        device,
        "--inicio",
        "0",
        "--fin",
        str(n_frames),
        "--fps",
        str(int(fps)),
    ])


def render_pruned(
    python_exe,
    scripts_dir,
    checkpoint,
    ids_csv,
    output_dir,
    n_frames,
    fps,
    device,
    force=False,
):
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"

    if (
        not force
        and count_frames(frames_dir) == n_frames
    ):
        print(
            f"[SKIP] Render completo: {frames_dir}"
        )
        return frames_dir

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run([
        python_exe,
        scripts_dir / "visualization" / "viz_render_subset.py",
        "--checkpoint",
        checkpoint,
        "--ids_csv",
        ids_csv,
        "--modo",
        "exclude",
        "--salida",
        output_dir,
        "--inicio",
        "0",
        "--fin",
        str(n_frames),
        "--device",
        device,
    ])

    return frames_dir


def compare_frames(
    python_exe,
    scripts_dir,
    baseline_frames,
    test_frames,
    out_csv,
    out_txt,
):
    text = run([
        python_exe,
        scripts_dir / "metrics" / "comparar_frames_psnr.py",
        "--a",
        baseline_frames,
        "--b",
        test_frames,
        "--out",
        out_csv,
    ], capture=True, log_path=out_txt)

    return parse_metrics(text)


def pct_name(value):
    value = float(value)

    if value.is_integer():
        return str(int(value))

    return str(value).replace(".", "_")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
    )
    parser.add_argument(
        "--baseline_frames",
        default=None,
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--inicio_pct",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--paso_pct",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--min_pct",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--max_pct",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--psnr_min",
        type=float,
        default=65.0,
    )

    parser.add_argument(
        "--sample_every_stats",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--crear_video_ganador",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    scripts_dir = root / "scripts"
    python_exe = Path(sys.executable)

    exp = resolve_path(root, args.exp)

    checkpoint = (
        resolve_path(root, args.checkpoint)
        if args.checkpoint
        else exp / "checkpoints" / "checkpoint_final.pt"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No existe checkpoint: {checkpoint}"
        )

    N, n_frames = checkpoint_info(checkpoint)

    work_dir = exp / "binary_pruning"

    if args.force and work_dir.exists():
        print(f"[FORCE] Limpiando resultados anteriores: {work_dir}")
        shutil.rmtree(work_dir)

    ids_dir = work_dir / "ids"
    tests_dir = work_dir / "tests"
    metrics_dir = work_dir / "metrics"
    checkpoints_dir = work_dir / "checkpoints"

    for directory in (
        work_dir,
        ids_dir,
        tests_dir,
        metrics_dir,
        checkpoints_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    stats_csv = (
        exp
        / "viz_stats"
        / "gaussian_stats.csv"
    )

    print("======================================================")
    print("BÚSQUEDA ADAPTATIVA DE PRUNING")
    print("======================================================")
    print(f"Experimento      : {exp}")
    print(f"Checkpoint       : {checkpoint}")
    print(f"N original       : {N}")
    print(f"Frames completos : {n_frames}")
    print(f"Inicio pruning   : {args.inicio_pct}%")
    print(f"Paso             : {args.paso_pct}%")
    print(f"Umbral PSNR      : {args.psnr_min} dB")

    if args.force or not stats_csv.exists():
        run([
            python_exe,
            scripts_dir / "visualization" / "viz_gaussian_stats.py",
            "--checkpoint",
            checkpoint,
            "--chunk",
            "2048",
            "--sample_every",
            str(args.sample_every_stats),
        ])

    if not stats_csv.exists():
        raise RuntimeError(
            f"No se generó: {stats_csv}"
        )

    if args.baseline_frames:
        baseline_frames = resolve_path(
            root,
            args.baseline_frames,
        )

        found = count_frames(baseline_frames)

        if found != n_frames:
            raise RuntimeError(
                "La carpeta baseline indicada tiene "
                f"{found} frames, pero se esperaban "
                f"{n_frames}: {baseline_frames}"
            )
    else:
        baseline_frames = (
            work_dir / "baseline" / "frames"
        )

        render_baseline(
            python_exe=python_exe,
            scripts_dir=scripts_dir,
            checkpoint=checkpoint,
            frames_dir=baseline_frames,
            n_frames=n_frames,
            device=args.device,
            fps=args.fps,
            force=args.force,
        )

    selector = AdaptiveSelector(stats_csv)

    print(f"Candidatas        : {len(selector.candidate_indices)}")
    print(f"Protegidas        : {len(selector.protected_indices)}")

    results = []
    cache = {}

    def evaluate(pct):
        key = float(pct)

        if key in cache:
            return cache[key]

        name = f"adaptativo_{pct_name(pct)}"
        ids_csv = ids_dir / f"{name}.csv"

        selection = selector.write_ids_csv(
            porcentaje=pct,
            out_csv=ids_csv,
        )

        output_dir = tests_dir / name

        frames_dir = render_pruned(
            python_exe=python_exe,
            scripts_dir=scripts_dir,
            checkpoint=checkpoint,
            ids_csv=ids_csv,
            output_dir=output_dir,
            n_frames=n_frames,
            fps=args.fps,
            device=args.device,
            force=args.force,
        )

        out_csv = (
            metrics_dir
            / f"baseline_vs_{name}.csv"
        )
        out_txt = (
            metrics_dir
            / f"baseline_vs_{name}.txt"
        )

        metrics = compare_frames(
            python_exe=python_exe,
            scripts_dir=scripts_dir,
            baseline_frames=baseline_frames,
            test_frames=frames_dir,
            out_csv=out_csv,
            out_txt=out_txt,
        )

        mse_mean = metrics["mse_mean"]

        if mse_mean is None:
            psnr_global = None
        elif mse_mean == 0.0:
            psnr_global = float("inf")
        elif mse_mean > 0.0:
            psnr_global = -10.0 * math.log10(mse_mean)
        else:
            psnr_global = None

        approved = (
            psnr_global is not None
            and psnr_global >= args.psnr_min
        )

        row = {
            "orden": len(results) + 1,
            "regla": name,
            "porcentaje": pct,
            "n_original": selection["n_original"],
            "n_eliminadas": selection["n_eliminadas"],
            "n_restantes": selection["n_restantes"],
            "cutoff_score": selection["cutoff_score"],
            "transitorias_protegidas": (
                selection["transitorias_protegidas"]
            ),
            "psnr_mean": metrics["psnr_mean"],
            "psnr_min": metrics["psnr_min"],
            "psnr_p5": metrics["psnr_p5"],
            "psnr_max": metrics["psnr_max"],
            "psnr_std": metrics["psnr_std"],
            "psnr_global": psnr_global,
            "mse_mean": metrics["mse_mean"],
            "mae_mean": metrics["mae_mean"],
            "aprobado": approved,
            "ids_csv": str(ids_csv),
            "frames_dir": str(frames_dir),
        }

        results.append(row)
        cache[key] = row

        print("")
        print("----------------------------------------------")
        print(f"Resultado       : {name}")
        print(f"Eliminadas      : {selection['n_eliminadas']}")
        print(f"Restantes       : {selection['n_restantes']}")
        print(f"PSNR global     : {psnr_global}")
        print(f"PSNR frames     : {metrics['psnr_mean']}")
        print(
            "Estado          : "
            + (
                "APROBADO"
                if approved
                else "RECHAZADO"
            )
        )
        print("----------------------------------------------")

        return row

    # ==================================================
    # Búsqueda gruesa en pasos de 5%
    # ==================================================
    initial = evaluate(args.inicio_pct)
    winner = None

    if initial["aprobado"]:
        winner = initial
        current = (
            args.inicio_pct
            + args.paso_pct
        )

        while current <= args.max_pct:
            result = evaluate(current)

            if result["aprobado"]:
                winner = result
                current += args.paso_pct
            else:
                break

    else:
        current = (
            args.inicio_pct
            - args.paso_pct
        )

        while current >= args.min_pct:
            result = evaluate(current)

            if result["aprobado"]:
                winner = result
                break

            current -= args.paso_pct

    # ==================================================
    # Guardar resumen de búsqueda
    # ==================================================
    summary_csv = (
        work_dir / "resumen_busqueda_pruning.csv"
    )

    fields = [
    "orden",
    "regla",
    "porcentaje",
    "n_original",
    "n_eliminadas",
    "n_restantes",
    "cutoff_score",
    "transitorias_protegidas",
    "psnr_global",
    "psnr_mean",
    "psnr_min",
    "psnr_p5",
    "psnr_max",
    "psnr_std",
    "mse_mean",
    "mae_mean",
    "aprobado",
    "ids_csv",
    "frames_dir",
]
    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(results)

    summary_txt = (
        work_dir / "resumen_busqueda_pruning.txt"
    )

    lines = [
        "======================================================",
        "RESUMEN BÚSQUEDA ADAPTATIVA DE PRUNING",
        "======================================================",
        f"Experimento: {exp}",
        f"Checkpoint: {checkpoint}",
        f"Gaussianas originales: {N}",
        f"Frames evaluados: {n_frames}",
        f"PSNR mínimo requerido: {args.psnr_min} dB",
        "",
        "PRUEBAS EJECUTADAS",
        "------------------------------------------------------",
    ]

    for row in results:
        lines.extend([
            f"{row['regla']}:",
            f"  Eliminadas: {row['n_eliminadas']}",
            f"  Restantes: {row['n_restantes']}",
            f"  PSNR global: {row['psnr_global']}",
            f"  PSNR promedio por frame: {row['psnr_mean']}",
            f"  PSNR mínimo: {row['psnr_min']}",
            f"  PSNR p5: {row['psnr_p5']}",
            f"  Estado: {'APROBADO' if row['aprobado'] else 'RECHAZADO'}",
            "",
        ])

    winner_ckpt = None
    winner_video = None

    if winner is not None:
        winner_name = winner["regla"]
        winner_ids = Path(winner["ids_csv"])

        winner_ckpt = (
            checkpoints_dir
            / f"checkpoint_{winner_name}_fp32.pt"
        )

        if args.force or not winner_ckpt.exists():
            run([
                python_exe,
                scripts_dir
                / "compression"
                / "viz_prune_checkpoint_by_stats.py",
                "--checkpoint",
                checkpoint,
                "--ids_csv",
                winner_ids,
                "--modo",
                "remove_ids",
                "--salida",
                winner_ckpt,
            ])

        if args.crear_video_ganador:
            winner_video = (
                work_dir
                / "ganador"
                / f"{winner_name}.mp4"
            )

            winner_video.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            run([
                python_exe,
                scripts_dir / "data" / "frames_a_video.py",
                "--frames",
                winner["frames_dir"],
                "--salida",
                winner_video,
                "--fps",
                str(args.fps),
            ])

        lines.extend([
            "======================================================",
            "GANADOR",
            "======================================================",
            f"Regla: {winner_name}",
            f"Porcentaje: {winner['porcentaje']}%",
            f"Gaussianas eliminadas: {winner['n_eliminadas']}",
            f"Gaussianas restantes: {winner['n_restantes']}",
            f"PSNR global: {winner['psnr_global']}",
            f"PSNR mínimo: {winner['psnr_min']}",
            f"Checkpoint: {winner_ckpt}",
            f"Tamaño original MB: {file_mb(checkpoint):.4f}",
            f"Tamaño pruneado MB: {file_mb(winner_ckpt):.4f}",
        ])

        if winner_video is not None:
            lines.append(f"Video: {winner_video}")

    else:
        lines.extend([
            "======================================================",
            "SIN GANADOR",
            "======================================================",
            "Ningún porcentaje probado alcanzó el PSNR requerido.",
        ])

    summary_txt.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("======================================================")
    print("BÚSQUEDA FINALIZADA")
    print("======================================================")
    print(f"CSV resumen: {summary_csv}")
    print(f"TXT resumen: {summary_txt}")

    if winner is not None:
        print(f"Ganador: {winner['regla']}")
        print(f"PSNR global: {winner['psnr_global']}")
        print(f"Checkpoint: {winner_ckpt}")
    else:
        print("No se encontró una variante aprobada.")


if __name__ == "__main__":
    main()
