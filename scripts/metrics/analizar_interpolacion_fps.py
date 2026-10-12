import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def cargar_frames(carpeta):
    archivos = sorted(Path(carpeta).glob("frame_*.png"))
    if not archivos:
        raise FileNotFoundError(f"No hay frame_*.png en {carpeta}")

    frames = []
    for ruta in archivos:
        img = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"No se pudo leer {ruta}")
        frames.append(img)

    return np.stack(frames, axis=0)


def psnr(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))

    if mse <= 1e-12:
        return float("inf")

    return 10.0 * math.log10((255.0 ** 2) / mse)


def ssim_canal(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    numerador = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominador = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)

    return float(np.mean(numerador / np.maximum(denominador, 1e-12)))


def ssim_rgb(a, b):
    return float(np.mean([
        ssim_canal(a[:, :, c], b[:, :, c])
        for c in range(3)
    ]))


def resumen_calidad(referencia, prediccion):
    n = min(len(referencia), len(prediccion))

    psnrs = []
    ssims = []

    for i in range(n):
        psnrs.append(psnr(referencia[i], prediccion[i]))
        ssims.append(ssim_rgb(referencia[i], prediccion[i]))

    psnr_finitos = [x for x in psnrs if np.isfinite(x)]

    return {
        "n_frames_comparados": n,
        "psnr_promedio": float(np.mean(psnr_finitos)) if psnr_finitos else None,
        "psnr_min": float(np.min(psnr_finitos)) if psnr_finitos else None,
        "psnr_p5": float(np.percentile(psnr_finitos, 5)) if psnr_finitos else None,
        "ssim_promedio": float(np.mean(ssims)),
        "ssim_min": float(np.min(ssims)),
    }


def frame_remuestreado(frames, u):
    """
    Obtiene un frame en tiempo normalizado u en [0,1] mediante
    interpolación lineal entre imágenes vecinas.
    """
    posicion = u * (len(frames) - 1)
    i0 = int(math.floor(posicion))
    i1 = min(i0 + 1, len(frames) - 1)
    alpha = posicion - i0

    a = frames[i0].astype(np.float32)
    b = frames[i1].astype(np.float32)

    return (1.0 - alpha) * a + alpha * b


def consistencia_entre_fps(modelo30, modelo50):
    psnrs = []
    ssims = []

    for j in range(len(modelo30)):
        u = j / max(1, len(modelo30) - 1)
        estimado50 = frame_remuestreado(modelo50, u)
        referencia30 = modelo30[j].astype(np.float32)

        psnrs.append(psnr(referencia30, estimado50))
        ssims.append(ssim_rgb(referencia30, estimado50))

    psnr_finitos = [x for x in psnrs if np.isfinite(x)]
    n_infinitos = sum(not np.isfinite(x) for x in psnrs)

    if psnr_finitos:
        psnr_promedio = float(np.mean(psnr_finitos))
        psnr_min = float(np.min(psnr_finitos))
        psnr_p5 = float(np.percentile(psnr_finitos, 5))
    else:
        # Todos los frames comparados son idénticos: MSE=0, PSNR infinito.
        psnr_promedio = None
        psnr_min = None
        psnr_p5 = None

    return {
        "psnr_promedio": psnr_promedio,
        "psnr_min": psnr_min,
        "psnr_p5": psnr_p5,
        "psnr_infinito_frames": int(n_infinitos),
        "todos_los_frames_identicos": bool(n_infinitos == len(psnrs)),
        "ssim_promedio": float(np.mean(ssims)),
        "ssim_min": float(np.min(ssims)),
    }


def metricas_temporales(frames):
    x = frames.astype(np.float32) / 255.0
    n = len(x)

    d1 = np.zeros(n, dtype=np.float64)
    d2 = np.zeros(n, dtype=np.float64)

    duplicados_exactos = 0

    for i in range(1, n):
        d1[i] = float(np.mean(np.abs(x[i] - x[i - 1])))

        if np.array_equal(frames[i], frames[i - 1]):
            duplicados_exactos += 1

    for i in range(1, n - 1):
        d2[i] = float(np.mean(np.abs(x[i + 1] - 2.0 * x[i] + x[i - 1])))

    dt = 1.0 / max(1, n - 1)
    velocidad = d1 / dt
    aceleracion = d2 / (dt * dt)

    indices_saltos = np.argsort(d2)[::-1][:10].tolist()

    return {
        "d1": d1,
        "d2": d2,
        "velocidad": velocidad,
        "aceleracion": aceleracion,
        "resumen": {
            "cambio_consecutivo_promedio": float(np.mean(d1[1:])),
            "cambio_consecutivo_p95": float(np.percentile(d1[1:], 95)),
            "segunda_diferencia_promedio": float(np.mean(d2[1:-1])),
            "segunda_diferencia_p95": float(np.percentile(d2[1:-1], 95)),
            "segunda_diferencia_max": float(np.max(d2)),
            "velocidad_normalizada_promedio": float(np.mean(velocidad[1:])),
            "aceleracion_normalizada_promedio": float(np.mean(aceleracion[1:-1])),
            "duplicados_exactos": int(duplicados_exactos),
            "porcentaje_duplicados": float(
                100.0 * duplicados_exactos / max(1, n - 1)
            ),
            "frames_mayor_salto": indices_saltos,
        },
    }


def guardar_contact_sheet(frames, indices, ruta):
    filas = []

    for idx in indices[:6]:
        idx = max(1, min(idx, len(frames) - 2))

        anterior = frames[idx - 1].copy()
        actual = frames[idx].copy()
        siguiente = frames[idx + 1].copy()

        diferencia = cv2.absdiff(anterior, siguiente)
        diferencia = np.clip(
            diferencia.astype(np.float32) * 5.0,
            0,
            255
        ).astype(np.uint8)

        paneles = [
            ("Anterior", anterior),
            (f"Frame {idx}", actual),
            ("Siguiente", siguiente),
            ("Diff x5", diferencia),
        ]

        procesados = []

        for texto, imagen in paneles:
            imagen = imagen.copy()
            cv2.rectangle(imagen, (0, 0), (220, 35), (0, 0, 0), -1)
            cv2.putText(
                imagen,
                texto,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            procesados.append(imagen)

        filas.append(np.concatenate(procesados, axis=1))

    if filas:
        cv2.imwrite(str(ruta), np.concatenate(filas, axis=0))


def guardar_csv(ruta, nombre, metricas):
    d1 = metricas["d1"]
    d2 = metricas["d2"]
    velocidad = metricas["velocidad"]
    aceleracion = metricas["aceleracion"]

    escribir_header = not ruta.exists()

    with ruta.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        if escribir_header:
            w.writerow([
                "secuencia",
                "frame",
                "tiempo_normalizado",
                "cambio_d1",
                "segunda_diferencia_d2",
                "velocidad_normalizada",
                "aceleracion_normalizada",
            ])

        for i in range(len(d1)):
            u = i / max(1, len(d1) - 1)
            w.writerow([
                nombre,
                i,
                f"{u:.8f}",
                f"{d1[i]:.10f}",
                f"{d2[i]:.10f}",
                f"{velocidad[i]:.10f}",
                f"{aceleracion[i]:.10f}",
            ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt30", required=True)
    p.add_argument("--modelo30", required=True)
    p.add_argument("--modelo50", required=True)
    p.add_argument("--salida", required=True)
    args = p.parse_args()

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    print("Cargando GT 30 FPS...")
    gt30 = cargar_frames(args.gt30)

    print("Cargando modelo 30 FPS...")
    modelo30 = cargar_frames(args.modelo30)

    print("Cargando modelo 50 FPS...")
    modelo50 = cargar_frames(args.modelo50)

    print("Calculando calidad de reconstrucción...")
    calidad30 = resumen_calidad(gt30, modelo30)

    print("Calculando consistencia 30 vs 50 FPS...")
    consistencia = consistencia_entre_fps(modelo30, modelo50)

    print("Calculando métricas temporales...")
    temporal30 = metricas_temporales(modelo30)
    temporal50 = metricas_temporales(modelo50)

    reporte = {
        "frames": {
            "gt30": int(len(gt30)),
            "modelo30": int(len(modelo30)),
            "modelo50": int(len(modelo50)),
        },
        "calidad_gt30_vs_modelo30": calidad30,
        "consistencia_modelo30_vs_modelo50": consistencia,
        "temporal_modelo30": temporal30["resumen"],
        "temporal_modelo50": temporal50["resumen"],
    }

    with (salida / "reporte_interpolacion.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(reporte, f, indent=2)

    csv_path = salida / "temporal_por_frame.csv"
    if csv_path.exists():
        csv_path.unlink()

    guardar_csv(csv_path, "modelo30", temporal30)
    guardar_csv(csv_path, "modelo50", temporal50)

    guardar_contact_sheet(
        modelo50,
        temporal50["resumen"]["frames_mayor_salto"],
        salida / "peores_saltos_50fps.png",
    )

    print("")
    print(json.dumps(reporte, indent=2))
    print("")
    print(f"Reporte: {salida / 'reporte_interpolacion.json'}")
    print(f"CSV: {csv_path}")
    print(f"Imagen: {salida / 'peores_saltos_50fps.png'}")


if __name__ == "__main__":
    main()
