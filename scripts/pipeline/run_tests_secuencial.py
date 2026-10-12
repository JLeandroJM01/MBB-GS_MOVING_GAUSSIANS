"""
Ejecucion secuencial de la ablacion sistematica de loss en 3 fases.

Estructura:
- Fase 1: ablacion del tipo de loss intra-frame (Nivel 1).
- Fase 2: ablacion de agregacion entre frames (Nivel 2),
          con el ganador de Fase 1 fijo.
- Fase 3: ablacion de agregacion a nivel pixel (Nivel 0),
          con el ganador de Fase 2 fijo.

Cada fase utiliza los parametros definidos en sus archivos JSON.
El script ejecuta las configuraciones tal como estan almacenadas y no modifica
automaticamente los resultados de una fase a partir de otra.

Uso:
    python scripts/pipeline/run_tests_secuencial.py

Por defecto corre la fase activa (ver variable FASE_ACTIVA abajo). El
nombre del experimento se infiere del stem del archivo de config
(p.ej. 'fase1_baseline.json' -> nombre_experimento='fase1_baseline').
"""
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime


# ============================================================
# FASES Y CONFIGS
# ============================================================

FASE_1 = [
    "configs/video/exp3_loss/fase1_baseline.json",   # L1 + DSSIM (referencia)
    "configs/video/exp3_loss/fase1_l1_mse.json",     # L1 + MSE + DSSIM
    "configs/video/exp3_loss/fase1_edge.json",       # L1 + Sobel edge + DSSIM
    "configs/video/exp3_loss/fase1_temporal.json",   # L1 + |dR - dGT| + DSSIM
    "configs/video/exp3_loss/fase1_motion.json",     # L1 ponderado por motion del GT + DSSIM
    "configs/video/exp3_loss/fase1_combo.json",      # motion x hard + DSSIM
]

# Fase 2 utiliza los parametros definidos en sus propios configs.
FASE_2 = [
    "configs/video/exp3_frame_aggregation/fase2_qframe1.json",    # exponente_frame=1 (referencia)
    "configs/video/exp3_frame_aggregation/fase2_qframe2.json",    # exponente_frame=2
    "configs/video/exp3_frame_aggregation/fase2_qframe4.json",    # exponente_frame=4
    "configs/video/exp3_frame_aggregation/fase2_qframe8.json",    # exponente_frame=8
    "configs/video/exp3_frame_aggregation/fase2_maxframe.json",   # usar_max_frame=true (max puro)
]

# Fase 3 utiliza los parametros definidos en sus propios configs.
FASE_3 = [
    "configs/video/exp3_pixel_aggregation/fase3_qpixel1.json",    # exponente_pixel=1 (referencia)
    "configs/video/exp3_pixel_aggregation/fase3_qpixel2.json",    # exponente_pixel=2
    "configs/video/exp3_pixel_aggregation/fase3_qpixel4.json",    # exponente_pixel=4
]

FASE_BASES = [
    "configs/video/exp1_temporal_basis/thriller_chebyshev_8k_360ep.json",
    "configs/video/exp1_temporal_basis/thriller_monomial_8k_360ep.json",
]

FASE_CAPACIDAD = [
    "configs/video/exp2_capacity/thriller_cheby_12k_gbase_300ep.json",
    "configs/video/exp2_capacity/thriller_cheby_16k_gbase_300ep.json",
    "configs/video/exp2_capacity/thriller_cheby_8k_g36_300ep.json",
    "configs/video/exp2_capacity/thriller_cheby_8k_g45_300ep.json",
]

FASES = {
    "capacidad": FASE_CAPACIDAD,
    "bases": FASE_BASES,
    "fase1": FASE_1,
    "fase2": FASE_2,
    "fase3": FASE_3,
    "todas": FASE_1 + FASE_2 + FASE_3,
}

# Cambia esto para correr otra fase. Tambien acepta primer argv.
FASE_ACTIVA = "fase1"

# Si True: si un test falla, sigue con el siguiente.
# Si False: si un test falla, se detiene todo.
CONTINUAR_SI_FALLA = True


def main():
    raiz = Path(__file__).resolve().parents[2]
    script_train = raiz / "scripts" / "train.py"

    fase_seleccionada = sys.argv[1] if len(sys.argv) > 1 else FASE_ACTIVA

    if fase_seleccionada not in FASES:
        print(f"ERROR: fase '{fase_seleccionada}' no existe. Opciones: {list(FASES.keys())}")
        sys.exit(2)

    configs = FASES[fase_seleccionada]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta_logs = raiz / "outputs" / "_batch_logs" / f"batch_{fase_seleccionada}_{timestamp}"
    carpeta_logs.mkdir(parents=True, exist_ok=True)

    resumen = []

    print("============================================================")
    print(f" EJECUCION SECUENCIAL: {fase_seleccionada}")
    print("============================================================")
    print(f"Raiz proyecto : {raiz}")
    print(f"Python usado  : {sys.executable}")
    print(f"Logs en       : {carpeta_logs}")
    print(f"Configs       : {len(configs)}")
    print("")

    inicio_total = time.time()

    for idx, config_rel in enumerate(configs, start=1):
        config_path = raiz / config_rel

        if not config_path.exists():
            msg = f"[ERROR] No existe config: {config_path}"
            print(msg)
            resumen.append((config_rel, "NO_EXISTE", 0.0))
            if not CONTINUAR_SI_FALLA:
                break
            continue

        # El nombre del experimento viene del stem del config.
        # Asi cada salida queda en outputs/<stem>/.
        nombre_exp = config_path.stem

        nombre_log = f"{idx:02d}_{nombre_exp}.log"
        ruta_log = carpeta_logs / nombre_log

        print("------------------------------------------------------------")
        print(f"[{idx}/{len(configs)}] Ejecutando: {config_rel}")
        print(f"Nombre experimento: {nombre_exp}")
        print(f"Log: {ruta_log}")
        print("------------------------------------------------------------")

        comando = [
            sys.executable,
            str(script_train),
            "--config", str(config_path),
            "--nombre-experimento", nombre_exp,
        ]

        inicio = time.time()

        with open(ruta_log, "w", encoding="utf-8", errors="replace") as f:
            f.write("============================================================\n")
            f.write(f"CONFIG: {config_rel}\n")
            f.write(f"NOMBRE_EXP: {nombre_exp}\n")
            f.write(f"INICIO: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"COMANDO: {' '.join(comando)}\n")
            f.write("============================================================\n\n")
            f.flush()

            proceso = subprocess.Popen(
                comando,
                cwd=str(raiz),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            for linea in proceso.stdout:
                print(linea, end="")
                f.write(linea)
                f.flush()

            codigo = proceso.wait()

        duracion = time.time() - inicio
        estado = "OK" if codigo == 0 else f"FALLO_{codigo}"
        resumen.append((config_rel, estado, duracion))

        print("")
        print(f"[{estado}] {config_rel} terminado en {duracion / 60:.1f} min")
        time.sleep(10)
        print("")

        if codigo != 0 and not CONTINUAR_SI_FALLA:
            print("Se detiene la ejecucion porque CONTINUAR_SI_FALLA=False")
            break

    duracion_total = time.time() - inicio_total

    ruta_resumen = carpeta_logs / "resumen_batch.txt"
    with open(ruta_resumen, "w", encoding="utf-8") as f:
        f.write(f"RESUMEN DE BATCH: {fase_seleccionada}\n")
        f.write("================\n")
        f.write(f"Inicio batch: {timestamp}\n")
        f.write(f"Duracion total min: {duracion_total / 60:.2f}\n\n")

        for config_rel, estado, duracion in resumen:
            f.write(f"{estado:12s} | {duracion / 60:8.2f} min | {config_rel}\n")

    print("============================================================")
    print("RESUMEN")
    print("============================================================")
    for config_rel, estado, duracion in resumen:
        print(f"{estado:12s} | {duracion / 60:8.2f} min | {config_rel}")

    print("")
    print(f"Resumen guardado en: {ruta_resumen}")


if __name__ == "__main__":
    main()
