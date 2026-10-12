"""
Loop de entrenamiento de UN canal de audio con Gabor splatting.

Se factoriza aqui para que tanto el entrenamiento mono como el estereo
(mono, estereo o componentes Mid-Side) compartan la misma logica de
optimizacion: construir el modelo, el optimizador Adam por grupos, el scheduler
por plateau y el bucle de epochs.

La funcion NO hace IO de archivos: recibe la senal de un canal ya cargada en el
device y devuelve el modelo entrenado, la reconstruccion y los historiales. El
script que la llama se encarga de guardar wavs, curvas, metricas, etc.
"""
import time

import torch

from gs2d_gabor.core.modelo_gabor import GaborAudio1D, construir_optimizador_gabor
from gs2d_gabor.core.perdidas_gabor import loss_gabor, metricas_audio


def entrenar_canal(x, sr, config, device, semilla, etiqueta=""):
    """
    Entrena un modelo Gabor para una senal mono x.

    Args:
        x        : tensor [T] en `device` (float32, rango [-1, 1]).
        sr       : sample rate (int).
        config   : dict de configuracion (mismas claves que el mono).
        device   : torch.device.
        semilla  : semilla para la inicializacion del modelo.
        etiqueta : prefijo para los prints, p.ej. "[L]" / "[R]".

    Returns:
        dict con:
            modelo          : GaborAudio1D entrenado
            x_hat           : tensor [T] reconstruido (detached, en device)
            historial_loss  : list[float]
            tiempos         : list[float] (s por epoch)
            metricas        : dict de metricas_audio del canal
            tiempo_total_s  : float
    """
    T = x.shape[0]

    modelo = GaborAudio1D(
        n_atomos=int(config["n_atomos"]),
        n_samples=T,
        sr=sr,
        device=device,
        sigma_inicial_samples=config.get("sigma_inicial_samples"),
        f_min_hz=float(config.get("f_min_hz", 40.0)),
        f_max_hz=config.get("f_max_hz"),
        k_sigma=float(config.get("k_sigma", 4.0)),
        semilla=int(semilla),
        init_modo=config.get("init_modo", "aleatorio"),
        senal=x,
        init_n_fft=int(config.get("init_n_fft", 2048)),
        init_hop=int(config.get("init_hop", 512)),
        init_alpha=float(config.get("init_alpha", 0.7)),
    )
    print(f"{etiqueta} modelo Gabor: N={modelo.numero_atomos()} atomos  "
          f"k_sigma={modelo.k_sigma}", flush=True)

    if bool(config.get("forzar_pytorch", False)):
        modelo._forzar_pytorch = True
        print(f"{etiqueta} forzando render PyTorch (fallback, lento)", flush=True)

    optimizer = construir_optimizador_gabor(modelo, config.get("lrs"))

    # scheduler simple por plateau
    usar_sched = bool(config.get("usar_scheduler", True))
    sched_factor = float(config.get("scheduler_factor", 0.5))
    sched_paciencia = int(config.get("scheduler_paciencia", 150))
    sched_min_lr = float(config.get("scheduler_min_lr", 1e-5))
    mejor_loss = float("inf")
    epochs_sin_mejora = 0

    n_epochs = int(config["epochs"])
    log_cada = int(config.get("log_cada", 50))

    historial_loss = []
    tiempos = []
    t0 = time.time()

    for epoch in range(n_epochs):
        te = time.time()
        optimizer.zero_grad(set_to_none=True)

        x_hat = modelo.render()
        loss, partes = loss_gabor(x_hat, x, config)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach())
        historial_loss.append(loss_val)
        tiempos.append(time.time() - te)

        if usar_sched:
            if loss_val < mejor_loss - 1e-6:
                mejor_loss = loss_val
                epochs_sin_mejora = 0
            else:
                epochs_sin_mejora += 1
                if epochs_sin_mejora >= sched_paciencia:
                    for grupo in optimizer.param_groups:
                        grupo["lr"] = max(grupo["lr"] * sched_factor, sched_min_lr)
                    epochs_sin_mejora = 0
                    lrs = [g["lr"] for g in optimizer.param_groups]
                    print(f"{etiqueta} [sched] epoch {epoch+1}: LR reducido. "
                          f"min={min(lrs):.2e} max={max(lrs):.2e}", flush=True)

        if (epoch == 0) or ((epoch + 1) % log_cada == 0) or (epoch == n_epochs - 1):
            with torch.no_grad():
                met = metricas_audio(x_hat.detach(), x)
            eta = (time.time() - t0) / (epoch + 1) * (n_epochs - epoch - 1)
            extra = "  ".join(f"{k}={v:.4f}" for k, v in partes.items())
            print(
                f"{etiqueta} epoch {epoch+1:5d}/{n_epochs}  loss={loss_val:.5f}  "
                f"{extra}  SNR={met['snr_db']:.2f}dB  PSNR={met['psnr_db']:.2f}dB  "
                f"t={tiempos[-1]*1000:.0f}ms  eta={eta/60:.1f}min",
                flush=True,
            )

    # gain matching post-hoc: g optimo de minimos cuadrados. Garantiza que la
    # escala/POLARIDAD global sea la correcta (SNR ~ SI-SDR), corrigiendo el caso
    # en que el gain optimizado quedo atascado en el signo equivocado por la
    # barrera en gain=0.
    with torch.no_grad():
        x_hat = modelo.render()
        g = torch.dot(x_hat, x) / torch.dot(x_hat, x_hat).clamp_min(1e-12)
        modelo.gain.data = modelo.gain.data * g
        x_hat = modelo.render()
        met_final = metricas_audio(x_hat, x)

    return {
        "modelo": modelo,
        "x_hat": x_hat.detach(),
        "historial_loss": historial_loss,
        "tiempos": tiempos,
        "metricas": met_final,
        "tiempo_total_s": time.time() - t0,
    }
