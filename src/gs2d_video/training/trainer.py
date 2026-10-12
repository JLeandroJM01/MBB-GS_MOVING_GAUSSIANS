"""
Entrenamiento del modelo de video con acumulacion de gradientes.

Soporta frames en CPU, muestreo temporal, checkpoints, metricas opcionales,
scheduler por plateau y rasterizacion CUDA tiled.
"""

import os
import time

import torch

from gs2d_video.core.perdidas import loss_render_frame, loss_smoothness
from gs2d_video.io.frames import frame_a_device as _frame_a_device
from gs2d_video.render.renderer import render_frame, loss_frame_cuda


# ============================================================
# Muestreo temporal por bloques
# ============================================================

class MuestreadorTemporalPorBloques:
    """
    Muestreador temporal estratificado sin reemplazo.

    Divide el video en bloques consecutivos de tamano frames_por_bloque.
    En cada epoch toma muestras_por_bloque frames de cada bloque.
    Dentro de cada bloque, no repite frames hasta agotar el bloque completo.
    """

    def __init__(self, n_frames, frames_por_bloque=30, muestras_por_bloque=3, seed=42):
        self.n_frames = int(n_frames)
        self.frames_por_bloque = int(frames_por_bloque)
        self.muestras_por_bloque = int(muestras_por_bloque)

        if self.frames_por_bloque <= 0:
            raise ValueError("frames_por_bloque debe ser > 0")
        if self.muestras_por_bloque <= 0:
            raise ValueError("muestras_por_bloque debe ser > 0")

        self.generador = torch.Generator(device="cpu").manual_seed(int(seed))

        self.bloques = []
        self.ordenes = []
        self.posiciones = []

        for inicio in range(0, self.n_frames, self.frames_por_bloque):
            fin = min(inicio + self.frames_por_bloque, self.n_frames)
            indices = list(range(inicio, fin))

            self.bloques.append(indices)
            self.ordenes.append(self._nuevo_orden(len(indices)))
            self.posiciones.append(0)

    def _nuevo_orden(self, n):
        return torch.randperm(n, generator=self.generador).tolist()

    def siguiente_epoch(self):
        indices_epoch = []

        for b, indices_bloque in enumerate(self.bloques):
            n = len(indices_bloque)
            tomar_total = min(self.muestras_por_bloque, n)

            elegidos_bloque = []

            while len(elegidos_bloque) < tomar_total:
                pos = self.posiciones[b]
                orden = self.ordenes[b]

                restantes = n - pos
                faltan = tomar_total - len(elegidos_bloque)
                tomar = min(restantes, faltan)

                if tomar > 0:
                    locales = orden[pos:pos + tomar]
                    elegidos_bloque.extend(indices_bloque[i] for i in locales)
                    self.posiciones[b] += tomar

                if self.posiciones[b] >= n:
                    self.ordenes[b] = self._nuevo_orden(n)
                    self.posiciones[b] = 0

            indices_epoch.extend(elegidos_bloque)

        perm = torch.randperm(len(indices_epoch), generator=self.generador).tolist()
        return [indices_epoch[i] for i in perm]


# ============================================================
# Utilidades
# ============================================================

def _psnr(a, b):
    mse = torch.mean((a - b) ** 2)
    if mse <= 0:
        return float("inf")
    return float(-10.0 * torch.log10(mse))


def _rasterizar_segun_config(params_j, H, W, config):
    return render_frame(params_j, H, W, config)


@torch.no_grad()
def _evaluar_psnr_promedio(modelo, frames, matrices_base, config):
    """
    PSNR promedio usando el rasterizador indicado por config.

    Compatible con:
    - frames en CUDA float32
    - frames en CPU uint8/float
    """
    n_frames, H, W, _ = frames.shape
    device_modelo = modelo.mu_a0.device

    suma = 0.0

    for j in range(n_frames):
        params_j = modelo.evaluar_en_frame(j, matrices_base)
        r = _rasterizar_segun_config(params_j, H, W, config).clamp(0, 1)

        target_j = _frame_a_device(frames[j], device_modelo)
        suma += _psnr(r, target_j)

        del params_j, r, target_j

    return suma / n_frames


def _indices_epoch(n_frames, frames_por_epoch):
    """
    Devuelve los indices de frames a usar en un epoch.
    Si frames_por_epoch es None, usa todos los frames.
    """
    if frames_por_epoch is None:
        return list(range(n_frames))

    k = int(frames_por_epoch)
    if k <= 0 or k >= n_frames:
        return list(range(n_frames))

    return torch.randperm(n_frames, device="cpu")[:k].tolist()


# ============================================================
# Entrenamiento principal
# ============================================================

def entrenar_batch_full(modelo, frames, matrices_base, optimizer, config, carpeta_salida=None):
    """
    Entrena el modelo con gradient accumulation.

    Flags utiles en config:
        calcular_psnr_durante_training: bool
        guardar_checkpoints_intermedios: bool
        guardar_verificacion_visual: bool
        frames_por_epoch: int | null
        usar_loss_cuda: bool
        loss_cuda_tipo: "l1" | "mse"
        usar_scheduler_lento: bool
        scheduler_lento_window: int
        scheduler_lento_min_delta: float
        scheduler_lento_factor: float
        scheduler_lento_min_lr: float
        scheduler_lento_cooldown: int

        usar_muestreo_temporal_por_bloques: bool
        fps_grupo_temporal: int
        frames_por_grupo_temporal: int
    """

    n_frames, H, W, _ = frames.shape
    device_modelo = modelo.mu_a0.device

    # ============================================================
    # Config general
    # ============================================================
    n_epochs = int(config["n_epochs"])
    beta = float(config["beta_smoothness"])
    pesos_smooth = config.get("pesos_smoothness", None)
    chk_each = int(config.get("checkpoint_cada_n_epochs", 50))

    sub_batch_fr = int(config.get("sub_batch_frames", 1))
    early_plateau = config.get("early_stop_plateau", None)
    frames_por_epoch = config.get("frames_por_epoch", None)

    usar_muestreo_temporal = bool(config.get("usar_muestreo_temporal_por_bloques", False))

    muestreador_temporal = None

    if usar_muestreo_temporal:
        frames_por_bloque = int(config.get("fps_grupo_temporal", 30))
        muestras_por_bloque = int(config.get("frames_por_grupo_temporal", 3))
        seed_muestreo = int(config.get("seed", 42))

        muestreador_temporal = MuestreadorTemporalPorBloques(
            n_frames=n_frames,
            frames_por_bloque=frames_por_bloque,
            muestras_por_bloque=muestras_por_bloque,
            seed=seed_muestreo,
        )

        print(
            f"[trainer] muestreo temporal por bloques activado "
            f"(bloque={frames_por_bloque} frames, "
            f"muestras_por_bloque={muestras_por_bloque}, "
            f"bloques={len(muestreador_temporal.bloques)})",
            flush=True,
        )

    # ============================================================
    # Loss
    # ============================================================
    tipo_loss = str(config.get("tipo_loss", "baseline")).lower().strip()
    lambda_dssim = float(config.get("lambda_dssim", 0.2))

    usar_loss_cuda = bool(config.get("usar_loss_cuda", False))
    loss_cuda_tipo = config.get("loss_cuda_tipo", "l1")

    # Agregacion del error entre frames
    # exponente_frame: p-norm sobre los losses de frame.
    #   p=1  -> mean(L_j)
    #   p=2  -> mean(L_j^2)
    #   p=4+ -> casi max(L_j)
    # usar_max_frame: si true, dentro de cada sub_batch usa max(L_j) en vez de sum.
    exponente_frame = float(config.get("exponente_frame", 1.0))
    usar_max_frame = bool(config.get("usar_max_frame", False))

    # El loss CUDA fusionado solo sirve para losses simples.
    # Para motion/hard/edge/temporal/combo usamos PyTorch encima del render CUDA.
    tipos_cuda_simples = ("baseline", "default", "l1", "l1_dssim")

    if usar_loss_cuda and (tipo_loss not in tipos_cuda_simples or lambda_dssim != 0.0):
        raise RuntimeError(
            "usar_loss_cuda=true solo soporta tipo_loss baseline/l1 y lambda_dssim=0.0. "
            "Para motion, hard, edge, temporal o DSSIM usa usar_loss_cuda=false."
        )

    if usar_loss_cuda:
        print(f"[trainer] usando loss CUDA fusionada tipo={loss_cuda_tipo}", flush=True)
    else:
        print(
            f"[trainer] usando loss PyTorch tipo_loss={tipo_loss} "
            f"lambda_dssim={lambda_dssim}",
            flush=True,
        )

    if exponente_frame != 1.0 or usar_max_frame:
        print(
            f"[trainer] agregacion frames: exponente_frame={exponente_frame}  "
            f"usar_max_frame={usar_max_frame}",
            flush=True,
        )

    temporal_activo = (
        tipo_loss in ("temporal", "motion_temporal", "combo")
        and float(config.get("lambda_temporal", 0.0)) > 0.0
    )

    motion_activo = (
        tipo_loss in ("motion", "motion_temporal", "combo")
        and float(config.get("lambda_motion", 0.0)) > 0.0
    )

    necesita_prev_target = temporal_activo or motion_activo

    if temporal_activo:
        print("[trainer] loss temporal activo: se renderiza frame previo cuando haga falta", flush=True)

    # ============================================================
    # Scheduler
    # ============================================================
    usar_scheduler_lento = bool(config.get("usar_scheduler_lento", False))
    scheduler_lento_window = int(config.get("scheduler_lento_window", 40))
    scheduler_lento_min_delta = float(config.get("scheduler_lento_min_delta", 3e-4))
    scheduler_lento_factor = float(config.get("scheduler_lento_factor", 0.5))
    scheduler_lento_min_lr = float(config.get("scheduler_lento_min_lr", 1e-6))
    scheduler_lento_cooldown = int(config.get("scheduler_lento_cooldown", scheduler_lento_window))

    ultimo_epoch_reduce_lr = -10**9

    if usar_scheduler_lento:
        print(
            f"[trainer] scheduler lento activado "
            f"(window={scheduler_lento_window}, "
            f"min_delta={scheduler_lento_min_delta}, "
            f"factor={scheduler_lento_factor}, "
            f"min_lr={scheduler_lento_min_lr}, "
            f"cooldown={scheduler_lento_cooldown})",
            flush=True,
        )

    # ============================================================
    # Renderer CUDA
    # ============================================================
    usar_cuda_conic = bool(config.get("usar_cuda_conic", False))
    usar_cuda_tiled = bool(config.get("usar_cuda_tiled", True))
    tile_size = int(config.get("cuda_tile_size", 16))
    k_sigma = float(config.get("cuda_k_sigma", 3.5))

    if usar_cuda_conic and usar_cuda_tiled:
        raise RuntimeError("No puedes usar usar_cuda_conic=true y usar_cuda_tiled=true al mismo tiempo")

    if not usar_cuda_tiled:
        raise RuntimeError("El entrenamiento requiere usar_cuda_tiled=true")

    print(
        f"[trainer] usando rasterizador CUDA tiled diferenciable "
        f"(tile={tile_size}, k_sigma={k_sigma})",
        flush=True,
    )

    if frames_por_epoch is not None:
        print(
            f"[trainer] modo frames_por_epoch={frames_por_epoch} "
            f"(subset temporal por epoch)",
            flush=True,
        )

    # ============================================================
    # Flags de salida/debug
    # ============================================================
    calcular_psnr = bool(config.get("calcular_psnr_durante_training", False))
    guardar_ckpts = bool(config.get("guardar_checkpoints_intermedios", False))
    guardar_verif = bool(config.get("guardar_verificacion_visual", False))

    # ============================================================
    # Historial
    # ============================================================
    losses_render = []
    losses_render_raw_mean = []
    losses_render_raw_max = []
    losses_smooth = []
    losses_total = []
    psnrs_chk = []
    tiempos = []

    plateau_count = 0
    mejor_loss = float("inf")
    t_inicio = time.time()

    def _lr_info():
        lrs_actuales = [g["lr"] for g in optimizer.param_groups]
        return f"  lr_min={min(lrs_actuales):.2e} lr_max={max(lrs_actuales):.2e}"

    # ============================================================
    # Loop principal
    # ============================================================
    for epoch in range(n_epochs):
        t_epoch = time.time()
        optimizer.zero_grad(set_to_none=True)

        if muestreador_temporal is not None:
            idx_epoch = muestreador_temporal.siguiente_epoch()
        else:
            idx_epoch = _indices_epoch(n_frames, frames_por_epoch)

        n_usados = len(idx_epoch)

        # Trackers para reporte:
        # - total: loss efectivo que entra al gradiente.
        # - raw_*: metricas interpretables del loss por frame sin exponente.
        # Los acumuladores se crean en el device del modelo.
        loss_render_epoch_total = torch.zeros((), device=device_modelo)
        loss_render_epoch_raw_sum = torch.zeros((), device=device_modelo)
        loss_render_epoch_raw_max = torch.zeros((), device=device_modelo)

        # --------------------------------------------------------
        # Render + loss con gradient accumulation
        # --------------------------------------------------------
        for inicio in range(0, n_usados, sub_batch_fr):
            sub_indices = idx_epoch[inicio:inicio + sub_batch_fr]
            loss_sub = None

            def _acumular(l_j):
                """
                Aplica exponente_frame y agrega l_j al sub-batch loss.
                Tambien actualiza los trackers raw (mean / max) para reporte.
                """
                nonlocal loss_sub, loss_render_epoch_raw_sum, loss_render_epoch_raw_max

                l_j_det = l_j.detach()
                loss_render_epoch_raw_sum = loss_render_epoch_raw_sum + l_j_det
                loss_render_epoch_raw_max = torch.maximum(loss_render_epoch_raw_max, l_j_det)

                if exponente_frame != 1.0:
                    l_j_eff = l_j.clamp_min(1e-12) ** exponente_frame
                else:
                    l_j_eff = l_j

                l_j_norm = l_j_eff / n_usados

                if loss_sub is None:
                    loss_sub = l_j_norm
                elif usar_max_frame:
                    # Solo el peor frame del sub-batch contribuye al gradiente.
                    loss_sub = torch.maximum(loss_sub, l_j_norm)
                else:
                    loss_sub = loss_sub + l_j_norm

            if usar_loss_cuda:
                # Camino rapido: solo L1/MSE fusionado en CUDA.
                for j in sub_indices:
                    params_j = modelo.evaluar_en_frame(j, matrices_base)
                    target_j = _frame_a_device(frames[j], device_modelo)

                    l_j = loss_frame_cuda(
                        params_j,
                        target_j,
                        H,
                        W,
                        config,
                    )
                    _acumular(l_j)

                    del params_j, target_j

            else:
                # Camino flexible: render CUDA + loss PyTorch configurable.
                render_cache = {}
                target_cache = {}

                # Renderizar frames actuales del sub-batch.
                for j in sub_indices:
                    params_j = modelo.evaluar_en_frame(j, matrices_base)
                    render_cache[j] = _rasterizar_segun_config(
                        params_j,
                        H,
                        W,
                        config,
                    )
                    target_cache[j] = _frame_a_device(frames[j], device_modelo)

                    del params_j

                # Calcular loss por frame.
                for j in sub_indices:
                    prev_render = None
                    prev_target = None

                    if necesita_prev_target and j > 0:
                        if (j - 1) in target_cache:
                            prev_target = target_cache[j - 1]
                        else:
                            prev_target = _frame_a_device(frames[j - 1], device_modelo)

                    if temporal_activo and j > 0:
                        if (j - 1) in render_cache:
                            prev_render = render_cache[j - 1]
                        else:
                            params_prev = modelo.evaluar_en_frame(j - 1, matrices_base)
                            prev_render = _rasterizar_segun_config(
                                params_prev,
                                H,
                                W,
                                config,
                            )
                            del params_prev

                    l_j = loss_render_frame(
                        render_cache[j],
                        target_cache[j],
                        config,
                        frames_all=None,
                        frame_idx=j,
                        prev_render=prev_render,
                        prev_target=prev_target,
                    )
                    _acumular(l_j)

            if loss_sub is not None:
                loss_sub.backward()
                loss_render_epoch_total = loss_render_epoch_total + loss_sub.detach() * n_usados

            # Liberar referencias del sub-batch.
            if not usar_loss_cuda:
                del render_cache
                del target_cache

        # loss_render_avg = promedio del loss EFECTIVO (con exponente_frame).
        # Si exponente_frame=1 y usar_max_frame=false, coincide con la version anterior.
        loss_render_avg = float((loss_render_epoch_total / max(1, n_usados)).item())

        # loss_render_raw_mean / raw_max = metricas interpretables del L_j RAW
        # sin aplicar exponente_frame.
        loss_render_raw_mean = float((loss_render_epoch_raw_sum / max(1, n_usados)).item())
        loss_render_raw_max = float(loss_render_epoch_raw_max.item())
        # --------------------------------------------------------
        # Smoothness
        # --------------------------------------------------------
        loss_smooth = loss_smoothness(modelo, pesos_por_param=pesos_smooth)
        loss_smooth_escalado = beta * loss_smooth

        if loss_smooth_escalado.requires_grad and beta != 0.0:
            loss_smooth_escalado.backward()

        # --------------------------------------------------------
        # Optimizer step
        # --------------------------------------------------------
        optimizer.step()

        # --------------------------------------------------------
        # Reporte numerico
        # --------------------------------------------------------
        loss_render_reportado = float(loss_render_avg)
        loss_smooth_reportado = float(loss_smooth.detach().item())
        loss_total_reportado = loss_render_reportado + beta * loss_smooth_reportado

        losses_render.append(loss_render_reportado)
        losses_render_raw_mean.append(loss_render_raw_mean)
        losses_render_raw_max.append(loss_render_raw_max)
        losses_smooth.append(loss_smooth_reportado)
        losses_total.append(loss_total_reportado)

        # --------------------------------------------------------
        # Scheduler lento despues del optimizer.step
        # --------------------------------------------------------
        if usar_scheduler_lento and len(losses_render) > scheduler_lento_window:
            loss_antes = losses_render[-scheduler_lento_window - 1]
            loss_ahora = losses_render[-1]
            mejora_ventana = loss_antes - loss_ahora

            paso_cooldown = (epoch + 1) - ultimo_epoch_reduce_lr >= scheduler_lento_cooldown

            if mejora_ventana < scheduler_lento_min_delta and paso_cooldown:
                for grupo in optimizer.param_groups:
                    lr_actual = grupo["lr"]
                    nuevo_lr = max(lr_actual * scheduler_lento_factor, scheduler_lento_min_lr)
                    grupo["lr"] = nuevo_lr

                ultimo_epoch_reduce_lr = epoch + 1

                lrs_actuales = [g["lr"] for g in optimizer.param_groups]
                print(
                    f"[trainer] scheduler lento redujo LR en epoch {epoch + 1}: "
                    f"mejora_ventana={mejora_ventana:.6f} < {scheduler_lento_min_delta:.6f} | "
                    f"lr_min={min(lrs_actuales):.2e} lr_max={max(lrs_actuales):.2e}",
                    flush=True,
                )

        tiempos.append(time.time() - t_epoch)

        # ========================================================
        # Logs/checkpoints
        # ========================================================
        es_checkpoint = (
            (epoch + 1) % chk_each == 0
            or epoch == 0
            or epoch == n_epochs - 1
        )

        if es_checkpoint:
            if calcular_psnr:
                psnr_avg = _evaluar_psnr_promedio(
                    modelo,
                    frames,
                    matrices_base,
                    config,
                )
            else:
                psnr_avg = -1.0

            psnrs_chk.append((epoch, psnr_avg))

            t_corrido = time.time() - t_inicio
            t_promedio = t_corrido / (epoch + 1)
            eta = t_promedio * (n_epochs - epoch - 1)

            print(
                f"  epoch {epoch + 1:4d}/{n_epochs}  "
                f"loss_r={loss_render_reportado:.5f}  "
                f"loss_r_mean={loss_render_raw_mean:.5f}  "
                f"loss_r_max={loss_render_raw_max:.5f}  "
                f"loss_s={loss_smooth_reportado:.3e}  "
                f"PSNR_avg={psnr_avg:.2f}  "
                f"t_epoch={tiempos[-1]:.1f}s  "
                f"eta={eta / 60:.1f}min"
                f"{_lr_info()}",
                flush=True,
            )

            if carpeta_salida is not None and guardar_ckpts:
                torch.save({
                    "state_dict_coefs": modelo.state_dict_coefs(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch_completado": epoch + 1,
                    "config": config,
                }, os.path.join(carpeta_salida, f"checkpoint_epoch{epoch + 1:04d}.pt"))

            if carpeta_salida is not None and guardar_verif:
                _guardar_verificacion_visual(
                    modelo,
                    frames,
                    matrices_base,
                    carpeta_salida,
                    epoch + 1,
                    config,
                )

        else:
            log_cada = max(1, chk_each // 5)

            if (epoch + 1) % log_cada == 0:
                t_corrido = time.time() - t_inicio
                t_promedio = t_corrido / (epoch + 1)
                eta = t_promedio * (n_epochs - epoch - 1)

                print(
                    f"  epoch {epoch + 1:4d}/{n_epochs}  "
                    f"loss_r={loss_render_reportado:.5f}  "
                    f"loss_r_mean={loss_render_raw_mean:.5f}  "
                    f"loss_r_max={loss_render_raw_max:.5f}  "
                    f"loss_s={loss_smooth_reportado:.3e}  "
                    f"t_epoch={tiempos[-1]:.1f}s  "
                    f"eta={eta / 60:.1f}min"
                    f"{_lr_info()}",
                    flush=True,
                )

        # ========================================================
        # Early stop manual por plateau
        # ========================================================
        if early_plateau is not None:
            if loss_total_reportado < mejor_loss - 1e-6:
                mejor_loss = loss_total_reportado
                plateau_count = 0
            else:
                plateau_count += 1

            if plateau_count >= early_plateau:
                print(
                    f"  early stop en epoch {epoch + 1} "
                    f"(plateau de {early_plateau} epochs)",
                    flush=True,
                )
                break

    return {
        "losses_render": losses_render,
        "losses_render_raw_mean": losses_render_raw_mean,
        "losses_render_raw_max": losses_render_raw_max,
        "losses_smooth": losses_smooth,
        "losses_total": losses_total,
        "psnrs_chk": psnrs_chk,
        "tiempos_por_epoch": tiempos,
        "tiempo_total": time.time() - t_inicio,
    }


@torch.no_grad()
def _guardar_verificacion_visual(modelo, frames, matrices_base, carpeta, epoch, config):
    """
    Renderiza primer y ultimo frame usando el rasterizador del config.
    Compatible con frames en CPU o GPU.
    """
    from PIL import Image
    import numpy as np

    n_frames, H, W, _ = frames.shape
    sub = os.path.join(carpeta, "verificacion")
    os.makedirs(sub, exist_ok=True)

    for nombre, j in [("primer", 0), ("ultimo", n_frames - 1)]:
        params_j = modelo.evaluar_en_frame(j, matrices_base)
        r = _rasterizar_segun_config(params_j, H, W, config).clamp(0, 1)

        if torch.is_tensor(r):
            r = r.detach().cpu().numpy()

        Image.fromarray((r * 255).astype(np.uint8)).save(
            os.path.join(sub, f"epoch{epoch:04d}_{nombre}_frame.png")
        )