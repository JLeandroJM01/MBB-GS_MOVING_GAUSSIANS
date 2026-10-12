"""
Visualizaciones clave.

1. trayectorias.png             : frame 0 con lineas mu_i(t) superpuestas
2. heatmap_opacity_temporal.png : (N, T) con sigma(opacity_i(t_j))
3. evolucion_parametros.png     : top-5 gaussianas, todos sus parametros vs t
4. coeficientes_magnitudes.png  : histograma de magnitudes a_k por k

El GIF/MP4 de reconstruccion se genera fuera de este modulo:
  - scripts/regenerar_clip_desde_checkpoint_streaming.py rasteriza desde checkpoint.
  - scripts/frames_a_video.py convierte la carpeta de frames en MP4.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt




@torch.no_grad()
def _evaluar_param_todos_frames(a0, hi, matriz_base):
    """coefs @ B.T -> shape (..., n_frames)."""
    coefs = torch.cat([a0, hi], dim=-1)
    return coefs @ matriz_base.T



@torch.no_grad()
def generar_trayectorias_png(modelo, frame_objetivo, matrices_base, ruta,
                              umbral_opacity_visualizar=0.1, max_lineas=200):
    grado_mu = modelo.grados['mu']
    grado_op = modelo.grados['opacity']
    grado_co = modelo.grados['color']

    mu_t    = _evaluar_param_todos_frames(modelo.mu_a0, modelo.mu_high, matrices_base[grado_mu])   # (N, 2, T)
    op_t    = torch.sigmoid(_evaluar_param_todos_frames(modelo.opacity_a0, modelo.opacity_high,
                                                          matrices_base[grado_op]).squeeze(1))      # (N, T)
    color_t = torch.sigmoid(_evaluar_param_todos_frames(modelo.color_a0, modelo.color_high,
                                                          matrices_base[grado_co]))                  # (N, 3, T)

    op_max  = op_t.max(dim=-1).values
    op_mean = op_t.mean(dim=-1)
    color_mean = color_t.mean(dim=-1)

    mascara = op_max > umbral_opacity_visualizar
    idx = mascara.nonzero(as_tuple=False).squeeze(-1)
    if len(idx) > max_lineas:
        orden = torch.argsort(op_mean[idx], descending=True)
        idx = idx[orden[:max_lineas]]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    ax.imshow(frame_objetivo.detach().clamp(0, 1).cpu().numpy())
    mu_np = mu_t.cpu().numpy()
    om = op_mean.cpu().numpy()
    cm = color_mean.cpu().numpy()
    for i in idx.cpu().numpy():
        ax.plot(mu_np[i, 1, :], mu_np[i, 0, :],
                color=tuple(cm[i].clip(0, 1)),
                linewidth=0.3 + 2.5 * om[i],
                alpha=0.8)
    ax.set_xlim(0, frame_objetivo.shape[1])
    ax.set_ylim(frame_objetivo.shape[0], 0)
    ax.axis('off')
    ax.set_title(f"Trayectorias mu_i(t)  (N visibles = {len(idx)})")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)



@torch.no_grad()
def generar_heatmap_opacity(modelo, matrices_base, ruta):
    grado_op = modelo.grados['opacity']
    op_t = torch.sigmoid(_evaluar_param_todos_frames(modelo.opacity_a0, modelo.opacity_high,
                                                       matrices_base[grado_op]).squeeze(1))         # (N, T)
    op_np = op_t.cpu().numpy()
    T = op_np.shape[1]
    eje_t = np.arange(T)
    pesos = op_np / (op_np.sum(axis=1, keepdims=True) + 1e-8)
    tiempo_medio = (pesos * eje_t).sum(axis=1)
    op_ordenado = op_np[np.argsort(tiempo_medio)]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=110)
    im = ax.imshow(op_ordenado, aspect='auto', cmap='inferno', interpolation='nearest')
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("gaussiana (ordenada por t medio de activacion)")
    ax.set_title("sigma(opacity_i(t))")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)



# Selecciona las gaussianas con mayor opacidad promedio.
@torch.no_grad()
def generar_evolucion_parametros(modelo, matrices_base, ruta, n_sel=5):
    grado_op = modelo.grados['opacity']
    op_t = torch.sigmoid(_evaluar_param_todos_frames(modelo.opacity_a0, modelo.opacity_high,
                                                       matrices_base[grado_op]).squeeze(1))
    op_mean = op_t.mean(dim=-1)
    top_idx = torch.argsort(op_mean, descending=True)[:n_sel].cpu().numpy()

    todos = {}
    for nombre, (a0, hi, grado, _) in modelo.parametros_temporales().items():
        v = _evaluar_param_todos_frames(a0, hi, matrices_base[grado])    # (N, dim_p, T)
        if nombre in ('opacity', 'color'):
            v = torch.sigmoid(v)
        elif nombre == 'scale':
            log_min = float(np.log(0.5))
            log_max = float(np.log(max(modelo.H, modelo.W)))
            v = torch.exp(v.clamp(min=log_min, max=log_max))
        todos[nombre] = v.cpu().numpy()

    n_params = len(todos)
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2.0 * n_params), dpi=100, sharex=True)
    if n_params == 1:
        axes = [axes]
    for ax, (nombre, v) in zip(axes, todos.items()):
        for i in top_idx:
            arr = v[i]
            for d in range(arr.shape[0]):
                ax.plot(arr[d],
                        label=f"g{i}-d{d}" if v.shape[1] > 1 else f"g{i}",
                        alpha=0.85)
        ax.set_ylabel(nombre)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("frame_idx")
    axes[0].legend(loc='upper right', fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)



@torch.no_grad()
def generar_coeficientes_magnitudes(modelo, ruta):
    """
    Histograma por parametro: magnitud media |a_k| en funcion de k (para todas
    las gaussianas). Si los k altos son ~0, significa que el modelo no
    necesito grado alto -- info muy valiosa para la tesis.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), dpi=100)
    axes = axes.flatten()

    for ax, (nombre, (a0, hi, grado, _)) in zip(axes, modelo.parametros_temporales().items()):
        # |a_0| y |a_k| para k=1..grado
        coefs_full = torch.cat([a0, hi], dim=-1).detach().cpu()    # (N, dim_p, q+1)
        # mediana de magnitudes sobre N y dim_p, por k
        mag = coefs_full.abs().reshape(-1, coefs_full.shape[-1]).median(dim=0).values.numpy()
        ax.bar(range(len(mag)), mag, color='tab:blue')
        ax.set_yscale('log')
        ax.set_title(f"{nombre}  (grado {grado})")
        ax.set_xlabel("k (orden coef)")
        ax.set_ylabel("mediana |a_k|")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Magnitudes de coeficientes - un k alto indica que el modelo usa ese grado")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)



