# Training

Source: `scripts/train.py`, `src/gs2d_video/training/trainer.py`,
`src/gs2d_video/core/optimizador.py`

```bash
python scripts/train.py --config configs/video/final/ganador_motion_200k_1200ep.json
```

`--nombre-experimento NAME` overrides the name in the config, which is useful
when launching the same configuration several times.

## What happens

1. The config is read and copied verbatim to the output folder.
2. The clip is loaded from `data/clips/<clip>/` as a PNG sequence.
3. One basis matrix is built per distinct degree in `grados`.
4. The model is created; colours can be sampled from frame 0.
5. The optimiser is built with twelve parameter groups.
6. Training runs with gradient accumulation over frames.
7. `checkpoint_final.pt` is written **immediately** after training, before any
   metric is computed, so a failure in evaluation cannot cost you the model.
8. Optional post-training pruning, then rendering and metrics.

## Output layout

```
outputs/<nombre_experimento>/
├── frames_renderizados/     rendered frames, post-pruning
├── checkpoints/             checkpoint_final.pt, modelo_pruneado.pt
├── logs/                    log_entrenamiento.csv, loss curves, visualisations
├── metricas.json            aggregates and per-frame, split pre/post pruning
├── metricas_por_frame.csv   one row per frame
├── config_usada.json        exact config used, overrides applied
└── info_clip.json           clip metadata: frames, resolution, fps, seed
```

By default the run refuses to start if the output folder already exists. Set
`sobreescribir_salida` to reuse it.

## The training loop

Each epoch selects a set of frames, renders and accumulates gradients over them
in sub-batches of `sub_batch_frames`, adds the smoothness penalty, and steps the
optimiser once. Frames are the unit of accumulation, not images in a batch: at
720p with 100k Gaussians a single frame already saturates the GPU.

Two frame-selection modes exist beyond using all frames. `frames_por_epoch`
takes a random subset each epoch. `usar_muestreo_temporal_por_bloques` divides
the clip into consecutive blocks of `fps_grupo_temporal` frames and draws
`frames_por_grupo_temporal` from each, without repeating a frame until its block
is exhausted — stratified sampling that guarantees every part of the timeline is
visited, which matters when the temporal polynomials are high degree.

### Reported metrics

Each log line separates the loss that receives the gradient from interpretable
diagnostics:

- `loss_r` — the effective loss, after `exponente_frame`
- `loss_r_mean` — mean of the raw per-frame losses
- `loss_r_max` — worst frame of the epoch
- `loss_s` — smoothness penalty

When `exponente_frame` is 1, `loss_r` and `loss_r_mean` coincide.

### Learning rate schedule

`usar_scheduler_lento` reduces every group's learning rate by
`scheduler_lento_factor` when the loss has not improved by at least
`scheduler_lento_min_delta` across a window of `scheduler_lento_window` epochs,
respecting a cooldown. It is a plateau scheduler with a deliberately long
window, since these runs are thousands of epochs long and the loss is noisy at
short scales.

## The optimiser

Adam with **twelve** parameter groups: six attributes × {`a0`, `a_high`}. Base
values and temporal coefficients need different step sizes, so they get separate
rates:

```json
"lrs": {
  "mu_a0": 0.001,    "mu_high": 0.0075,
  "opacity_a0": 0.05, "opacity_high": 0.04,
  "color_a0": 0.01,   "color_high": 0.008,
  "scale_a0": 0.005,  "scale_high": 0.004,
  "theta_a0": 0.005,  "theta_high": 0.0025,
  "depth_a0": 0.001,  "depth_high": 0.0001
}
```

Any missing key falls back to a default. The fused Adam implementation is used
when available, then `foreach`, then the plain one.

## Memory

A clip of $T$ frames at 720p in float32 needs about
$T \times 720 \times 1280 \times 3 \times 4$ bytes just to hold the frames —
around 11 MB per frame, so 600 frames is 6.6 GB before the model exists.

Three storage modes are available:

| Mode | Where frames live | When to use |
| --- | --- | --- |
| default | GPU float32 | small clips only |
| `frames_en_gpu_uint8` | GPU uint8 | large VRAM; no per-frame transfer |
| `frames_en_cpu` | CPU uint8, pinned | the usual choice for 720p |

Only the frame in use is converted to float32 on the device. The two flags are
mutually exclusive and the code rejects setting both.

Recommended for large trainings:

```json
"frames_en_cpu": true,
"frames_en_gpu_uint8": false,
"evitar_render_completo_en_train": true,
"usar_metricas_streaming": true
```

`evitar_render_completo_en_train` stops the trainer from stacking the whole clip
in GPU memory at the end. `usar_metricas_streaming` then renders, saves and
measures one frame at a time, which is what makes 720p with 100k Gaussians
feasible on 8 GB of VRAM. In streaming mode the GIF and the compression metrics
are disabled automatically, since both require the full stack in memory; use
`scripts/reconstruction/` afterwards instead.

If you hit out-of-memory, reduce `n_gaussianas_inicial`, the resolution, the
number of frames, `sub_batch_frames`, or the temporal degrees. Note that keeping
frames on CPU trades VRAM for system RAM, so check which one actually ran out.

## Configuration reference

### Clip and run

| Key | Meaning |
| --- | --- |
| `nombre_experimento` | output folder name; falls back to a timestamp |
| `clip` | folder under `data/clips/` |
| `max_frames` | truncate the clip |
| `video_mp4`, `fps_extraccion`, `n_frames_extraer`, `resolucion_extraccion` | extract frames automatically before training |
| `device` | `cuda`, `cpu` or `mps`; training requires CUDA |
| `seed` | seeds torch and the model's own generator |
| `sobreescribir_salida` | reuse an existing output folder |

### Model

| Key | Meaning |
| --- | --- |
| `n_gaussianas_inicial` | $N$ |
| `base_temporal` | `chebyshev` (main) or `monomial` (ablation) |
| `grados` | polynomial degree per attribute |
| `escala_inicial_px` | initial scale; its log becomes `scale`'s `a_0` |
| `inicializar_color_desde_frame0` | sample initial colours from frame 0 |
| `checkpoint_inicial`, `cargar_optimizer_state` | resume from a checkpoint |

### Optimisation

`n_epochs`, `lrs`, `sub_batch_frames`, `frames_por_epoch`,
`early_stop_plateau`, `beta_smoothness`, `pesos_smoothness`, and the
`scheduler_lento_*` family.

### Loss

`tipo_loss`, `lambda_dssim`, `lambda_motion`, `lambda_mse`, `lambda_hard`,
`lambda_edge`, `lambda_temporal`, `motion_umbral`, `motion_blur`,
`motion_clip`, `hard_clip`, `exponente_pixel`, `exponente_frame`,
`usar_max_pixel`, `usar_max_frame`, `usar_pnorm_root`, `usar_loss_cuda`,
`loss_cuda_tipo`. See [Loss Functions](Loss-Functions).

### Rasteriser

| Key | Meaning |
| --- | --- |
| `usar_cuda_tiled` | must be true; training requires it |
| `cuda_tile_size` | tile side in pixels, typically 16 |
| `cuda_k_sigma` | how many standard deviations a Gaussian reaches |

### Pruning and outputs

`ejecutar_pruning_post`, `umbral_pruning_post`, `pruning_n_samples`,
`calcular_metricas`, `usar_ssim`, `usar_lpips`, `calcular_compresion`,
`calidades_avif`, `guardar_frames_rasterizados`, `guardar_visualizaciones`,
`guardar_gif`, `guardar_checkpoints_intermedios`,
`guardar_verificacion_visual`, `checkpoint_cada_n_epochs`.

## Running on a cluster

`jobs/video/` holds the Slurm scripts used on Khipu, with the exact partition,
memory and time limits of each reported run; `jobs/audio/` holds the equivalent
ones for the Gabor trainings. They activate a conda environment and call
`scripts/train.py`. Remember that the environment needs `pip install -e .` once;
see [Installation](Installation).
