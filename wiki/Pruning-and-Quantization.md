# Pruning and Quantization

Because every Gaussian owns an explicit, inspectable set of coefficients, the
model can be reduced after training without retraining anything. Two independent
mechanisms do this: dropping Gaussians, and storing the remaining coefficients
in fewer bits.

## Post-training pruning inside training

Source: `src/gs2d_video/core/pruning_post.py`

The cheap version, run automatically at the end of `train.py` when
`ejecutar_pruning_post` is true. It computes, for each Gaussian,

$$\max_t \sigma\big(\alpha_i(t)\big)$$

sampled at `pruning_n_samples` instants, and removes every Gaussian that never
crosses `umbral_pruning_post` at any point in the clip. The reasoning is that a
Gaussian which is never visible cannot be contributing anything.

Taking the maximum over time, rather than the mean, is what protects
**transient** Gaussians — primitives that are invisible most of the clip but
carry a short, important event.

The temporal basis used to evaluate opacity must be the one the model was
trained with. `prunear_post` takes it as an argument and `train.py` passes
whatever the config resolved; the default is Chebyshev.

## Adaptive pruning

Source: `scripts/compression/run_binary_pruning_adaptativo.py`

The serious version, used for the reported reduction results. Despite the file
name it is not a classical binary search: it evaluates increasing percentages
and keeps the largest reduction that still meets a quality bar.

```
full checkpoint
      ↓
per-Gaussian statistics
      ↓
adaptive ranking
      ↓
try a percentage → render → compare against baseline
      ↓                              ↓
  PSNR sufficient              PSNR insufficient
  keep and continue            step back
```

### Statistics

Produced by `scripts/visualization/viz_gaussian_stats.py`:
`path_length_px`, `color_path`, `op_mean`, `op_max`, `op_std`, `active_frac`,
`scale_std`.

### Protecting transient Gaussians

Gaussians with `op_max >= 0.90` and `active_frac <= 0.10` are excluded from
removal. These are precisely the primitives that are highly visible for a brief
moment — a ranking based on averages would discard them, and they are often the
ones carrying a visible event.

### The ranking

Candidates are those with `op_mean >= 0.05`, minus the protected set. Each
metric is converted to a percentile rank and combined:

```
0.30 * path_length_px
0.25 * color_path
0.25 * op_std
0.20 * scale_std
```

The lowest scores are removed first. This is deliberately **not** ranking by
opacity alone: a Gaussian that is visible but never moves, never changes colour
and never changes size is contributing something a smaller, cheaper
representation could also contribute.

### Running it

```bash
python scripts/compression/run_binary_pruning_adaptativo.py \
    --exp outputs/mi_experimento \
    --checkpoint outputs/mi_experimento/checkpoints/checkpoint_final.pt \
    --baseline_frames outputs/mi_experimento/frames_renderizados \
    --fps 30 --device cuda \
    --inicio_pct 10 --paso_pct 5 --min_pct 10 --max_pct 50 \
    --psnr_min 65 --crear_video_ganador
```

Acceptance uses global PSNR, computed from the mean MSE across frames, against
the baseline reconstruction — not against the original video. The question being
asked is "how much can I remove before the output changes", not "how good is the
model".

Results land in `outputs/<exp>/binary_pruning/`, with the winning checkpoint
kept separate from the original.

## UINT16 quantization

Per-tensor affine quantization to 16-bit integers:

$$q = \text{round}\left(\frac{x - x_{\min}}{s}\right), \quad
s = \frac{x_{\max} - x_{\min}}{65535}, \quad
\hat{x} = x_{\min} + s\,q$$

Each tensor gets its own range, so all 65536 levels are spent inside the values
that tensor actually takes.

### SAFE

`scripts/compression/pack_checkpoint_uint16.py` and its `unpack` counterpart.
Quantizes the bulky high-order coefficients — `mu_high`, `color_high`,
`opacity_high`, `scale_high` — and leaves the rest in float32. This is the
variant the audiovisual pipeline uses.

```bash
python scripts/compression/pack_checkpoint_uint16.py \
    --in_ckpt modelo_pruneado.pt \
    --out_pkg modelo_uint16_safe.pkg.pt \
    --other_float fp32 \
    --quant_tensors mu_high color_high opacity_high scale_high \
    --omit_zero_depth_high

python scripts/compression/unpack_checkpoint_uint16.py \
    --in_pkg modelo_uint16_safe.pkg.pt \
    --out_ckpt modelo_uint16_safe_render.pt \
    --out_float fp32
```

### ALL

`pack_checkpoint_uint16_all.py` quantizes everything:

```bash
python scripts/compression/pack_checkpoint_uint16_all.py \
    --in_ckpt modelo_pruneado.pt \
    --out_pkg modelo_uint16_all.pkg.pt \
    --omit_zero_depth_high
```

### Omitting `depth_high`

When `depth_high` is all zeros within a small tolerance — which happens whenever
depth was given a low degree and never needed to move — it is dropped from the
package and rebuilt as zeros on unpacking. It is kept if it holds anything
meaningful.

## Validating a reduction

Smaller is not the same as correct. The only acceptable check is round-trip:

```
FP32 model → quantize → dequantize → render → compare against baseline
```

The pipeline computes PSNR between the quantized frames and the reference
reconstruction. A reduction is only reported once that number holds up.

## Reported result

Starting from 150k Gaussians, 750 frames at 720p: adaptive pruning at 20 %
brings the model to 120k Gaussians, and quantization takes it from roughly
236 MB to 85 MB, at about 63 dB PSNR against the full reconstruction.
