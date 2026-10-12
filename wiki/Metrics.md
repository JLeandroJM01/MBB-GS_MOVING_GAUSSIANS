# Metrics

Source: `src/gs2d_video/metrics/`, `scripts/metrics/`

## What is measured

| Metric | Meaning | Better |
| --- | --- | --- |
| PSNR | log of the inverse MSE, `data_range = 1.0` | higher |
| SSIM | structural similarity, 11×11 Gaussian window | higher |
| LPIPS | perceptual distance, AlexNet backbone | lower |
| Temporal PSNR | PSNR between consecutive-frame *differences* | higher |

Temporal PSNR is the one specific to this work. It compares how the
reconstruction changes from frame to frame against how the original changes, so
it measures temporal coherence rather than per-frame quality. A model can score
well on PSNR while flickering; temporal PSNR is what catches that.

Each metric is reported per frame and aggregated as mean, min, max, 5th
percentile and standard deviation. The minimum and the 5th percentile matter as
much as the mean: they say whether quality is uniform across the clip or whether
some frames are much worse.

## Streaming versus stacked

`reporte_completo` takes the whole rendered clip and the whole ground truth as
tensors. Simple, and only viable for small clips.

`reporte_completo_streaming` takes a function that yields one
`(render, target)` pair at a time and accumulates the statistics as it goes.
Nothing beyond a single frame is ever held. This is what `train.py` uses when
`usar_metricas_streaming` is on, and it is what makes evaluating a 720p clip
with 100k Gaussians possible on 8 GB of VRAM.

Both paths produce the same report structure.

## Cost

SSIM and especially LPIPS are expensive, and LPIPS downloads AlexNet weights on
first use — which fails on a compute node with no internet, a common situation
on clusters. Both can be disabled:

```json
"usar_ssim": false,
"usar_lpips": false
```

For a run that finished with LPIPS missing, compute it afterwards on the saved
frames:

```bash
python scripts/metrics/lpips_post_hoc.py --exp outputs/mi_experimento
```

It reads the rendered frames and the originals named in `config_usada.json`, and
updates the metric files in place.

## Comparing two folders of frames

```bash
python scripts/metrics/comparar_frames_psnr.py --a carpeta_a --b carpeta_b
```

Used to compare a reduced or quantized model against its baseline
reconstruction, which is how [pruning](Pruning-and-Quantization) decides whether
a reduction is acceptable.

## Compression metrics

`src/gs2d_video/metrics/compresion.py` reports the model size in bytes and
compares it against the original video and against saving every frame as AVIF at
several quality levels, via `pillow-avif-plugin`. This gives a reference point:
how does an explicit temporal representation compare to simply compressing the
frames well.

AVIF is optional; if the plugin is absent that part of the report is skipped.
Compression metrics need the full stack of renders in memory, so they are
disabled automatically in streaming mode — run them from a reconstruction
afterwards.

## Output files

```
metricas.json            aggregates and per-frame, split pre/post pruning
metricas_por_frame.csv   frame_idx, psnr, ssim, lpips, psnr_temporal
metricas_compresion.json model size, AVIF comparisons
```

`metricas.json` keeps the pre-pruning and post-pruning reports separate, so the
effect of pruning is visible rather than folded into a single number. In
streaming mode the pre-pruning report is left empty on purpose: filling it would
mean rasterising the whole clip a second time with the unpruned model, and the
post-pruning streaming pass already measures the model that is actually being
shipped.
