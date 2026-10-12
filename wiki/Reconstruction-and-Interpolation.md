# Reconstruction and Interpolation

Source: `scripts/reconstruction/`, `src/gs2d_video/io/checkpoints.py`

## The central checkpoint loader

Every tool that reads a checkpoint goes through
`gs2d_video.io.checkpoints.cargar_modelo_desde_checkpoint`. It rebuilds the
`GaussianasPolinomial2D` object, reads `N`, `H`, `W`, `n_frames`, the degrees,
the initial scale and the seed, resolves the device, and builds the basis
matrices for the temporal basis the checkpoint declares.

Having one loader matters: reconstruction, interpolation, pruning and the
visualisations all interpret a checkpoint identically, so none of them can drift
into reading metadata a different way.

Accepted basis names are `chebyshev`/`cheby`/`cheb` and `monomial`/`mono`,
normalised internally. A checkpoint with no `base_temporal` field is treated as
Chebyshev, which keeps older experiments loadable.

If `H`, `W` or `n_frames` cannot be found in the checkpoint, the loader looks
for `info_clip.json` next to it in the experiment folder.

## Rendering a clip from a checkpoint

```bash
python scripts/reconstruction/regenerar_clip_desde_checkpoint_streaming.py \
    --checkpoint outputs/mi_experimento/checkpoints/checkpoint_final.pt \
    --salida outputs/mi_experimento/frames_renderizados \
    --device cuda
```

Streaming means one frame at a time: render, write the PNG, release. Nothing is
stacked in GPU memory, which is what makes 720p with 100k Gaussians possible on
a modest GPU. It also writes `metricas_streaming_simples.csv` with MSE, MAE and
PSNR per frame when the original clip is available.

Useful options: `--inicio` and `--fin` to render a range, `--comparaciones` to
emit `original | render | difference` triptychs with `--factor_diff` amplifying
the difference, `--crear_video --fps 30` to produce an MP4 directly, and
`--limpiar_cache_cada` to control how often the CUDA cache is released.

To turn an existing folder of frames into a video:

```bash
python scripts/data/frames_a_video.py \
    --frames outputs/mi_experimento/frames_renderizados \
    --salida outputs/mi_experimento/video_reconstruido.mp4 --fps 30
```

## Temporal interpolation

This is where the representation pays off. Attributes are continuous functions
of time, so the model can be evaluated at instants that were never frames.

Concretely: instead of building the basis matrix at
$\tau(j) = 2j/(T-1) - 1$ for integer $j$, build it at intermediate values of
$\tau$. The **same learned coefficients** are evaluated at new points. No second
network, no optical flow, no training.

```bash
python scripts/reconstruction/regenerar_fps_interpolado.py \
    --checkpoint outputs/mi_experimento/checkpoints/checkpoint_final.pt \
    --salida outputs/mi_experimento/frames_60fps \
    --fps_origen 30 --fps_salida 60 --device cuda
```

The script reads the basis the checkpoint declares and generates the new matrix
in that same basis. The interpolation mathematics is nothing more than
evaluating the learned polynomials at new points.

Analysing the result against the true intermediate frames:

```bash
python scripts/metrics/analizar_interpolacion_fps.py ...
```

### What to expect

On a model trained with 90k Gaussians on 180 frames at 720p for 2200 epochs, the
reconstruction reaches 40.97 dB PSNR and 0.985 SSIM, and the same model can be
sampled between frames for a higher frame rate.

Be honest about the limits, though. This is not a specialised video frame
interpolation method and it does not behave like one. A polynomial fitted to
observed frames is smooth between them by construction, which is right for
continuous motion and wrong for fast or discontinuous motion: with rapid
movement the intermediate frames can show trajectories that do not match what
actually happened between the captured frames. The thesis reports this as a
promising direction rather than a solved problem.

## Comparing against the original

To extract the matching ground-truth segment as a video with the same duration
and resolution as an experiment:

```bash
python scripts/data/extraer_video_gt.py --exp outputs/mi_experimento
```

It reads `info_clip.json` to get the frame count and resolution, so the two
videos line up frame for frame.
