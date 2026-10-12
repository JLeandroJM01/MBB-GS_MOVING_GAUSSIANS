# Loss Functions

Source: `src/gs2d_video/core/perdidas.py`

The loss is selected with `tipo_loss` and tuned with a set of `lambda_*` keys.
A term whose lambda is zero is not computed at all.

## Available types

| `tipo_loss` | What it is |
| --- | --- |
| `baseline` | L1 plus optional DSSIM |
| `l1_mse` | L1 plus MSE plus optional DSSIM |
| `motion` | L1 weighted by motion in the ground truth, plus optional DSSIM |
| `hard` | L1 weighted by the current error — online hard example mining |
| `edge` | L1 plus a Sobel edge term |
| `temporal` | L1 plus a term comparing consecutive-frame differences |
| `motion_temporal` | motion plus optional MSE and temporal terms |
| `combo` | activates whichever lambdas are greater than zero |

The configuration used for the reported video results is `motion` with
`lambda_motion: 2.0` and `lambda_dssim: 0.25`.

## The motion loss

This is the one that matters. The idea is that a uniform per-pixel error treats
a static background and a moving limb as equally important, even though the
static background is trivially easy and dominates the pixel count. Weighting the
error towards moving regions spends model capacity where the video actually
changes.

A motion mask is built from **the ground truth only**, from two consecutive
frames:

$$M_t = \text{normalise}\left(\left|I_t - I_{t-1}\right|\right)$$

and turned into a per-pixel weight:

$$W_t = 1 + \lambda_{\text{motion}} M_t$$

so static regions get a weight near 1 and dynamic ones get more. The weighted
loss is

$$L_{\text{motion}} = \frac{\sum_p W_t(p)\left|\hat{I}_t(p) - I_t(p)\right|}{\sum_p W_t(p)}$$

and the final loss mixes in DSSIM:

$$L = 0.75\,L_{\text{motion}} + 0.25\,L_{\text{DSSIM}}$$

The mask is detached from the graph. It is a fixed guide computed from the
target, never something the model can influence by changing its output.

### Implementation details worth knowing

The mask is the mean absolute difference across channels, then:

- `motion_umbral` subtracts a floor and clamps at zero, killing sensor noise;
- `motion_blur` applies an average pool of that kernel size, so the weight
  covers the neighbourhood of a moving edge rather than a one-pixel line;
- the mask is normalised by its **mean**, not its maximum, which keeps the
  weight scale stable across frames with different amounts of motion;
- `motion_clip` (default 5.0) caps the normalised mask so a single fast object
  cannot dominate the whole frame.

Frame 0 has no previous frame, so its motion weight is zero by construction.

The previous frame is taken from the real clip whenever the trainer can supply
it, indexing on whatever device the frames live on, so it works with frames held
as CPU `uint8`, GPU `uint8` or GPU `float32`.

## Aggregation knobs

Two orthogonal exponents control how errors are combined. Both apply on top of
whatever `tipo_loss` is selected.

### Across pixels: `exponente_pixel`

| Value | Effect |
| ---: | --- |
| 1 | `mean(abs(R - T))` — the default |
| 2 | mean of squared errors, MSE-like |
| 4 | punishes bad pixels far more than the average |
| ≥ 8 | approaches the maximum, but keeps some gradient elsewhere |

Setting `usar_pnorm_root` takes the $p$-th root afterwards to preserve the
original scale; with Adam the practical effect is small. `usar_max_pixel` uses
the pure maximum, where only the single worst pixel receives gradient — usable
as a probe, not as stable training.

### Across frames: `exponente_frame`

Applies to the per-frame losses within an iteration. $q = 1$ averages them
uniformly; larger $q$ shifts weight onto the hardest frames.

The ablation says to leave it at 1. The numbers are the `fase2_qframe*` records
under `results/video/`:

| q | PSNR | PSNR min | SSIM | LPIPS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 33.24 | 30.02 | 0.964 | 0.048 |
| 2 | 29.77 | 27.43 | 0.942 | 0.099 |
| 4 | 21.19 | 16.84 | 0.859 | 0.336 |
| 8 | 14.70 | 13.37 | 0.517 | 0.821 |

Quality falls monotonically. The likely reason is that high $q$ concentrates
most of the update on a few high-error frames, which unbalances the gradients
reaching the high-order temporal coefficients — exactly the coefficients that
need many frames to be estimated well. `usar_max_frame` is the degenerate case
where only the worst frame in a sub-batch contributes.

## DSSIM

`lambda_dssim` mixes in a structural term:

$$L = (1 - \lambda)\,L_{\text{base}} + \lambda\,L_{\text{DSSIM}}$$

with $L_{\text{DSSIM}} = (1 - \text{SSIM})/2$. It uses `pytorch_msssim` when
available and an equivalent internal implementation with an 11×11 Gaussian
window otherwise.

## Temporal smoothness

Separate from the render loss, a penalty discourages large high-order temporal
coefficients:

```json
"beta_smoothness": 1e-6,
"pesos_smoothness": {
  "mu": 0.0, "opacity": 0.0, "color": 0.0,
  "scale": 0.0, "theta": 1.0, "depth": 2.0
}
```

Coefficient $k$ is weighted by $k^2$, so higher-frequency temporal behaviour is
penalised more. Because $k = 0$ has zero weight, the base value `a_0` is
naturally excluded — which is exactly why the model keeps `a_0` and `a_high` in
separate tensors, as described in
[Model and Temporal Representation](Model-and-Temporal-Representation).

In the reported configuration the penalty is applied only to `theta` and
`depth`: position, opacity, colour and scale are left free to move.

## The fused CUDA loss

`usar_loss_cuda` computes L1 or MSE fused into the rasteriser's backward pass,
which is faster but supports only the simple losses. The trainer rejects the
combination explicitly:

```
usar_loss_cuda=true only supports tipo_loss baseline/l1 with lambda_dssim=0.0
```

Anything involving motion, edges, temporal terms or DSSIM runs the flexible path:
CUDA render, PyTorch loss.
