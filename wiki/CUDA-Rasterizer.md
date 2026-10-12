# CUDA Rasterizer

Source: `cuda/raster_cuda/`, `src/gs2d_video/render/cuda_tiled.py`,
`src/gs2d_video/render/renderer.py`

The rasteriser turns a set of evaluated Gaussians into a frame, and propagates
gradients back to them. It is a differentiable CUDA extension written for this
project; it takes the general approach of 3D Gaussian Splatting as a conceptual
reference, adapted to work directly with 2D Gaussians — there are no cameras and
no 3D projection, since a Gaussian's position is already a position in the
image.

## Why it has to be CUDA

Training renders hundreds of frames per epoch for thousands of epochs, with
$10^5$ Gaussians each time. Evaluating every Gaussian against every pixel is
hopeless, and so is doing it in Python.

## Tiling

The image is divided into square tiles of `cuda_tile_size` pixels, typically 16.
A Gaussian does not affect the whole image, only a neighbourhood of its centre;
`cuda_k_sigma` (default 3.5) sets how many standard deviations out it reaches.

A preprocessing pass computes which tiles each Gaussian touches, given its
position, scale, rotation and $k_\sigma$, and builds two structures:

- a list of Gaussian identifiers to process, and
- the range of positions in that list belonging to each tile.

During rendering each tile reads only the Gaussians that can actually affect its
pixels. `RASTER_BATCH_SIZE` (default 256, compile-time) controls how many of
them are staged into shared memory at a time; it sizes the kernel's `__shared__`
arrays, so changing it means recompiling.

## Depth and compositing

Within a tile, Gaussians are sorted by depth. That gives a defined order, and
colours are combined by alpha blending, the same compositing used in Gaussian
Splatting. `depth` therefore does not represent a distance to a camera — it is
purely an ordering key, which is why a low polynomial degree is enough for it.

## Forward and backward

The forward pass evaluates the Gaussians and produces the frame, keeping the
transmittance and contribution counts the backward pass needs. The backward pass
computes gradients with respect to position, the conic form, opacity and colour,
and then converts the conic gradients back into gradients on scale and rotation.
From there PyTorch's autograd carries them into the temporal coefficients.

Parameters flow as float32 tensors; Gaussian identifiers and per-tile ranges are
integers. Keeping those separate is what lets the continuous values stay
differentiable while the tile bookkeeping stays cheap.

## The Python side

`gs2d_video/render/renderer.py` is the interface the rest of the code uses:

```python
render_frame(params_j, H, W, config)      # one frame
render_clip(modelo, matrices_base, ...)   # every frame, stacked
loss_frame_cuda(params_j, target_j, ...)  # fused L1/MSE loss
```

`render_clip` stacks all renders in GPU memory, so it is only appropriate for
small clips; for anything large use
[reconstruction in streaming mode](Reconstruction-and-Interpolation).

`cuda_tiled.py` locates the compiled extension, either at `cuda/raster_cuda`
relative to the repository root or wherever `RUTA_RASTER_CUDA` points, and adds
it to `sys.path`. It queries the extension for optional entry points
(`build_conic`, `preprocess_tiled`, `grad_conic_to_scale_theta`) and falls back
to PyTorch implementations when a build does not provide them, which keeps older
compiled extensions usable.

## The fused loss path

`loss_un_frame_cuda_tiled` computes L1 or MSE against the target inside the same
kernel, avoiding a full-resolution intermediate. It is faster but only supports
the simple losses; see [Loss Functions](Loss-Functions).

## Configuration

| Key | Default | Meaning |
| --- | --- | --- |
| `usar_cuda_tiled` | true | required for training |
| `cuda_tile_size` | 16 | tile side in pixels |
| `cuda_k_sigma` | 3.5 | reach of a Gaussian, in standard deviations |
| `usar_loss_cuda` | false | fused loss in the kernel |
| `loss_cuda_tipo` | `l1` | `l1` or `mse` |

Raising `cuda_k_sigma` makes each Gaussian influence a wider area, so more of
them land in each tile and rendering gets slower; lowering it can produce
visible cutoffs at Gaussian borders.

## Note on the old tests

`cuda/raster_cuda/tests/` holds historical benchmarks that import module names
from earlier versions of the code. They are not part of the test suite —
`pyproject.toml` restricts pytest to `tests/` — and they are not maintained.
