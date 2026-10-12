# MBB-GS Wiki

MBB-GS represents multimedia signals with explicit primitives instead of frame
or sample arrays. A video becomes a fixed population of 2D Gaussians whose
attributes are polynomials in time; audio becomes a sum of Gabor atoms.

The [README](https://github.com/JLeandroJM01/MBB-GS_MOVING_GAUSSIANS#readme) covers installation and a first experiment. These
pages cover the rest.

## Model and rendering

- **[Model and Temporal Representation](Model-and-Temporal-Representation)** —
  how a Gaussian's attributes become functions of time, why Chebyshev, and how
  the coefficients are stored and evaluated.
- **[CUDA Rasterizer](CUDA-Rasterizer)** — tiling, depth ordering, alpha
  compositing, forward and backward passes.

## Using the framework

- **[Installation](Installation)** — environment, dependencies, building the
  CUDA extensions on Linux and Windows.
- **[Data Preparation](Data-Preparation)** — going from an MP4 to the PNG
  sequence the trainer expects.
- **[Training](Training)** — the training loop, memory modes, and the full
  configuration reference.
- **[Loss Functions](Loss-Functions)** — every `tipo_loss`, the motion
  weighting, and the aggregation exponents.
- **[Reconstruction and Interpolation](Reconstruction-and-Interpolation)** —
  rendering from a checkpoint and evaluating at intermediate instants.
- **[Metrics](Metrics)** — PSNR, SSIM, LPIPS, temporal PSNR, and how they are
  computed in streaming mode.
- **[Pruning and Quantization](Pruning-and-Quantization)** — reducing the model
  after training.

## Audio and integration

- **[Gabor Audio](Gabor-Audio)** — the atom model, Mid-Side stereo, and the
  audio losses.
- **[Audiovisual Pipeline](Audiovisual-Pipeline)** — running video and audio
  end to end on the same segment.

## Reference

- **[Results](Results)** — the reported experiments with their settings.
- **[Reproducibility](Reproducibility)** — seeds, saved metadata, and what is
  needed to reproduce a number.
- **[Troubleshooting](Troubleshooting)** — the failures that actually happen.

## Conventions

Code, comments and configuration keys are in Spanish, since that is the
language of the thesis this work comes from. The documentation is in English.
Reading the code alongside these pages, the mapping is direct: `perdidas` is
losses, `grados` is polynomial degrees, `escala` is scale, `mu` is position.
