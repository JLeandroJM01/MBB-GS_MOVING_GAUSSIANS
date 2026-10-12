# Gabor Audio

Source: `src/gs2d_gabor/`, `scripts/audio/`, `cuda/gabor_audio_cuda/`

A complementary extension: the same philosophy — explicit primitives instead of
a sampled array — applied to audio.

## Why not Gaussians, and why not a spectrogram

The first attempt was to reuse the video approach: Gaussians on a spectrogram
whose parameters evolve through Chebyshev polynomials. It does not work well. A
plain Gaussian is a smooth bump with no oscillation, and audio *is* oscillation;
approximating a waveform with non-oscillating bumps needs an unreasonable number
of them. Working on a spectrogram also means dealing with phase separately, and
phase is where the perceptual quality lives.

That first attempt is kept as a baseline: `results/audio/gauss_pura_30s_mono_*`
holds three runs of plain Gaussians fitted to the same 30 s waveform at 48k,
250k and 500k primitives, next to the Gabor runs on the same signal.

A Gabor atom is a Gaussian **modulated by a sinusoid**, so oscillation is built
into the primitive:

$$\hat{x}(t) = \sum_i A_i \exp\left(-\frac{(t - \mu_i)^2}{2\sigma_i^2}\right)\cos\left(2\pi f_i (t - \mu_i) + \phi_i\right)$$

Each atom has amplitude $A_i$, position $\mu_i$ in samples, width $\sigma_i$,
frequency $f_i$ and phase $\phi_i$. Atoms live directly on the time axis of the
raw waveform. There is no STFT, no separate phase, and — unlike the video model
— **no temporal polynomials**: each atom is static, and the signal is the sum.

## Losses

Source: `src/gs2d_gabor/core/perdidas_gabor.py`

Comparing waveforms sample by sample is perceptually poor: two signals that
sound identical can differ a lot per sample because of tiny phase offsets. The
available terms are:

- **`wave`** — L1 on the waveform.
- **`mrstft`** — multi-resolution STFT: compares spectrogram *magnitude* at
  several resolutions, each combining spectral convergence and a log-magnitude
  term. This is the standard in modern neural vocoders.
- **`sisdr`** — scale-invariant SDR, which ignores overall gain.

`si_sdr` combined with gain matching is what the final audio configuration uses.

Evaluation metrics available in the same module: SI-SDR in dB, LSD (log-spectral
distance), Mel-L1, and MR-STFT as a metric.

## Initialisation

Atoms can be initialised by energy, placing them where the signal actually has
content instead of uniformly. On a signal with silence and transients this
matters a great deal — uniformly placed atoms spend capacity on silence.

## Stereo: L/R versus Mid-Side

`scripts/audio/train_gabor_stereo.py` supports two domains, selected with
`dominio`:

- **`LR`** — two independent models, one per channel.
- **`MS`** — train on $M = (L+R)/2$ and $S = (L-R)/2$, reconstruct
  $L = M+S$ and $R = M-S$.

Mid-Side is the better trade. In typical music almost all the energy is in Mid,
while Side is low-energy and carries the stereo image. So Side can be given
substantially fewer atoms (`n_atomos_side`) at little perceptual cost, which
compresses better while preserving stereo width. The final configuration uses
96k atoms for Mid and 64k for Side.

## Running

```bash
python scripts/audio/train_gabor.py --config configs/audio/gabor/gabor_rock_2s_smoke.json
python scripts/audio/train_gabor_stereo.py --config configs/audio/gabor/gabor_rock_30s_stereo_MS_N48k16k.json
```

Figures comparing original and reconstruction:

```bash
python scripts/audio/visualizar.py --exp outputs/gabor/<experimento>
```

It detects automatically whether the run was mono or stereo and produces
waveform plots, spectrograms, and a difference spectrogram.

## CUDA and the fallback

`src/gs2d_gabor/render/cuda_gabor.py` wraps the CUDA extension in an autograd
function, with a pure-PyTorch fallback for CPU validation. The fallback
materialises a dense $[N, T]$ matrix: correct, and completely impractical for
real training — 160k atoms against 400k samples is not a matrix you want to
allocate.

`scripts/audio/test_gradientes_gabor.py` validates that the analytic gradients
in the CUDA kernel match PyTorch autograd on the same dense render. It runs on
CPU and needs no GPU, and the same check is part of the test suite.

## Quantization

`src/gs2d_gabor/core/cuantizacion.py` offers `fp32`, `fp16mix` and `fp16full`.
`mu_t` is kept as an integer even in the mixed scheme, because it holds sample
positions whose magnitude makes float16 lose too much precision — an atom
misplaced by several samples is an audible artefact.

## Reported result

9 s of stereo at 44.1 kHz, 160k atoms (96k Mid, 64k Side), 6000 epochs:

| Metric | Value |
| --- | ---: |
| Stereo SNR | 27.78 dB |
| Stereo SI-SDR | 27.77 dB |
| Stereo PSNR | 44.24 dB |
| Mean LSD | 4.57 dB |
| Mean Mel-L1 | 0.1985 |
| Mean MR-STFT | 1.8532 |
| Mid SNR | 28.50 dB |
| Side SNR | 20.01 dB |

The Side channel scoring lower than Mid is expected: it carries less energy and
receives fewer atoms by design.
