# Audio experiment records

Lightweight records for 23 audio experiments: the metrics each run
reported, the exact configuration it used, and the per-epoch training log. They
are here so the numbers reported in the paper can be checked without retraining
anything.

| File | Contents |
| --- | --- |
| `metricas.json` | SNR, SI-SDR, PSNR, LSD, Mel-L1 and MR-STFT for the run |
| `config_usada.json` | the exact configuration used |
| `info_audio.json` | source audio metadata: sample rate, duration, channels |
| `log_entrenamiento.csv` | per-epoch loss and timings (mono runs) |
| `log_entrenamiento_M.csv`, `log_entrenamiento_S.csv` | the same, per component, for Mid-Side stereo runs |
| `log_entrenamiento_L.csv`, `log_entrenamiento_R.csv` | the same, per channel, for L/R stereo runs |

## What the folders are

| Group | Experiments |
| --- | --- |
| `gabor_rock_8s_*` | number of atoms on an 8 s fragment, including the selected configuration |
| `gabor_rock_30s_mono_f1_*` | training loss: waveform L1, MR-STFT, SI-SDR, and the waveform plus MR-STFT combination |
| `gabor_rock_30s_mono_f2_N*` | number of atoms, from 16k to 64k |
| `gabor_rock_30s_mono_f4*` | initialisation by matching pursuit and its variants |
| `gabor_rock_30s_stereo_LR_*` | stereo trained as two independent channels |
| `gabor_rock_30s_stereo_MS_*` | stereo trained in the Mid-Side domain, with fewer atoms for Side |
| `gauss_pura_30s_mono_N*` | plain Gaussians instead of Gabor atoms, as the baseline that motivates the modulated primitive |

The `gauss_pura_*` runs are the comparison behind the claim that Gabor atoms
suit oscillating signals better than plain Gaussians: they need far more
primitives to reach a comparable reconstruction.

## What is not here

Reconstructed and target waveforms (`.wav`), checkpoints (`.pt`) and the loss
curve images are published separately; see the link in the main README. The
source audio consists of commercial recordings and is not redistributed.
