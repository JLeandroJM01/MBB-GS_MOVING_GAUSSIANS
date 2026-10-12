# scripts/audio

The Gabor atom audio extension. A different model from the video one: atoms are
static on the time axis, with no temporal polynomials.

| Script | What it does |
| --- | --- |
| `train_gabor.py` | mono training on the raw waveform |
| `train_gabor_stereo.py` | stereo training in the L/R or Mid-Side domain |
| `visualizar.py` | waveform and spectrograms, original against reconstruction |
| `test_gradientes_gabor.py` | validates the CUDA kernel backward against autograd; runs on CPU |

See the wiki page **Gabor Audio**.
