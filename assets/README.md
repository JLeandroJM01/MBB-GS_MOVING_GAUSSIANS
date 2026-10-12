# Figures

Images used by the main README. All commands below run from the repository root.

| File | Where it appears | What it shows |
| --- | --- | --- |
| `top_gaussianas.png` | README header | the original clip, the full 160k-Gaussian reconstruction, and renders keeping only the top 20k and top 80k Gaussians by visual importance |
| `pipeline.png` | Method | the training loop: video → model → render → frame → loss, with backpropagation returning to the model |
| `reconstruccion_n_gaussianas.png` | Results | the same frame reconstructed at 5k, 10k, 20k, 40k, 60k and 150k Gaussians |

Keep files reasonably small: GitHub loads every one of them on each README view.

## Regenerating them

`pipeline.png` is a drawn diagram, not script output.

The other two are rendered from a trained checkpoint:

```bash
# render using only a subset of the Gaussians
python scripts/visualization/viz_render_subset.py --help

# which Gaussians the pruning ranking keeps
python scripts/compression/viz_prune_checkpoint_by_stats.py --help
```

Check each script's `--help` before running: the flags differ between them.

## Other figures worth having

Not currently in the README, but strong material for the paper:

| Figure | Command |
| --- | --- |
| Gaussian trajectories over the frame, as a GIF | `viz_trayectorias_marcadores.py --checkpoint <ckpt> --modo top --n 12 --gif` |
| One Gaussian's attributes against time | `viz_atributos_gaussiana_tiempo.py --checkpoint <ckpt> --salida <dir> --id <n> --inicio 0 --fin 600` |
| Training filmstrip | `viz_tira_evolucion.py --exp outputs/<exp> --gif` |
| Per-Gaussian statistics | `viz_gaussian_stats.py --checkpoint <ckpt> --salida <dir>` |

A temporal-interpolation teaser would be the strongest addition, since that is
the headline claim: render the same model at 4× the source frame rate with
`scripts/reconstruction/regenerar_fps_interpolado.py`, then

```bash
COMP=outputs/<exp>/comparaciones_streaming

ffmpeg -framerate 15 -i $COMP/comparacion_%04d.png \
    -vf "scale=1200:-1:flags=lanczos,palettegen" -y /tmp/pal.png

ffmpeg -framerate 15 -i $COMP/comparacion_%04d.png -i /tmp/pal.png \
    -lavfi "scale=1200:-1:flags=lanczos[x];[x][1:v]paletteuse" \
    -loop 0 -y assets/teaser.gif
```

captioned *"the same model, evaluated between the recorded frames"*.

For audio, `scripts/audio/visualizar.py --exp outputs/gabor/<exp>` produces
waveform, spectrogram and difference-spectrogram comparisons; it detects mono
versus stereo automatically.

## Source material

`reconstruccion_n_gaussianas.png` and `top_gaussianas.png` contain frames of
commercial recordings, shown as figures illustrating the method. The recordings
themselves are not redistributed in this repository; see the main README.
