# Experiment records

Lightweight records for 37 runs — 14 video and 23 audio. Each folder holds the
metrics a run reported, the exact configuration it used and its training log, so
the numbers in the paper can be checked without retraining anything.

```
results/video/    14 runs   see video/README.md
results/audio/    23 runs   see audio/README.md
```

The split matches `configs/` and `jobs/`, and a run keeps the same name across
the three: its configuration, the job that launched it, and its record.

Rendered frames, videos, reconstructed audio and checkpoints are too large for
version control. They are published on the Drive linked from the main README, in
folders with these same names.

## What is not here yet

Four reported experiments ran on the cluster but their records are not committed
in this repository. Their configurations are in `configs/` and their full
outputs are on the Drive:

- the temporal basis comparison, Chebyshev vs. monomial
  (`configs/video/exp1_temporal_basis/`);
- the temporal interpolation run, 90k Gaussians over 180 frames;
- the model reduction run, 150k Gaussians over 750 frames;
- the final stereo audio run, 160k atoms at 6000 epochs
  (`configs/audio/gabor/gabor_rock_31_40_stereo_MS_160k_6000ep.json`).

## Reading a record

Video aggregates live under `post_pruning`; in streaming mode the `pre_pruning`
block is left null on purpose, as explained in [Metrics](../wiki/Metrics.md).

```bash
python -c "import json; print(json.load(open('results/video/fase1_motion/metricas.json'))['post_pruning']['agregados'])"
python -c "import json; print(json.load(open('results/audio/gabor_rock_8s_mono_ganador_N48k/metricas.json'))['snr_db'])"
```
