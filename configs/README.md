# Experiment configurations

Every experiment is fully described by one JSON file. `scripts/train.py` reads
it and writes a verbatim copy to `config_usada.json` in the output folder, so
each result can be traced back to the exact settings that produced it.

Folders are grouped by modality and then by the experiment they belong to. The
same `video/` and `audio/` split is used by `jobs/` and `results/`, so a run can
be followed across all three under the same name.

The **Records** column says how many of a folder's configurations have their
metrics committed under `results/`. The rest ran on the cluster but their
lightweight records are not in the repository yet; the full outputs for every
run are on the Drive linked from the main README.

## Video

| Folder | Configs | Records | Experiment |
| --- | ---: | ---: | --- |
| `video/exp1_temporal_basis/` | 2 | 0 | Monomial vs. Chebyshev temporal basis, all else held fixed |
| `video/exp2_capacity/` | 8 | 0 | Number of Gaussians and polynomial degrees |
| `video/exp3_loss/` | 7 | 4 | Loss ablation: `baseline`, `l1_mse`, `edge`, `temporal`, `motion`, `combo` |
| `video/exp3_frame_aggregation/` | 6 | 5 | Exponent `q` used to combine per-frame errors, plus the pure `max` variant |
| `video/exp3_pixel_aggregation/` | 3 | 0 | Exponent applied at pixel level (`exponente_pixel`) |
| `video/scaling_gaussians/` | 3 | 0 | 25k / 50k / 75k Gaussians at a fixed 800 epochs |
| `video/scaling_epochs/` | 4 | 4 | 400 / 800 / 1200 / 1600 epochs at a fixed 150k Gaussians |
| `video/final/` | 1 | 0 | Final video configuration: `motion` loss, 200k Gaussians, 1200 epochs |
| `video/misc/` | 1 | 0 | One-off run kept for traceability; not part of any reported result |

The two scaling folders sweep different axes and use different clips, so they
are not a single series: `scaling_gaussians` varies capacity on one clip,
`scaling_epochs` varies training length on another.

`video/final/ganador_motion_200k_1200ep.json` is the configuration behind the
reconstruction, temporal interpolation and model reduction results.

## Audio

| Folder | Configs | Records | Experiment |
| --- | ---: | ---: | --- |
| `audio/gabor/` | 24 | 20 | Gabor atom model and its ablations: loss, initialisation, number of atoms, mono vs. stereo, L/R vs. Mid-Side |
| `audio/audio_only/` | 3 | 0 | Long stereo Mid-Side runs used for the final audio results |

`audio/audio_only/` is split into one subfolder per run length
(`hero_gabor_4000ep/`, `hero_gabor_6000ep_side/`).

`results/audio/` additionally holds three `gauss_pura_30s_mono_*` runs: plain
Gaussians fitted to a waveform, the first attempt that motivated switching to
Gabor atoms. Their configurations are recoverable from the `config_usada.json`
in each record but are not committed here.

## Audiovisual

| Folder | Configs | Records | Experiment |
| --- | ---: | ---: | --- |
| `audiovisual/thriller_10s_1ep/` | 3 | 0 | Full pipeline on a 10 s segment: `video.json`, `audio.json`, `pipeline.json` |
| `audiovisual/rockyourbody_10s_1ep/` | 3 | 0 | Same pipeline on a second segment |

Each audiovisual experiment needs its three files: `pipeline.json` is the
master configuration and points to the video and audio ones.

## Naming

`nombre_experimento` inside a configuration is what names the output folder, and
it matches the file name for every video and Gabor configuration. The
exceptions are deliberate: the audiovisual `video.json` / `audio.json` pairs and
the `audio_only/hero_*` files carry a longer name that records the source
timestamp.

## Running an experiment

```bash
python scripts/train.py --config configs/video/final/ganador_motion_200k_1200ep.json
```

`scripts/pipeline/run_tests_secuencial.py` runs the loss ablation phases in
order. `jobs/video/*.sbatch` and `jobs/audio/*.sbatch` are the exact Slurm
scripts used on the Khipu HPC cluster, one per reported run.

## Input data

Configs reference clips by name under `data/clips/<clip>/` and audio by path
under `data/audio/`. The source video and audio files are not distributed with
this repository; see the main README for how to obtain them and regenerate the
clips.
