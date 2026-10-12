# Reproducibility

## What a run records

Every training writes, alongside its results:

| File | Contents |
| --- | --- |
| `config_usada.json` | the exact configuration, with CLI overrides applied |
| `info_clip.json` | clip name, frame count, H, W, seed, fps, source video |
| `metricas.json` | aggregates and per-frame values, split pre/post pruning |
| `metricas_por_frame.csv` | one row per frame |
| `logs/log_entrenamiento.csv` | per-epoch losses and timings |

A Gabor audio run writes the same kind of record with `info_audio.json` in place
of `info_clip.json` and no per-frame CSV; a stereo run splits the log per
channel.

`config_usada.json` is the important one. It is written before training starts
and is a verbatim copy of what the run actually used, so a result can never
disagree with the configuration file it claims to come from — even if the
original config in `configs/` is edited afterwards.

## Seeds

`seed` in the configuration drives `torch.manual_seed`, the CUDA seed, and the
model's own `torch.Generator`, which controls initial positions, colours and
depths. The block-wise temporal sampler is seeded from the same value.

Full bit-exact determinism is not guaranteed. CUDA reductions and the
rasteriser's atomic accumulation are not deterministic across runs, and
`--use_fast_math` is enabled in the CUDA build, which can shift floating point
results slightly. Expect runs to agree closely, not exactly.

## Checkpoints

A checkpoint stores coefficients and metadata, not frames:

```python
{
  "state_dict_coefs": {
      "mu_a0": ..., "mu_high": ...,      # and the other five attributes
      "grados": {...}, "N": ..., "H": ..., "W": ..., "n_frames": ...,
  },
  "config": {...},
}
```

`checkpoint_final.pt` is written immediately after training, before any metric
is computed, so an evaluation failure cannot cost the model.
`modelo_pruneado.pt` additionally carries the pre- and post-pruning metric
reports.

Older checkpoints without a `base_temporal` field are interpreted as Chebyshev,
which keeps earlier experiments loadable. See
[Reconstruction and Interpolation](Reconstruction-and-Interpolation) for the
loader.

## Reproducing a reported number

1. Rebuild the clip from your own copy of the source material, using the start
   point and duration recorded in the configuration.
2. Install the environment. `requirements-tesis-khipu.txt` freezes the exact
   package versions used on the cluster.
3. Build the CUDA extension in that environment.
4. Run `scripts/train.py` with the configuration from `configs/`.
5. Compare against the `metricas.json` committed under
   `results/video/<run>/` or `results/audio/<run>/`, where `<run>` is the
   configuration's `nombre_experimento`.

The lightweight per-experiment records are versioned here precisely so that step
5 does not require re-running anything. `results/README.md` lists which runs are
covered and which ones are only on the Drive.

## Cluster runs

`jobs/video/` and `jobs/audio/` hold the Slurm scripts used on Khipu, including
the partition, GPU allocation, memory and time limit of each run. They are the
record of how each experiment was actually launched, not just how it could be,
and they are named after the configuration they run, so a job, its configuration
under `configs/` and its record under `results/` share the same name.

Remember that the environment they activate needs `pip install -e .` once.

## Line endings and encoding

`.gitattributes` normalises source files to LF. All tracked text files are UTF-8
without BOM.
