# Slurm jobs

The scripts that launched each experiment on the Khipu HPC cluster at UTEC.
They are kept as a record of how the reported runs were actually executed, not
just how they could be, so the partition, GPU share, memory and wall-clock limit
of every run are visible.

```
jobs/video/    video trainings
jobs/audio/    Gabor audio trainings
```

Each job is named after the configuration it runs, so a run's job, its
configuration under `configs/` and its record under `results/` share the same
name.

## What a job does

`jobs/video/fase1_motion.sbatch`, abridged:

```bash
#SBATCH --partition=gpu
#SBATCH --gres=shard:6
#SBATCH --mem=32G
#SBATCH --time=12:00:00

cd $HOME/MBB-GS
module load cuda/12.8 gnu12/12.4.0 miniconda/3.0
conda activate mbb-gs

python scripts/train.py \
    --config configs/video/exp3_loss/fase1_motion.json \
    --nombre-experimento fase1_motion
```

Every video job requests a `shard:6` GPU share with 32-64 GB of RAM and 10 to
24 hours, except the final 200k-Gaussian run, which asks for `shard:8` and
36 hours. Audio jobs are lighter: 24-32 GB and 2 to 12 hours. All run on the
`gpu` partition except `motion_150k_0800ep`, which used `data-science`.

## Adapting them

These are cluster-specific and will not run elsewhere unchanged. Before reusing
one, adjust:

- `cd $HOME/MBB-GS` — the checkout path on the cluster;
- the `module load` lines — the module names are the real ones on Khipu, and
  `cuda/12.8` must stay compatible with the PyTorch build in the environment;
- `conda activate mbb-gs` — the environment name;
- `--mail-user=TU_CORREO@utec.edu.pe` — a placeholder, replace or delete it;
- `logs_slurm/` must exist in the working directory, or Slurm drops the job
  before it starts.

The environment a job activates needs `pip install -e .` run once inside it, and
the CUDA extensions compiled there. See [Installation](../wiki/Installation.md).

## The two generic audio jobs

`audio/gabor_mono.sbatch` and `audio/gabor_stereo.sbatch` take the configuration
and the experiment name as arguments instead of hard-coding them:

```bash
sbatch --job-name=st_LR jobs/audio/gabor_stereo.sbatch \
    configs/audio/gabor/gabor_rock_30s_stereo_LR_N48k.json \
    gabor_rock_30s_stereo_LR_N48k
```

A stereo run trains two channels, so it takes roughly twice as long as the mono
equivalent.
