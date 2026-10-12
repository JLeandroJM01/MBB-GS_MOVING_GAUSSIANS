#!/bin/bash
# Job GENERICO mono. Uso:
#   sbatch --job-name=<exp> jobs/audio/gabor_mono.sbatch <ruta_config.json> <nombre_exp>
# Ej:
#   sbatch --job-name=f1_wave jobs/audio/gabor_mono.sbatch \
#       configs/audio/gabor/gabor_rock_30s_mono_f1_wave.json gabor_rock_30s_mono_f1_wave
#SBATCH --job-name=gabor_mono
#SBATCH --partition=gpu
#SBATCH --exclude=ag001
#SBATCH --gres=shard:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs_slurm/%x_%j.out
#SBATCH --error=logs_slurm/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=TU_CORREO@utec.edu.pe

set -e
CONFIG="$1"
EXP="$2"
if [ -z "$CONFIG" ]; then echo "ERROR: falta ruta del config (arg 1)"; exit 1; fi

cd $HOME/MBB-GS
module load cuda/12.8
module load gnu12/12.4.0
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate mbb-gs

echo "=== COMPILAR extension Gabor (si hace falta) ==="
cd $HOME/MBB-GS/cuda/gabor_audio_cuda
export SYSROOT="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"
unset CPATH; unset C_INCLUDE_PATH; unset CPLUS_INCLUDE_PATH
export CFLAGS="--sysroot=$SYSROOT"
export CXXFLAGS="--sysroot=$SYSROOT"
export CPPFLAGS="--sysroot=$SYSROOT"
export CC=$(which gcc); export CXX=$(which g++); export CUDAHOSTCXX=$(which g++)
export NVCC_PREPEND_FLAGS="-I$SYSROOT/usr/include -I$SYSROOT/usr/include/x86_64-linux-gnu -Xcompiler=--sysroot=$SYSROOT"
python setup.py build_ext --inplace
cd $HOME/MBB-GS

echo "=== START $(date)  config=$CONFIG  exp=$EXP ==="
if [ -z "$EXP" ]; then
  python scripts/audio/train_gabor.py --config "$CONFIG"
else
  python scripts/audio/train_gabor.py --config "$CONFIG" --nombre-experimento "$EXP"
fi
echo "=== END $(date) ==="
