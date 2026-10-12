#!/bin/bash
#SBATCH --job-name=gabor_rock_2s_smoke
#SBATCH --partition=gpu
#SBATCH --exclude=ag001
#SBATCH --gres=shard:6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs_slurm/%x_%j.out
#SBATCH --error=logs_slurm/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=TU_CORREO@utec.edu.pe

set -e
cd $HOME/MBB-GS

module load cuda/12.8
module load gnu12/12.4.0
module load miniconda/3.0
eval "$(conda shell.bash hook)"

conda activate mbb-gs

echo "=== INFO GPU ==="
nvidia-smi
echo "=== COMPILAR extension Gabor (primera vez) ==="
cd $HOME/MBB-GS/cuda/gabor_audio_cuda

# Sysroot de conda para que nvcc/g++ encuentren features.h y demas headers de glibc.
export SYSROOT="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"
unset CPATH
unset C_INCLUDE_PATH
unset CPLUS_INCLUDE_PATH
export CFLAGS="--sysroot=$SYSROOT"
export CXXFLAGS="--sysroot=$SYSROOT"
export CPPFLAGS="--sysroot=$SYSROOT"
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$(which g++)
export NVCC_PREPEND_FLAGS="-I$SYSROOT/usr/include -I$SYSROOT/usr/include/x86_64-linux-gnu -Xcompiler=--sysroot=$SYSROOT"

python setup.py build_ext --inplace
cd $HOME/MBB-GS

echo "=== INFO PyTorch ==="
python -c "import torch; print('torch', torch.__version__, 'cuda_ok', torch.cuda.is_available())"
echo "=== START $(date) ==="

python scripts/audio/train_gabor.py \
    --config configs/audio/gabor/gabor_rock_2s_smoke.json \
    --nombre-experimento gabor_rock_2s_smoke

echo "=== END $(date) ==="
