#!/bin/bash
#SBATCH --job-name=fase2_maxframe
#SBATCH --partition=gpu
#SBATCH --exclude=ag001
#SBATCH --gres=shard:6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
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
echo "=== INFO PyTorch ==="
python -c "import torch; print('torch', torch.__version__, 'cuda_ok', torch.cuda.is_available(), 'dev', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
echo "=== START $(date) ==="

python scripts/train.py \
    --config configs/video/exp3_frame_aggregation/fase2_maxframe.json \
    --nombre-experimento fase2_maxframe

echo "=== END $(date) ==="
