#!/bin/bash
#SBATCH --job-name=detr_train
#SBATCH --output=outputs/logs/train_%j.out
#SBATCH --error=outputs/logs/train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Go to project directory
cd /scratch/ma7752/CV_Assignment3

# Create output directories
mkdir -p outputs/logs
mkdir -p outputs/checkpoints

# Load modules (NYU Greene specific)
module purge
module load python/intel/3.8.6
module load cuda/11.6.2

# Activate virtual environment
source venv/bin/activate

# Run training
python -m src.train

echo "Training complete!"
echo "End time: $(date)"