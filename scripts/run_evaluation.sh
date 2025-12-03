#!/bin/bash
#SBATCH --job-name=detr_eval
#SBATCH --output=outputs/logs/eval_%j.out
#SBATCH --error=outputs/logs/eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Create output directories
mkdir -p outputs/logs
mkdir -p outputs/results

# Run evaluation
cd $SLURM_SUBMIT_DIR
python -m src.evaluate

echo "Evaluation complete!"