#!/bin/bash
#SBATCH --account=clidyn.clidyn
#SBATCH --job-name=cnn
#SBATCH --output=output_cnn.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=50gb
#SBATCH --time=08:00:00
#SBATCH --qos=12h
#SBATCH -c36
#SBATCH --mail-type=end      
#SBATCH --mail-type=fail
#SBATCH --mail-user=shishir.sunar@awi.de

module load conda
module load mesa/22.0.2
source ~/.bashrc
conda activate eddy-tracking


srun python3 ~/CNN_eddy_detection/scripts/cnn_training.py


