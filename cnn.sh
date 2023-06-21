#!/bin/bash
#SBATCH --job-name=interpolator
#SBATCH --output=output_interpolator.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=50gb
#SBATCH --time=08:00:00
#SBATCH --qos=large
#SBATCH -c36
#SBATCH --mail-type=end      
#SBATCH --mail-type=fail
#SBATCH --mail-user=shishir.sunar@awi.de

module load python3/3.7.10_intel2021

#source activate eddy-tracking


srun python3 ~/CNN_eddy_detection/script/cnn_training.py


