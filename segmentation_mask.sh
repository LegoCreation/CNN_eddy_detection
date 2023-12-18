#!/bin/bash
#SBATCH --account=clidyn.clidyn
#SBATCH --job-name=interpolator
#SBATCH --output=seg_mask.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
##SBATCH --mem=128gb
#SBATCH --mem-per-cpu=3000MB
#SBATCH --time=08:00:00
#SBATCH --qos=12h
##SBATCH -c36
#SBATCH --array=1-12 # job array index for each month
#SBATCH --mail-type=end      
#SBATCH --mail-type=fail
#SBATCH --mail-user=shishir.sunar@awi.de

module load conda
module load mesa/22.0.2
source ~/.bashrc
conda activate eddy-tracking



srun python3 ~/CNN_eddy_detection/scripts/generate_segmentation_mask.py
#wait

