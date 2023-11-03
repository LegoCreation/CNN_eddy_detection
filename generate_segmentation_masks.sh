#!/bin/bash
#SBATCH --job-name=masks_generator    # Specify job name
#SBATCH --partition=shared         # Specify partition name
#SBATCH --mem=30G                  # Specify amount of memory needed
#SBATCH --time=00:30:00
#SBATCH --account=ab0995           # Charge resources on this project account
#SBATCH --output=output_masks_generator.log
#SBATCH --array=1-12               # job array index for each month

#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=massimiliano.fronza@unitn.it

source ~/.bashrc
conda activate eddy-tracking

# Execute serial programs, e.g.
srun python3 ~/CNN_eddy_detection/scripts/run2.py ${SLURM_ARRAY_TASK_ID} "/home/b/b382485/CNN_eddy_detection/scripts/interpolator.yaml"
