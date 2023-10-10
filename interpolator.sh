#!/bin/bash
#SBATCH --account=clidyn.clidyn
#SBATCH --job-name=interpolator
#SBATCH --output=output_interpolator.log
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



srun python3 ~/CNN_eddy_detection/scripts/run.py ${SLURM_ARRAY_TASK_ID} "/albedo/home/ssunar/CNN_eddy_detection/scripts/interpolator.yaml"
# srun python3 ~/CNN_eddy_detection/scripts/run.py ${SLURM_ARRAY_TASK_ID} "/home/ollie/ssunar/CNN_eddy_detection/scripts/interpolator2.yaml"
# srun python3 ~/CNN_eddy_detection/scripts/run.py ${SLURM_ARRAY_TASK_ID} "/home/ollie/ssunar/CNN_eddy_detection/scripts/interpolator3.yaml"

#srun python3 run.py '/work/ollie/nkolduno/output_orca12/ssh.fesom.1963.nc' '/work/ollie/nkolduno/meshes/FORCA12/fesom.mesh.diag.nc' '/work/ollie/nkolduno/meshes/FORCA12/nod2d.out' '/work/ollie/nkolduno/meshes/FORCA12/elem2d.out' 1963 ${SLURM_ARRAY_TASK_ID} -70 30 -60 -20 0
#srun python3 run.py '/work/ollie/nkolduno/output_orca12/ssh.fesom.1961.nc' '/work/ollie/nkolduno/meshes/FORCA12/fesom.mesh.diag.nc' '/home/ollie/vamuelle/pyfesom2/tests/data/pi-grid/nod2d.out'  1961 ${SLURM_ARRAY_TASK_ID} -70 30 -60 -20 0 &
#wait

