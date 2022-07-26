#!/bin/bash
#SBATCH --job-name=interpolator
#SBATCH --output=output_interpolator.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
##SBATCH --mem=128gb
#SBATCH --mem-per-cpu=2000MB
#SBATCH --time=01:00:00
#SBATCH --qos=large
##SBATCH -c36
#SBATCH --array=1-2 # job array index for each month
#SBATCH --mail-type=end      
#SBATCH --mail-type=fail
#SBATCH --mail-user=shishir.sunar@awi.de
#SBATCH -p smp_new

module load anaconda2
source activate eddy-tracking
srun python3 run.py ${SLURM_ARRAY_TASK_ID}
#srun python3 run.py '/work/ollie/nkolduno/output_orca12/ssh.fesom.1963.nc' '/work/ollie/nkolduno/meshes/FORCA12/fesom.mesh.diag.nc' '/work/ollie/nkolduno/meshes/FORCA12/nod2d.out' '/work/ollie/nkolduno/meshes/FORCA12/elem2d.out' 1963 ${SLURM_ARRAY_TASK_ID} -70 30 -60 -20 0
#srun python3 run.py '/work/ollie/nkolduno/output_orca12/ssh.fesom.1961.nc' '/work/ollie/nkolduno/meshes/FORCA12/fesom.mesh.diag.nc' '/home/ollie/vamuelle/pyfesom2/tests/data/pi-grid/nod2d.out'  1961 ${SLURM_ARRAY_TASK_ID} -70 30 -60 -20 0 &
#wait

