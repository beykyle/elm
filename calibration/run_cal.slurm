#!/bin/bash --login

# Job name:
#SBATCH --job-name=elm_cal_og

# Number of tasks 
#SBATCH --ntasks=64

# Memory per node
#SBATCH --mem=100G

# Wall time
# Format: "minutes", "hours:minutes:seconds", 
# "days-hours", or "days-hours:minutes"
#SBATCH --time=12:00:00

# Mail type
#SBATCH --mail-type=ALL

# Mail address
#SBATCH --mail-user=beyerkyl@msu.edu

# Standard output and error to file
# %x: job name, %j: job ID
#SBATCH --output=%x-%j.SLURMout

# set up environment

## modules
module purge
module load Miniforge3/24.3.0-0

## activate the mamba environment for this job
conda activate /mnt/ffs24/home/beyerkyl/local_conda_pkgfiles
eval "$(mamba shell hook --shell bash)"
mamba activate elm

## set any env variables we need
export X43I_DATAPATH="/mnt/home/beyerkyl/x4db/unpack_exfor-2024/X4-2024-12-31"

python -c 'import elm; print(f"Running model version: {elm.__version__}")'

# run calibration, with one chain on each available core
#mpirun -np $SLURM_NTASKS python3 ./run_cal.py \
#  --input ./walker.xz \
#  --steps 10000 \
#  --burnin 1000 \
#  --verbose \
#  --output ./

# gather and write the results to a single file
mpirun -np $SLURM_NTASKS python3 ./gather_walkers.py --input ./
