#!/bin/bash --login

# Job name:
#SBATCH --job-name=elm_cal_og_production

# Number of tasks 
#SBATCH --ntasks=64

# Memory per node
#SBATCH --mem=100G

# Wall time
# Format: "minutes", "hours:minutes:seconds", 
# "days-hours", or "days-hours:minutes"
#SBATCH --time=24:00:00

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
mamba activate elm

## set any env variables we need
export X43I_DATAPATH="/mnt/home/beyerkyl/x4db/unpack_exfor-2024/X4-2024-12-31"

python -c 'import elm; print(f"Running model version: {elm.__version__}")'

# output directory
OUTPUT_DIR=$SLURM_JOB_NAME
mkdir -p $OUTPUT_DIR

# run calibration, with one chain on each available core
mpirun -np $SLURM_NTASKS python3 ./run_cal.py \
  --input ./walker.pkl.xz \
  --steps 100000 \
  --burnin 10000 \
  --output $OUTPUT_DIR

# gather and write the results to a single file
mpirun -np $SLURM_NTASKS python3 ./gather_walkers.py --input $OUTPUT_DIR
