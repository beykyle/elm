#!/bin/bash --login
#SBATCH --nodes 1
#SBATCH --cpus-per-task 12
#SBATCH --time 03:00:00
#SBATCH --mem 64G
#SBATCH --job-name jupyter-notebook
#SBATCH -o /mnt/home/beyerkyl/jupyter_log/jupyter-notebook-%J.log
#SBATCH -e /mnt/home/beyerkyl/jupyter_log/jupyter-notebook-%J.log

# get tunneling info

port=9999
node=$(hostname -s)
user=$(whoami)

## modules
module purge
module load Miniforge3/24.3.0-0

## activate the mamba environment for this job
mamba activate elm

## set any env variables we need
export X43I_DATAPATH="/mnt/home/beyerkyl/x4db/unpack_exfor-2024/X4-2024-12-31"

# run jupyter lab
jupyter-lab --no-browser --port=${port} --ip=${node}
