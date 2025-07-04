#!/bin/bash --login
#SBATCH --nodes 1
#SBATCH --cpus-per-task 12
#SBATCH --time 03:00:00
#SBATCH --mem 64G
#SBATCH --job-name jupyter-notebook
#SBATCH --output /mnt/home/beyerkyl/jupyter_log/jupyter-notebook-%J.log

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

# Print out the tunneling info
echo "To access the Jupyter Lab, run the following command on your local machine:"
echo "ssh -N -L ${port}:${node}:${port} ${user}@$(hostname -s)"

# Print out the URL to access Jupyter Lab
url = "http://localhost:9999/lab?token=$(jupyter-lab list | grep -oP '(?<=token=)[^ ]+')"
echo "You can access Jupyter Lab at: $url"
