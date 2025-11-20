#!/bin/bash
#SBATCH --job-name=disthpc
#SBATCH --output=disthpc.out
#SBATCH --error=disthpc.err
#SBATCH --partition=gpu          # Change this to your HPC's GPU partition name
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --time=00:10:00          # Set a time limit (5 minutes)
#SBATCH --mem=20G                 # Request 4GB of memory
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# launch with: sbatch ./disthpc.slurm
# check with: squeue -u csb66

# Load necessary modules (if required on your HPC)
module use /opt/nvidia/hpc_sdk/modulefiles
module load nvhpc-hpcx-cuda12/24.5

# Full path to the directory containing your Python files
python_files_dir="/home/csc29/projects/Zero-Shot_SkinLesion_Segmentation/CAM-SAM"

# Init conda
source /etc/profile.d/conda.sh

# Conda environment name
conda_env="CAM-SAM"

# Activate the Conda environment
conda activate $conda_env

# Run the Python script
# python ${python_files_dir}/testhpc.py
python ${python_files_dir}/test.py


echo "Process ID: $SLURM_PROCID running on $(hostname)"

# Deactivate the Conda environment
conda deactivate

wait # Wait for all experiment types to finish
echo "All experiments completed"