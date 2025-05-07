#!/bin/bash

#SBATCH --job-name=ner-training         # Job name
#SBATCH --output=baseline.%j.out   # Name of output file (%j is job ID)
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --gres=gpu 					    # Request a GPU
#SBATCH --time=04:00:00                 # Run time (hh:mm:ss)
#SBATCH --partition=scavenge			# Specifies the partition (queue) to submit the job to
#SBATCH --mail-type=BEGIN,END,FAIL		# E-mail when status changes

# Prints the name of the node (computer) where the job is running.
echo "Running on $(hostname):" 

# Command that shows GPU usage and available GPUs
nvidia-smi

# Move to folder
cd ~/hpc_jobs/

# Run Scripts
python baseline.py
python train_aug.py
