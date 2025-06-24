#!/bin/bash
#
#  -- Use TACC's launcher utility to run multiple serial 
#       executables at the same time, execute "module load launcher" 
#       followed by "module help launcher".
#----------------------------------------------------

#SBATCH -J RL-BEM           # Job name
#SBATCH -o RL-BEM.log       # Name of stdout output file
#SBATCH -e RL-BEM.log       # Name of stderr error file
#SBATCH -p gpu-a100-small        # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -A MSS23005
#SBATCH -t 48:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all
#SBATCH --mail-user=funnyengineer@utexas.edu

# Initialize variables
version=""

# Parse arguments
while getopts "v:" opt; do
  case $opt in
    v) version="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
        echo "Usage: sbatch submit.sh -v your_version_string"
        exit 1
    ;;
  esac
done

# Check if the version argument is provided
if [ -z "$version" ]; then
  echo "Error: No version argument provided."
  echo "Usage: sbatch submit.sh -v your_version_string"
  exit 1
fi

# Any other commands must follow all #SBATCH directives...
module load python3/3.9.7
source /work/08388/tudai/ls6/envs/bem/bin/activate

# Set the environment variable before running
export HCP_ENVIRONMENT=true

# Launch serial code...
python3 src/modeling/train_surrogate_model.py --version $version
