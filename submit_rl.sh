#!/bin/bash
#
#  -- Use TACC's launcher utility to run multiple serial 
#       executables at the same time, execute "module load launcher" 
#       followed by "module help launcher".
#----------------------------------------------------

#SBATCH --job-name=rl-bem-all-agents
#SBATCH --output=logs/rl_bem_all_agents_%j.log
#SBATCH --error=logs/rl_bem_all_agents_%j.log
#SBATCH -p gpu-a100-small        # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -A MSS23005
#SBATCH -t 48:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all
#SBATCH --mail-user=funnyengineer@utexas.edu

module load python3/3.9.7
source /work/08388/tudai/ls6/envs/bem/bin/activate
export PYTHONPATH=$(pwd)

# Run all DQN agents
bash run_all_dqn_agents.sh

# Run random agent baseline
bash run_random_agent_baseline.sh
