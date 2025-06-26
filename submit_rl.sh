#!/bin/bash
#
#  -- Use TACC's launcher utility to run multiple serial 
#       executables at the same time, execute "module load launcher" 
#       followed by "module help launcher".
#----------------------------------------------------

#SBATCH --job-name=rl-bem-all-agents
#SBATCH --output=logs/rl_bem_all_agents_%j.out
#SBATCH --error=logs/rl_bem_all_agents_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load python/3.9
source ~/myenv/bin/activate  # or your actual venv path

# Run all DQN agents
bash run_all_dqn_agents.sh

# Run random agent baseline
bash run_random_agent_baseline.sh
