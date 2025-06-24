#!/bin/bash

# Usage: ./rsync_wandb_logs.sh
# Example: ./rsync_wandb_logs.sh

REMOTE_USER="tudai"
REMOTE_HOST="ls6"
REMOTE_PATH="/scratch/08388/tudai/RL-BEM/wandb/"
LOCAL_PATH="./wandb/"

rsync -avz --progress ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH} ${LOCAL_PATH} 