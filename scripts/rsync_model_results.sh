#!/bin/bash

# Usage: ./rsync_model_results.sh <version>
# Example: ./rsync_model_results.sh 0.0.0_lstm_initial

if [ $# -ne 1 ]; then
  echo "Usage: $0 <version>"
  exit 1
fi

VERSION=$1
REMOTE_USER="tudai"
REMOTE_HOST="ls6"
REMOTE_PATH="/scratch/08388/tudai/RL-BEM/models/$VERSION/"
LOCAL_PATH="./models/$VERSION/"

rsync -avz --progress --exclude='*.ckpt' ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH} ${LOCAL_PATH} 