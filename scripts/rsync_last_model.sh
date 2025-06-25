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
REMOTE_PATH="/scratch/08388/tudai/RL-BEM/models/$VERSION/last.ckpt"
LOCAL_PATH="./models/$VERSION/last.ckpt"

# rsync options explained:
# -a: archive mode (preserves permissions, timestamps, etc.)
# -v: verbose output (shows what files are being transferred)
# -z: compress data during transfer (faster over slow connections)
# Create the version directory if it doesn't exist
mkdir -p "./models/$VERSION"

rsync -avz --progress ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH} ${LOCAL_PATH} 