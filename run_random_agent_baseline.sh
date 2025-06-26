#!/bin/bash

mkdir -p reports/random_agent
python3 src/rl/run_random_agent.py --episodes 50 --max-steps 10 --save-dir reports/random_agent/ --model-path models/0.0.3_bilstm_lr5e-4/surrogate_model_final.ckpt