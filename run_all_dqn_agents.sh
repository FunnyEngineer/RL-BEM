#!/bin/bash

AGENTS=(dqn double dueling prioritized rainbow)
for AGENT in "${AGENTS[@]}"; do
  echo "Running agent: $AGENT"
  python3 src/rl/run_dqn_rl.py --agent-type $AGENT --episodes 50 --max-steps 10 --agent-save reports/${AGENT}_agent.pt
  echo "Finished agent: $AGENT"
done 