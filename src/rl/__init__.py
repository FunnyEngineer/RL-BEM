"""
Reinforcement Learning module for active learning in building energy modeling.

This module implements RL agents for selecting optimal simulation points
and control actions to improve surrogate model performance.

This package will now use src/rl/agents/ for all RL agent classes (tabular and deep RL)
See src/rl/agents/ for: QLearningAgent, DQNAgent, DoubleDQNAgent, DuelingDQNAgent, PrioritizedReplayDQNAgent, RainbowDQNAgent
"""

from .environment import BuildingEnergyEnvironment
from .agent import QLearningAgent

__all__ = [
    'BuildingEnergyEnvironment',
    'QLearningAgent'
]
