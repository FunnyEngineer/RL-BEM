import numpy as np
import pickle

class QLearningAgent:
    """
    Research-grade Q-learning agent for discrete action spaces in building energy RL environment.
    Supports Q-table save/load, logging, reproducibility, and evaluation mode.
    """
    def __init__(self, state_size, action_bins, learning_rate=0.1, discount_factor=0.99, epsilon=0.1, seed=None, normalize_state=False, state_mean=None, state_std=None):
        """
        Args:
            state_size: int, size of the flattened state vector
            action_bins: list of arrays, each array contains discrete values for one action dimension
            learning_rate: float, Q-learning learning rate
            discount_factor: float, Q-learning discount factor
            epsilon: float, epsilon-greedy exploration rate
            seed: random seed
            normalize_state: whether to normalize state
            state_mean, state_std: for normalization
        """
        self.state_size = state_size
        self.action_bins = action_bins
        self.n_actions = np.prod([len(b) for b in action_bins])
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.normalize_state = normalize_state
        self.state_mean = state_mean
        self.state_std = state_std
        # Q-table: dict mapping (state_tuple, action_index) -> Q-value
        self.Q = {}
        self.exploration_count = 0
        self.exploitation_count = 0
        self.eval_mode = False

    def discretize_action(self, action):
        """Convert continuous action to discrete index."""
        indices = [np.argmin(np.abs(b - a)) for b, a in zip(self.action_bins, action)]
        return np.ravel_multi_index(indices, [len(b) for b in self.action_bins])

    def _normalize(self, state):
        if self.state_mean is None or self.state_std is None:
            return state
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def select_action(self, state):
        """Epsilon-greedy action selection. If eval_mode, always exploit."""
        if self.normalize_state:
            state = self._normalize(state)
        state_tuple = tuple(np.round(state, 3))
        if (not self.eval_mode) and (self.rng.random() < self.epsilon):
            # Random action
            action_index = self.rng.integers(self.n_actions)
            self.exploration_count += 1
        else:
            # Greedy action
            q_values = [self.Q.get((state_tuple, a), 0.0) for a in range(self.n_actions)]
            action_index = int(np.argmax(q_values))
            self.exploitation_count += 1
        # Convert index to action vector
        indices = np.unravel_index(action_index, [len(b) for b in self.action_bins])
        action = np.array([self.action_bins[i][idx] for i, idx in enumerate(indices)])
        return action, action_index

    def update(self, state, action_index, reward, next_state, done):
        """Q-learning update."""
        if self.normalize_state:
            state = self._normalize(state)
            next_state = self._normalize(next_state)
        state_tuple = tuple(np.round(state, 3))
        next_state_tuple = tuple(np.round(next_state, 3))
        q_sa = self.Q.get((state_tuple, action_index), 0.0)
        if done:
            target = reward
        else:
            next_qs = [self.Q.get((next_state_tuple, a), 0.0) for a in range(self.n_actions)]
            target = reward + self.discount_factor * max(next_qs)
        self.Q[(state_tuple, action_index)] = q_sa + self.learning_rate * (target - q_sa)

    def reset(self):
        """Reset the Q-table and counters."""
        self.Q = {}
        self.exploration_count = 0
        self.exploitation_count = 0

    def save(self, path):
        """Save Q-table to file."""
        with open(path, 'wb') as f:
            pickle.dump(self.Q, f)

    def load(self, path):
        """Load Q-table from file."""
        with open(path, 'rb') as f:
            self.Q = pickle.load(f)

    def set_eval_mode(self, eval_mode=True):
        """Set evaluation mode (no exploration)."""
        self.eval_mode = eval_mode

    def log_stats(self):
        total = self.exploration_count + self.exploitation_count
        if total == 0:
            print("No actions taken yet.")
        else:
            print(f"Exploration: {self.exploration_count} ({100*self.exploration_count/total:.1f}%) | Exploitation: {self.exploitation_count} ({100*self.exploitation_count/total:.1f}%)") 