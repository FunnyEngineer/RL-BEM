import numpy as np
try:
    import gym
    from gym import spaces
except ImportError:
    raise ImportError("The 'gym' package is required. Install it via 'pip install gym'.")

class BuildingEnergyEnvironment(gym.Env):
    """
    RL environment for building energy modeling with a windowed time series state.
    State: rolling window of recent time series (e.g., temperature, energy, weather, etc.)
    Action: next input parameters (e.g., setpoints)
    Reward: negative prediction error (or custom)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, surrogate_model, action_space_config, 
                 initial_window=None, reward_fn=None, state_window=24, 
                 target_timeseries=None, normalize_state=False, state_mean=None, state_std=None, seed=None):
        """
        Args:
            surrogate_model: Trained surrogate model with a predict(state_window, action) method
            action_space_config: Dict defining action space (e.g., bounds for setpoints). Supports multi-dimensional actions (e.g., heating/cooling setpoints).
            initial_window: np.ndarray, initial window of time series (shape: [window, features])
            reward_fn: Callable (window, action, next_window, target_window) -> float, or None for default
            state_window: Number of time steps in the state window
            target_timeseries: np.ndarray, target/desired time series for reward (shape: [T, features])
            normalize_state: Whether to normalize state
            state_mean, state_std: For normalization (if used)
            seed: Random seed
        """
        # action_space_config can be multi-dimensional, e.g., {'heating_setpoint': (18,22), 'cooling_setpoint': (24,28)}
        super().__init__()
        self.surrogate_model = surrogate_model
        self.state_window = state_window
        self.action_space_config = action_space_config
        self.reward_fn = reward_fn if reward_fn is not None else self._default_reward
        self.normalize_state = normalize_state
        self.state_mean = state_mean
        self.state_std = state_std
        self.rng = np.random.default_rng(seed)
        self.seed(seed)
        self.target_timeseries = target_timeseries
        self.current_step = 0

        # Action space: vector of setpoints for the next time step
        self.action_space = spaces.Box(
            low=np.array([v[0] for v in action_space_config.values()]),
            high=np.array([v[1] for v in action_space_config.values()]),
            dtype=np.float32
        )
        # State: window of recent time series (window, features)
        # Infer feature dimension from initial_window or default to 4
        if initial_window is not None:
            self.window_shape = initial_window.shape
        else:
            self.window_shape = (state_window, 4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.window_shape, dtype=np.float32)

        self.initial_window = initial_window if initial_window is not None else np.zeros(self.window_shape, dtype=np.float32)
        self.current_window = self.initial_window.copy()
        self.episode_reward = 0.0
        self.episode_history = []

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)

    def reset(self):
        self.current_window = self.initial_window.copy()
        self.episode_reward = 0.0
        self.episode_history = []
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        # Predict next time step using surrogate model
        # Input: current window (shape: [window, features]), action (shape: [features,])
        # Output: next_timestep (shape: [features,])
        next_timestep = self.surrogate_model.predict(self.current_window, action)
        # Update window: append new prediction, drop oldest
        next_window = np.roll(self.current_window, -1, axis=0)
        next_window[-1, :] = next_timestep
        # Compute reward
        if self.target_timeseries is not None:
            # Use the corresponding target window for reward
            target_window = self.target_timeseries[self.current_step:self.current_step + self.state_window, :]
        else:
            target_window = None
        reward = self.reward_fn(self.current_window, action, next_window, target_window)
        self.current_window = next_window
        self.episode_reward += reward
        self.episode_history.append((self.current_window.copy(), action.copy(), reward))
        self.current_step += 1
        done = self.current_step >= (self.target_timeseries.shape[0] - self.state_window) if self.target_timeseries is not None else False
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'history': self.episode_history if done else None
        }
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        obs = self.current_window.copy()
        if self.normalize_state and self.state_mean is not None and self.state_std is not None:
            obs = (obs - self.state_mean) / (self.state_std + 1e-8)
        return obs

    def _default_reward(self, window, action, next_window, target_window):
        # Default: negative L2 norm between predicted window and target window (if provided), else between next and current window
        if target_window is not None and target_window.shape == next_window.shape:
            return -np.linalg.norm(next_window - target_window)
        else:
            return -np.linalg.norm(next_window - window)

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, State window: {self.current_window}, Reward: {self.episode_reward}")

    def close(self):
        pass 