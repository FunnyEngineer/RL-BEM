import argparse
import numpy as np
import os
import sys
import pickle
import datetime
import matplotlib.pyplot as plt
from agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, PrioritizedReplayDQNAgent, RainbowDQNAgent
from environment import BuildingEnergyEnvironment

def get_surrogate_model(use_mock, model_path=None):
    if use_mock:
        class MockSurrogateModel:
            def predict(self, state, action):
                return state[-1] + action + np.random.normal(0, 0.05, size=state[-1].shape)
        return MockSurrogateModel()
    else:
        from src.modeling.train_surrogate_model import LSTMSurrogateModel
        if model_path is None:
            raise ValueError("Must provide model_path for real surrogate model.")
        return LSTMSurrogateModel.load_from_checkpoint(model_path)

def get_agent(agent_type, state_dim, action_dim, **kwargs):
    if agent_type == 'dqn':
        return DQNAgent(state_dim, action_dim, **kwargs)
    elif agent_type == 'double':
        return DoubleDQNAgent(state_dim, action_dim, **kwargs)
    elif agent_type == 'dueling':
        return DuelingDQNAgent(state_dim, action_dim, **kwargs)
    elif agent_type == 'prioritized':
        return PrioritizedReplayDQNAgent(state_dim, action_dim, **kwargs)
    elif agent_type == 'rainbow':
        return RainbowDQNAgent(state_dim, action_dim, **kwargs)
    else:
        raise ValueError(f'Unknown agent type: {agent_type}')

def main():
    parser = argparse.ArgumentParser(description='Run DQN or variant agent in BuildingEnergyEnvironment')
    parser.add_argument('--agent-type', type=str, default='dqn', choices=['dqn','double','dueling','prioritized','rainbow'], help='Type of DQN agent to use')
    parser.add_argument('--use-mock', action='store_true', help='Use mock surrogate model')
    parser.add_argument('--model-path', type=str, default=None, help='Path to real surrogate model checkpoint')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=10, help='Max steps per episode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode (no exploration)')
    parser.add_argument('--agent-save', type=str, default=None, help='Path to save agent after training')
    parser.add_argument('--agent-load', type=str, default=None, help='Path to load agent before training')
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Directory with X_train.npy and y_train.npy')
    parser.add_argument('--window-index', type=int, default=None, help='Index of window to use (random if not set)')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial epsilon for epsilon-greedy')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay factor per episode')
    parser.add_argument('--reward-type', type=str, default='l2', choices=['l2', 'comfort'], help='Reward shaping type')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load initial window and target window from processed data
    X = np.load(os.path.join(args.data_dir, 'X_train.npy'))
    y = np.load(os.path.join(args.data_dir, 'y_train.npy'))
    n_windows = X.shape[0]
    if args.window_index is not None:
        idx = args.window_index
    else:
        idx = np.random.randint(n_windows)
    initial_window = X[idx]  # shape: [window, features]
    target_window = y[idx]   # shape: [window, targets]

    state_size = initial_window.shape[1]
    action_dim = 2  # Only first two features are controlled
    action_space_config = {'heating_setpoint': (18.0, 22.0), 'cooling_setpoint': (24.0, 28.0)}
    surrogate_model = get_surrogate_model(args.use_mock, args.model_path)

    # Reward shaping
    def comfort_reward(window, action, next_window, target_window):
        temp = next_window[-1, 0]
        comfort_low, comfort_high = 20.0, 26.0
        penalty = 0.0
        if temp < comfort_low:
            penalty = comfort_low - temp
        elif temp > comfort_high:
            penalty = temp - comfort_high
        return -penalty
    reward_fn = None
    if args.reward_type == 'comfort':
        reward_fn = comfort_reward

    env = BuildingEnergyEnvironment(
        surrogate_model,
        action_space_config,
        initial_window=initial_window,
        target_timeseries=target_window,
        state_window=initial_window.shape[0],
        reward_fn=reward_fn,
        seed=args.seed
    )

    # DQN agents use discrete actions: discretize both action dims into 5 bins each
    action_bins = [np.linspace(18.0, 22.0, 5), np.linspace(24.0, 28.0, 5)]
    n_actions = len(action_bins[0]) * len(action_bins[1])
    agent_kwargs = dict(
        lr=1e-3,
        gamma=0.99,
        epsilon=args.epsilon,
        epsilon_min=0.05,
        epsilon_decay=args.epsilon_decay,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100
    )
    agent = get_agent(args.agent_type, state_dim=state_size, action_dim=n_actions, **agent_kwargs)

    if args.agent_load is not None and os.path.exists(args.agent_load):
        agent.load(args.agent_load)
        print(f"Loaded agent from {args.agent_load}")

    if args.eval:
        if hasattr(agent, 'epsilon'):
            agent.epsilon = 0.0
        if hasattr(agent, 'set_eval_mode'):
            agent.set_eval_mode(True)
        print("Running in evaluation mode (no exploration)")

    n_episodes = args.episodes
    max_steps = args.max_steps
    episode_rewards = []
    all_trajectories = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        trajectory = []
        for step in range(max_steps):
            state_for_action = state[-1][0:2]  # shape: (2,)
            action_index = agent.select_action(state_for_action)
            indices = np.unravel_index(action_index, [len(action_bins[0]), len(action_bins[1])])
            action = np.zeros(state_size)
            action[0] = action_bins[0][indices[0]]
            action[1] = action_bins[1][indices[1]]
            next_state, reward, done, info = env.step(action)
            agent.buffer.push(state_for_action, action_index, reward, next_state[-1][0:2], done)
            agent.update()
            trajectory.append((state.copy(), action.copy(), reward))
            state = next_state
            total_reward += reward
            if done:
                break
        episode_rewards.append(total_reward)
        all_trajectories.append(trajectory)
        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f} | Epsilon = {getattr(agent, 'epsilon', 0):.3f}")

    # --- Create unique run directory ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('reports', f'dqn_run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    # Save rewards and agent to file
    rewards_path = os.path.join(run_dir, 'dqn_episode_rewards.npy')
    np.save(rewards_path, np.array(episode_rewards))
    traj_path = os.path.join(run_dir, 'dqn_trajectories.pkl')
    with open(traj_path, 'wb') as f:
        pickle.dump(all_trajectories, f)
    if args.agent_save is not None:
        agent.save(args.agent_save)
        print(f"Agent saved to {args.agent_save}")
    print(f"Episode rewards saved to {rewards_path}")
    print(f"Trajectories saved to {traj_path}")

    # --- Visualization: Plot temperature and action trajectories for first 3 episodes ---
    for ep in range(min(3, len(all_trajectories))):
        traj = all_trajectories[ep]
        temps = [s[-1][0] for s, a, r in traj]
        actions0 = [a[0] for s, a, r in traj]
        actions1 = [a[1] for s, a, r in traj]
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(temps, marker='o')
        plt.axhspan(20, 26, color='green', alpha=0.1, label='Comfort Band')
        plt.xlabel('Step')
        plt.ylabel('Predicted Temp (feature 0)')
        plt.title(f'Episode {ep+1} Temp Trajectory')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(actions0, marker='o', label='Heating Setpoint')
        plt.plot(actions1, marker='x', label='Cooling Setpoint')
        plt.xlabel('Step')
        plt.ylabel('Action Value')
        plt.title(f'Episode {ep+1} Actions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f'dqn_traj_episode_{ep+1}.png'), dpi=150)
        plt.close()
    print(f"Saved temperature and action trajectory plots for first 3 episodes in {run_dir}/.")

    # Plot learning curve
    plt.figure(figsize=(8, 5))
    plt.plot(episode_rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Agent Learning Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'dqn_learning_curve.png'), dpi=150)
    plt.show()
    print(f"Learning curve plot saved to {os.path.join(run_dir, 'dqn_learning_curve.png')}")
    print(f"All outputs for this run are in: {run_dir}")

if __name__ == '__main__':
    main() 