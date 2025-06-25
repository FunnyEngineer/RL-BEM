import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
import pickle
from environment import BuildingEnergyEnvironment
from agent import QLearningAgent

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# --- Surrogate Model Selection ---
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL agent in BuildingEnergyEnvironment")
    parser.add_argument('--use-mock', action='store_true', help='Use mock surrogate model')
    parser.add_argument('--model-path', type=str, default=None, help='Path to real surrogate model checkpoint')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=10, help='Max steps per episode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode (no exploration)')
    parser.add_argument('--agent-save', type=str, default=None, help='Path to save Q-table after training')
    parser.add_argument('--agent-load', type=str, default=None, help='Path to load Q-table before training')
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Directory with X_train.npy and y_train.npy')
    parser.add_argument('--window-index', type=int, default=None, help='Index of window to use (random if not set)')
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

    # Environment setup
    state_size = initial_window.shape[1]
    action_space_config = {f'setpoint{i+1}': (18.0, 26.0) for i in range(state_size)}
    surrogate_model = get_surrogate_model(args.use_mock, args.model_path)
    env = BuildingEnergyEnvironment(
        surrogate_model,
        action_space_config,
        initial_window=initial_window,
        target_timeseries=target_window,
        state_window=initial_window.shape[0],
        seed=args.seed
    )

    # Discretize each action dimension into 3 bins for Q-learning
    action_bins = [np.linspace(v[0], v[1], 3) for v in action_space_config.values()]
    agent = QLearningAgent(state_size=state_size, action_bins=action_bins, seed=args.seed)

    if args.agent_load is not None and os.path.exists(args.agent_load):
        agent.load(args.agent_load)
        print(f"Loaded agent Q-table from {args.agent_load}")

    if args.eval:
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
            action, action_index = agent.select_action(state[-1])  # Use last time step for action
            next_state, reward, done, info = env.step(action)
            agent.update(state[-1], action_index, reward, next_state[-1], done)
            trajectory.append((state.copy(), action.copy(), reward))
            state = next_state
            total_reward += reward
            if done:
                break
        episode_rewards.append(total_reward)
        all_trajectories.append(trajectory)
        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}")
    agent.log_stats()

    # Save rewards and trajectories to file
    os.makedirs('reports', exist_ok=True)
    rewards_path = 'reports/rl_episode_rewards.npy'
    traj_path = 'reports/rl_trajectories.pkl'
    np.save(rewards_path, np.array(episode_rewards))
    with open(traj_path, 'wb') as f:
        pickle.dump(all_trajectories, f)
    print(f"Episode rewards saved to {rewards_path}")
    print(f"Trajectories saved to {traj_path}")

    # Save agent if requested
    if args.agent_save is not None:
        agent.save(args.agent_save)
        print(f"Agent Q-table saved to {args.agent_save}")

    # Plot learning curve
    plt.figure(figsize=(8, 5))
    plt.plot(episode_rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('RL Agent Learning Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/rl_learning_curve.png', dpi=150)
    plt.show()
    print("Learning curve plot saved to reports/rl_learning_curve.png") 