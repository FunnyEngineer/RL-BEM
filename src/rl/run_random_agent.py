import argparse
import numpy as np
import os
import sys
import pickle
import datetime
import matplotlib.pyplot as plt
from environment import BuildingEnergyEnvironment

def get_surrogate_model(use_mock, model_path=None):
    if use_mock:
        class MockSurrogateModel:
            def predict(self, state, action):
                return state[-1] + action + np.random.normal(0, 0.05, size=state[-1].shape)
        return MockSurrogateModel()
    else:
        from src.modeling.models.lstm_model import LSTMSurrogateModel
        if model_path is None:
            raise ValueError("Must provide model_path for real surrogate model.")
        return LSTMSurrogateModel.load_from_checkpoint(model_path)

def main():
    parser = argparse.ArgumentParser(description='Run random agent baseline in BuildingEnergyEnvironment')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=10, help='Max steps per episode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Directory with X_train.npy and y_train.npy')
    parser.add_argument('--window-index', type=int, default=None, help='Index of window to use (random if not set)')
    parser.add_argument('--model-path', type=str, default=None, help='Path to real surrogate model checkpoint')
    parser.add_argument('--use-mock', action='store_true', help='Use mock surrogate model')
    parser.add_argument('--save-dir', type=str, default='reports/random_agent', help='Directory to save results')
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

    env = BuildingEnergyEnvironment(
        surrogate_model,
        action_space_config,
        initial_window=initial_window,
        target_timeseries=target_window,
        state_window=initial_window.shape[0],
        reward_fn=None,
        seed=args.seed
    )

    # Discretize both action dimensions into 5 bins each
    action_bins = [np.linspace(18.0, 22.0, 5), np.linspace(24.0, 28.0, 5)]
    n_actions = len(action_bins[0]) * len(action_bins[1])

    n_episodes = args.episodes
    max_steps = args.max_steps
    episode_rewards = []
    all_trajectories = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        trajectory = []
        for step in range(max_steps):
            # Randomly select a discrete action index
            action_index = np.random.randint(n_actions)
            indices = np.unravel_index(action_index, [len(action_bins[0]), len(action_bins[1])])
            action = np.zeros(state_size)
            action[0] = action_bins[0][indices[0]]
            action[1] = action_bins[1][indices[1]]
            next_state, reward, done, info = env.step(action)
            trajectory.append((state.copy(), action.copy(), reward))
            state = next_state
            total_reward += reward
            if done:
                break
        episode_rewards.append(total_reward)
        all_trajectories.append(trajectory)
        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}")

    os.makedirs(args.save_dir, exist_ok=True)
    rewards_path = os.path.join(args.save_dir, 'random_episode_rewards.npy')
    np.save(rewards_path, np.array(episode_rewards))
    traj_path = os.path.join(args.save_dir, 'random_trajectories.pkl')
    with open(traj_path, 'wb') as f:
        pickle.dump(all_trajectories, f)
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
        plt.savefig(os.path.join(args.save_dir, f'random_traj_episode_{ep+1}.png'), dpi=150)
        plt.close()
    print(f"Saved temperature and action trajectory plots for first 3 episodes in {args.save_dir}/.")

    # Plot learning curve
    plt.figure(figsize=(8, 5))
    plt.plot(episode_rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Random Agent Learning Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'random_learning_curve.png'), dpi=150)
    plt.show()
    print(f"Learning curve plot saved to {os.path.join(args.save_dir, 'random_learning_curve.png')}")
    print(f"All outputs for this run are in: {args.save_dir}")

if __name__ == '__main__':
    main() 