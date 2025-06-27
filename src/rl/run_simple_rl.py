import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
import pickle
from environment import BuildingEnergyEnvironment
from agents.q_learning_agent import QLearningAgent
import datetime
import seaborn as sns

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
        # Use new refactored model path (train_surrogate_model is deprecated)
        from src.modeling.models.lstm_model import LSTMSurrogateModel
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
    parser.add_argument('--epsilon', type=float, default=0.1, help='Initial epsilon for epsilon-greedy')
    parser.add_argument('--epsilon-decay', type=float, default=1.0, help='Epsilon decay factor per episode')
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

    # Use a 2D action space: control two features (e.g., heating and cooling setpoints)
    state_size = initial_window.shape[1]
    action_dim = 2  # Control first two features
    action_space_config = {'heating_setpoint': (18.0, 22.0), 'cooling_setpoint': (24.0, 28.0)}
    surrogate_model = get_surrogate_model(args.use_mock, args.model_path)

    # Reward shaping
    def comfort_reward(window, action, next_window, target_window):
        # Example: penalize deviation from comfort band (20-26C) in predicted temp (assume feature 0 is temp)
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
    # else use default (L2) in env

    env = BuildingEnergyEnvironment(
        surrogate_model,
        action_space_config,
        initial_window=initial_window,
        target_timeseries=target_window,
        state_window=initial_window.shape[0],
        reward_fn=reward_fn,
        seed=args.seed
    )

    # Discretize both action dimensions into 5 bins each for Q-learning
    action_bins = [np.linspace(18.0, 22.0, 5), np.linspace(24.0, 28.0, 5)]
    agent = QLearningAgent(state_size=state_size, action_bins=action_bins, epsilon=args.epsilon, epsilon_decay=args.epsilon_decay, seed=args.seed)

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
            # Use first two features of the last time step for state/action
            state_for_action = state[-1][0:2]  # shape: (2,)
            action, action_index = agent.select_action(state_for_action)
            # Pad action to match expected input for surrogate model (if needed)
            full_action = np.zeros(state_size)
            full_action[0:2] = action[0:2]
            next_state, reward, done, info = env.step(full_action)
            agent.update(state_for_action, action_index, reward, next_state[-1][0:2], done)
            trajectory.append((state.copy(), action.copy(), reward))
            state = next_state
            total_reward += reward
            if done:
                break
        episode_rewards.append(total_reward)
        all_trajectories.append(trajectory)
        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f} | Epsilon = {agent.epsilon:.3f}")
        agent.decay_epsilon()

    # --- Create unique run directory ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('reports', f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    # --- DEBUG: Surrogate model sensitivity to actions ---
    print(f"\n[DEBUG] Surrogate model sensitivity to actions at initial state (outputs in {run_dir}):")
    test_actions = np.array(np.meshgrid(action_bins[0], action_bins[1])).T.reshape(-1, 2)
    for a in test_actions:
        full_action = np.zeros(state_size)
        full_action[0:2] = a
        pred = surrogate_model.predict(initial_window, full_action)
        print(f"Action: {a}, Predicted temp (feature 0): {pred[0]:.2f}")
    print("[End DEBUG]\n")

    # --- Sensitivity heatmap for multiple random states ---
    N = 3  # Number of states to test
    state_indices = [idx]  # Always include the initial window
    if n_windows > 1:
        state_indices += list(np.random.choice([i for i in range(n_windows) if i != idx], size=min(N-1, n_windows-1), replace=False))
    for i, sidx in enumerate(state_indices):
        test_window = X[sidx]
        temp_grid = np.zeros((len(action_bins[0]), len(action_bins[1])))
        for hi, h in enumerate(action_bins[0]):
            for ci, c in enumerate(action_bins[1]):
                full_action = np.zeros(state_size)
                full_action[0] = h
                full_action[1] = c
                pred = surrogate_model.predict(test_window, full_action)
                temp_grid[hi, ci] = pred[0]  # feature 0
        plt.figure(figsize=(6,5))
        sns.heatmap(
            temp_grid, 
            annot=True, 
            fmt=".2f", 
            xticklabels=[str(x) for x in action_bins[1]], 
            yticklabels=[str(y) for y in action_bins[0]], 
            cmap="coolwarm"
        )
        plt.xlabel('Cooling Setpoint')
        plt.ylabel('Heating Setpoint')
        plt.title(f'Sensitivity Heatmap (State {i+1}, idx={sidx})')
        plt.tight_layout()
        heatmap_path = os.path.join(run_dir, f'sensitivity_heatmap_state_{i+1}.png')
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        print(f"Saved surrogate sensitivity heatmap for state {i+1} (idx={sidx}) to {heatmap_path}")

    # Save rewards and trajectories to file
    rewards_path = os.path.join(run_dir, 'rl_episode_rewards.npy')
    traj_path = os.path.join(run_dir, 'rl_trajectories.pkl')
    np.save(rewards_path, np.array(episode_rewards))
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
        plt.savefig(os.path.join(run_dir, f'rl_traj_episode_{ep+1}.png'), dpi=150)
        plt.close()
    print(f"Saved temperature and action trajectory plots for first 3 episodes in {run_dir}/.")

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
    plt.savefig(os.path.join(run_dir, 'rl_learning_curve.png'), dpi=150)
    plt.show()
    print(f"Learning curve plot saved to {os.path.join(run_dir, 'rl_learning_curve.png')}")
    print(f"Reward type: {args.reward_type}")
    print(f"All outputs for this run are in: {run_dir}") 