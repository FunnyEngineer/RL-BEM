import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
import pickle
import datetime
import json
from pathlib import Path
import torch
from .environment import BuildingEnergyEnvironment
from .agent import QLearningAgent

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.modeling.train_surrogate_model import LSTMSurrogateModel, train_surrogate_model

class ActiveLearningLoop:
    """
    Implements RL-driven active learning for building energy modeling.
    
    This class manages the iterative process of:
    1. Training a surrogate model on current data
    2. Using RL agent to select new simulation points
    3. Generating new data using the surrogate model
    4. Retraining the surrogate model with expanded dataset
    5. Repeating the process
    """
    
    def __init__(self, 
                 data_dir='data/processed',
                 models_dir='models',
                 initial_model_path=None,
                 n_iterations=5,
                 n_episodes_per_iteration=20,
                 n_new_samples_per_iteration=50,
                 window_size=168,
                 seed=42):
        
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.initial_model_path = initial_model_path
        self.n_iterations = n_iterations
        self.n_episodes_per_iteration = n_episodes_per_iteration
        self.n_new_samples_per_iteration = n_new_samples_per_iteration
        self.window_size = window_size
        self.seed = seed
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load initial data
        self.X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        self.y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        self.X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
        self.y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
        self.X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        self.y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        print(f"Initial dataset sizes:")
        print(f"  Train: {self.X_train.shape}")
        print(f"  Val: {self.X_val.shape}")
        print(f"  Test: {self.X_test.shape}")
        
        # Initialize tracking
        self.iteration_results = []
        self.model_paths = []
        self.validation_metrics = []
        
        # Create results directory
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = os.path.join('reports', f'active_learning_{timestamp}')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_or_train_initial_model(self):
        """Load existing model or train initial surrogate model."""
        if self.initial_model_path and os.path.exists(self.initial_model_path):
            print(f"Loading existing model from {self.initial_model_path}")
            model = LSTMSurrogateModel.load_from_checkpoint(self.initial_model_path)
        else:
            print("Training initial surrogate model...")
            model = train_surrogate_model(
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
                save_dir=self.models_dir,
                model_name="initial_surrogate"
            )
        return model
    
    def create_rl_environment_and_agent(self, surrogate_model):
        """Create RL environment and agent for the current iteration."""
        # Sample a random window for RL training
        n_windows = self.X_train.shape[0]
        window_idx = np.random.randint(n_windows)
        initial_window = self.X_train[window_idx]
        target_window = self.y_train[window_idx]
        
        # 2D action space configuration
        state_size = initial_window.shape[1]
        action_space_config = {
            'heating_setpoint': (18.0, 22.0), 
            'cooling_setpoint': (24.0, 28.0)
        }
        
        # Create environment
        env = BuildingEnergyEnvironment(
            surrogate_model=surrogate_model,
            action_space_config=action_space_config,
            initial_window=initial_window,
            target_timeseries=target_window,
            state_window=self.window_size,
            reward_fn=None,  # Use default L2 reward
            seed=self.seed
        )
        
        # Create agent with 2D action space
        action_bins = [
            np.linspace(18.0, 22.0, 5),  # Heating setpoint bins
            np.linspace(24.0, 28.0, 5)   # Cooling setpoint bins
        ]
        agent = QLearningAgent(
            state_size=state_size,
            action_bins=action_bins,
            epsilon=0.3,  # Higher initial exploration
            epsilon_decay=0.95,  # Slower decay
            seed=self.seed
        )
        
        return env, agent
    
    def collect_new_samples(self, surrogate_model, agent, env):
        """Use RL agent to collect new samples for training."""
        print(f"Collecting {self.n_new_samples_per_iteration} new samples using RL agent...")
        
        new_X = []
        new_y = []
        
        # Run multiple episodes to collect diverse samples
        n_episodes = max(1, self.n_new_samples_per_iteration // 10)
        samples_per_episode = self.n_new_samples_per_iteration // n_episodes
        
        for episode in range(n_episodes):
            state = env.reset()
            episode_samples = 0
            
            for step in range(50):  # Max 50 steps per episode
                if episode_samples >= samples_per_episode:
                    break
                    
                # Select action using agent
                state_for_action = state[-1][0:2]  # Use first 2 features
                action, action_index = agent.select_action(state_for_action)
                
                # Pad action to full state size
                full_action = np.zeros(state.shape[1])
                full_action[0:2] = action[0:2]
                
                # Take step in environment
                next_state, reward, done, info = env.step(full_action)
                
                # Store the transition as new training data
                new_X.append(state.copy())
                new_y.append(next_state.copy())
                
                episode_samples += 1
                state = next_state
                
                if done:
                    break
        
        # Convert to numpy arrays
        new_X = np.array(new_X)
        new_y = np.array(new_y)
        
        print(f"Collected {len(new_X)} new samples")
        return new_X, new_y
    
    def retrain_surrogate_model(self, surrogate_model, new_X, new_y, iteration):
        """Retrain surrogate model with expanded dataset."""
        print(f"Retraining surrogate model for iteration {iteration}...")
        
        # Combine original and new data
        combined_X = np.concatenate([self.X_train, new_X], axis=0)
        combined_y = np.concatenate([self.y_train, new_y], axis=0)
        
        print(f"Combined dataset size: {combined_X.shape}")
        
        # Retrain model
        updated_model = train_surrogate_model(
            X_train=combined_X,
            y_train=combined_y,
            X_val=self.X_val,
            y_val=self.y_val,
            save_dir=self.models_dir,
            model_name=f"surrogate_iteration_{iteration}",
            pretrained_model=surrogate_model
        )
        
        return updated_model
    
    def evaluate_model(self, model, iteration):
        """Evaluate model performance and save metrics."""
        print(f"Evaluating model for iteration {iteration}...")
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_predictions = model.predict_batch(self.X_val)
            val_mse = np.mean((val_predictions - self.y_val) ** 2)
            val_mae = np.mean(np.abs(val_predictions - self.y_val))
        
        # Evaluate on test set
        test_predictions = model.predict_batch(self.X_test)
        test_mse = np.mean((test_predictions - self.y_test) ** 2)
        test_mae = np.mean(np.abs(test_predictions - self.y_test))
        
        metrics = {
            'iteration': iteration,
            'val_mse': val_mse,
            'val_mae': val_mae,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'train_size': self.X_train.shape[0]
        }
        
        self.validation_metrics.append(metrics)
        
        print(f"Iteration {iteration} - Val MSE: {val_mse:.4f}, Test MSE: {test_mse:.4f}")
        return metrics
    
    def save_iteration_results(self, iteration, new_X, new_y, model_path):
        """Save results for current iteration."""
        iteration_dir = os.path.join(self.results_dir, f'iteration_{iteration}')
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Save new samples
        np.save(os.path.join(iteration_dir, 'new_X.npy'), new_X)
        np.save(os.path.join(iteration_dir, 'new_y.npy'), new_y)
        
        # Save model path
        self.model_paths.append(model_path)
        
        # Save metrics
        with open(os.path.join(self.results_dir, 'metrics.json'), 'w') as f:
            json.dump(self.validation_metrics, f, indent=2)
    
    def plot_learning_progress(self):
        """Plot learning progress across iterations."""
        if len(self.validation_metrics) < 2:
            return
        
        iterations = [m['iteration'] for m in self.validation_metrics]
        val_mse = [m['val_mse'] for m in self.validation_metrics]
        test_mse = [m['test_mse'] for m in self.validation_metrics]
        train_sizes = [m['train_size'] for m in self.validation_metrics]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot MSE over iterations
        ax1.plot(iterations, val_mse, 'o-', label='Validation MSE')
        ax1.plot(iterations, test_mse, 's-', label='Test MSE')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Mean Squared Error')
        ax1.set_title('Model Performance Over Iterations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot training set size
        ax2.plot(iterations, train_sizes, 'o-', color='green')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Training Set Size')
        ax2.set_title('Training Set Growth')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'learning_progress.png'), dpi=150)
        plt.close()
        
        print(f"Learning progress plot saved to {self.results_dir}")
    
    def run(self):
        """Run the complete active learning loop."""
        print("Starting RL-driven Active Learning Loop")
        print(f"Results will be saved to: {self.results_dir}")
        
        # Load or train initial model
        surrogate_model = self.load_or_train_initial_model()
        
        # Evaluate initial model
        initial_metrics = self.evaluate_model(surrogate_model, 0)
        
        # Main active learning loop
        for iteration in range(1, self.n_iterations + 1):
            print(f"\n=== Iteration {iteration}/{self.n_iterations} ===")
            
            # Create RL environment and agent
            env, agent = self.create_rl_environment_and_agent(surrogate_model)
            
            # Train RL agent
            print(f"Training RL agent for {self.n_episodes_per_iteration} episodes...")
            episode_rewards = []
            
            for episode in range(self.n_episodes_per_iteration):
                state = env.reset()
                total_reward = 0
                
                for step in range(20):  # Max 20 steps per episode
                    state_for_action = state[-1][0:2]
                    action, action_index = agent.select_action(state_for_action)
                    
                    full_action = np.zeros(state.shape[1])
                    full_action[0:2] = action[0:2]
                    
                    next_state, reward, done, info = env.step(full_action)
                    agent.update(state_for_action, action_index, reward, next_state[-1][0:2], done)
                    
                    state = next_state
                    total_reward += reward
                    
                    if done:
                        break
                
                episode_rewards.append(total_reward)
                agent.decay_epsilon()
            
            print(f"Average episode reward: {np.mean(episode_rewards):.2f}")
            
            # Collect new samples using trained agent
            new_X, new_y = self.collect_new_samples(surrogate_model, agent, env)
            
            # Retrain surrogate model
            model_path = os.path.join(self.models_dir, f'surrogate_iteration_{iteration}')
            surrogate_model = self.retrain_surrogate_model(surrogate_model, new_X, new_y, iteration)
            
            # Evaluate updated model
            metrics = self.evaluate_model(surrogate_model, iteration)
            
            # Save iteration results
            self.save_iteration_results(iteration, new_X, new_y, model_path)
            
            # Update training data
            self.X_train = np.concatenate([self.X_train, new_X], axis=0)
            self.y_train = np.concatenate([self.y_train, new_y], axis=0)
        
        # Final evaluation and plotting
        print("\n=== Active Learning Complete ===")
        self.plot_learning_progress()
        
        # Save final results
        final_results = {
            'total_iterations': self.n_iterations,
            'final_train_size': self.X_train.shape[0],
            'final_val_mse': self.validation_metrics[-1]['val_mse'],
            'final_test_mse': self.validation_metrics[-1]['test_mse'],
            'improvement': self.validation_metrics[0]['val_mse'] - self.validation_metrics[-1]['val_mse']
        }
        
        with open(os.path.join(self.results_dir, 'final_results.json'), 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"Final results saved to {self.results_dir}")
        print(f"Validation MSE improvement: {final_results['improvement']:.4f}")
        
        return final_results

def main():
    parser = argparse.ArgumentParser(description="Run RL-driven Active Learning Loop")
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Directory with processed data')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--initial-model', type=str, default=None, help='Path to initial model checkpoint')
    parser.add_argument('--iterations', type=int, default=5, help='Number of active learning iterations')
    parser.add_argument('--episodes-per-iter', type=int, default=20, help='RL episodes per iteration')
    parser.add_argument('--new-samples-per-iter', type=int, default=50, help='New samples per iteration')
    parser.add_argument('--window-size', type=int, default=168, help='Window size for time series')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create and run active learning loop
    active_learning = ActiveLearningLoop(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        initial_model_path=args.initial_model,
        n_iterations=args.iterations,
        n_episodes_per_iteration=args.episodes_per_iter,
        n_new_samples_per_iteration=args.new_samples_per_iter,
        window_size=args.window_size,
        seed=args.seed
    )
    
    results = active_learning.run()
    print("Active learning loop completed successfully!")

if __name__ == "__main__":
    main() 