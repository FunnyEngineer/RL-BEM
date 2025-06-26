#!/usr/bin/env python3
"""
Run the RL-driven Active Learning Loop for Task 4.3.

This script implements the iterative process of:
1. Training a surrogate model on current data
2. Using RL agent to select new simulation points
3. Generating new data using the surrogate model
4. Retraining the surrogate model with expanded dataset
5. Repeating the process

Usage:
    python run_active_learning.py --iterations 3 --episodes-per-iter 10 --new-samples-per-iter 20
"""

import os
import sys
import argparse

# Add src to path
sys.path.append('src')

from src.rl.active_learning_loop import ActiveLearningLoop

def main():
    parser = argparse.ArgumentParser(description="Run RL-driven Active Learning Loop")
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Directory with processed data')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--initial-model', type=str, default=None, help='Path to initial model checkpoint')
    parser.add_argument('--iterations', type=int, default=3, help='Number of active learning iterations')
    parser.add_argument('--episodes-per-iter', type=int, default=10, help='RL episodes per iteration')
    parser.add_argument('--new-samples-per-iter', type=int, default=20, help='New samples per iteration')
    parser.add_argument('--window-size', type=int, default=168, help='Window size for time series')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RL-Driven Active Learning Loop for Building Energy Modeling")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Models directory: {args.models_dir}")
    print(f"Initial model: {args.initial_model}")
    print(f"Iterations: {args.iterations}")
    print(f"Episodes per iteration: {args.episodes_per_iter}")
    print(f"New samples per iteration: {args.new_samples_per_iter}")
    print(f"Window size: {args.window_size}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # Check if data exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist!")
        print("Please run the data preparation script first.")
        return
    
    required_files = ['X_train.npy', 'y_train.npy', 'X_val.npy', 'y_val.npy', 'X_test.npy', 'y_test.npy']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(args.data_dir, f))]
    
    if missing_files:
        print(f"Error: Missing required data files: {missing_files}")
        print("Please run the data preparation script first.")
        return
    
    # Create and run active learning loop
    try:
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
        
        print("\n" + "=" * 60)
        print("ACTIVE LEARNING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Final training set size: {results['final_train_size']}")
        print(f"Final validation MSE: {results['final_val_mse']:.4f}")
        print(f"Final test MSE: {results['final_test_mse']:.4f}")
        print(f"Validation MSE improvement: {results['improvement']:.4f}")
        print(f"Results saved to: {active_learning.results_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during active learning: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 