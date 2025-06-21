"""
Quick Analysis of Preprocessed Time Series Data

Simple script to quickly analyze the preprocessed data without heavy plotting.
"""

import numpy as np
import pickle
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def quick_analysis():
    """Quick analysis of the preprocessed data."""
    data_dir = Path('data/processed')
    
    print("\n" + "="*60)
    print("QUICK PREPROCESSED DATA ANALYSIS")
    print("="*60)
    
    # Load data shapes
    print("\nDataset Shapes:")
    X_train = np.load(data_dir / 'X_train.npy', allow_pickle=True)
    y_train = np.load(data_dir / 'y_train.npy', allow_pickle=True)
    X_val = np.load(data_dir / 'X_val.npy', allow_pickle=True)
    y_val = np.load(data_dir / 'y_val.npy', allow_pickle=True)
    X_test = np.load(data_dir / 'X_test.npy', allow_pickle=True)
    y_test = np.load(data_dir / 'y_test.npy', allow_pickle=True)
    
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # Load feature config
    try:
        import json
        with open('config/feature_config.json', 'r') as f:
            config = json.load(f)
        
        feature_names = []
        feature_names.extend(config.get('static_features', []))
        feature_names.extend(config.get('dynamic_features', []))
        feature_names.extend(['hour_sin', 'hour_cos', 'day_sin', 'day_cos'])
        
        target_names = config.get('primary_target', []) + config.get('secondary_targets', [])
        
        print(f"\nFeature Names ({len(feature_names)}):")
        for i, name in enumerate(feature_names):
            print(f"  {i+1:2d}. {name}")
        
        print(f"\nTarget Names ({len(target_names)}):")
        for i, name in enumerate(target_names):
            print(f"  {i+1}. {name}")
            
    except FileNotFoundError:
        print("\nFeature config not found, using generic names")
        feature_names = [f'feature_{i}' for i in range(X_train.shape[2])]
        target_names = [f'target_{i}' for i in range(y_train.shape[2])]
    
    # Quick statistics on a sample
    print(f"\nQuick Statistics (using 1000 random samples):")
    sample_size = min(1000, X_train.shape[0])
    X_sample = X_train[:sample_size]
    y_sample = y_train[:sample_size]
    
    print(f"\nInput Features (sample of {sample_size} windows):")
    for i, name in enumerate(feature_names):
        feature_data = X_sample[:, :, i].flatten()
        print(f"  {name}:")
        print(f"    Mean: {np.mean(feature_data):.4f}")
        print(f"    Std:  {np.std(feature_data):.4f}")
        print(f"    Range: [{np.min(feature_data):.4f}, {np.max(feature_data):.4f}]")
    
    print(f"\nTarget Variables (sample of {sample_size} windows):")
    for i, name in enumerate(target_names):
        target_data = y_sample[:, :, i].flatten()
        print(f"  {name}:")
        print(f"    Mean: {np.mean(target_data):.4f}")
        print(f"    Std:  {np.std(target_data):.4f}")
        print(f"    Range: [{np.min(target_data):.4f}, {np.max(target_data):.4f}]")
    
    # Data split summary
    total_windows = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    print(f"\nData Split Summary:")
    print(f"  Total windows: {total_windows:,}")
    print(f"  Train: {X_train.shape[0]:,} ({X_train.shape[0]/total_windows*100:.1f}%)")
    print(f"  Val:   {X_val.shape[0]:,} ({X_val.shape[0]/total_windows*100:.1f}%)")
    print(f"  Test:  {X_test.shape[0]:,} ({X_test.shape[0]/total_windows*100:.1f}%)")
    
    print(f"\nWindow Configuration:")
    print(f"  Input window: {X_train.shape[1]} hours ({X_train.shape[1]/24:.1f} days)")
    print(f"  Output window: {y_train.shape[1]} hours ({y_train.shape[1]/24:.1f} days)")
    print(f"  Step size: {X_train.shape[1] - y_train.shape[1]} hours")
    
    print(f"\nMemory Usage:")
    print(f"  X_train: {X_train.nbytes / 1024**3:.2f} GB")
    print(f"  y_train: {y_train.nbytes / 1024**3:.2f} GB")
    print(f"  Total: {(X_train.nbytes + y_train.nbytes) / 1024**3:.2f} GB")

if __name__ == '__main__':
    quick_analysis() 