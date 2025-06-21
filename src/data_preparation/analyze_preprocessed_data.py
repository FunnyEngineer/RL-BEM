"""
Analysis of Preprocessed Time Series Data

This script analyzes the preprocessed ResStock time series data to provide
insights into the dataset characteristics, feature distributions, and target variables.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from pathlib import Path
import logging

# Configure logging and plotting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataAnalyzer:
    """Analyzer for preprocessed time series data."""
    
    def __init__(self, data_dir: str = 'data/processed'):
        self.data_dir = Path(data_dir)
        self.scalers = None
        self.encoders = None
        self.feature_names = None
        self.target_names = None
        
    def load_data(self):
        """Load processed data and artifacts."""
        logging.info("Loading processed data...")
        
        # Load data splits
        self.X_train = np.load(self.data_dir / 'X_train.npy', allow_pickle=True)
        self.y_train = np.load(self.data_dir / 'y_train.npy', allow_pickle=True)
        self.X_val = np.load(self.data_dir / 'X_val.npy', allow_pickle=True)
        self.y_val = np.load(self.data_dir / 'y_val.npy', allow_pickle=True)
        self.X_test = np.load(self.data_dir / 'X_test.npy', allow_pickle=True)
        self.y_test = np.load(self.data_dir / 'y_test.npy', allow_pickle=True)
        
        # Load scalers and encoders
        with open(self.data_dir / 'scalers.pkl', 'rb') as f:
            self.scalers = pickle.load(f)
        with open(self.data_dir / 'encoders.pkl', 'rb') as f:
            self.encoders = pickle.load(f)
            
        logging.info(f"Loaded data shapes:")
        logging.info(f"  X_train: {self.X_train.shape}")
        logging.info(f"  y_train: {self.y_train.shape}")
        logging.info(f"  X_val: {self.X_val.shape}")
        logging.info(f"  y_val: {self.y_val.shape}")
        logging.info(f"  X_test: {self.X_test.shape}")
        logging.info(f"  y_test: {self.y_test.shape}")
        
        # Load feature config to get names
        self._load_feature_names()
        
    def _load_feature_names(self):
        """Load feature names from config."""
        try:
            import json
            with open('config/feature_config.json', 'r') as f:
                config = json.load(f)
            
            # Get feature names (excluding bldg_id, timestamp)
            all_features = []
            all_features.extend(config.get('static_features', []))
            all_features.extend(config.get('dynamic_features', []))
            all_features.extend(['hour_sin', 'hour_cos', 'day_sin', 'day_cos'])
            
            self.feature_names = all_features
            self.target_names = config.get('primary_target', []) + config.get('secondary_targets', [])
            
        except FileNotFoundError:
            logging.warning("Feature config not found, using generic names")
            self.feature_names = [f'feature_{i}' for i in range(self.X_train.shape[2])]
            self.target_names = [f'target_{i}' for i in range(self.y_train.shape[2])]
    
    def generate_dataset_statistics(self):
        """Generate comprehensive dataset statistics."""
        logging.info("Generating dataset statistics...")
        
        stats = {
            'dataset_info': {
                'total_windows': self.X_train.shape[0] + self.X_val.shape[0] + self.X_test.shape[0],
                'train_windows': self.X_train.shape[0],
                'val_windows': self.X_val.shape[0],
                'test_windows': self.X_test.shape[0],
                'input_window_size': self.X_train.shape[1],
                'output_window_size': self.y_train.shape[1],
                'num_features': self.X_train.shape[2],
                'num_targets': self.y_train.shape[2],
                'train_split_ratio': self.X_train.shape[0] / (self.X_train.shape[0] + self.X_val.shape[0] + self.X_test.shape[0]),
                'val_split_ratio': self.X_val.shape[0] / (self.X_train.shape[0] + self.X_val.shape[0] + self.X_test.shape[0]),
                'test_split_ratio': self.X_test.shape[0] / (self.X_train.shape[0] + self.X_val.shape[0] + self.X_test.shape[0])
            }
        }
        
        # Feature statistics
        X_combined = np.concatenate([self.X_train, self.X_val, self.X_test], axis=0)
        feature_stats = {}
        if self.feature_names:
            for i, name in enumerate(self.feature_names):
                feature_data = X_combined[:, :, i].flatten()
                feature_stats[name] = {
                    'mean': float(np.mean(feature_data)),
                    'std': float(np.std(feature_data)),
                    'min': float(np.min(feature_data)),
                    'max': float(np.max(feature_data)),
                    'median': float(np.median(feature_data)),
                    'q25': float(np.percentile(feature_data, 25)),
                    'q75': float(np.percentile(feature_data, 75))
                }
        stats['feature_statistics'] = feature_stats
        
        # Target statistics
        y_combined = np.concatenate([self.y_train, self.y_val, self.y_test], axis=0)
        target_stats = {}
        if self.target_names:
            for i, name in enumerate(self.target_names):
                target_data = y_combined[:, :, i].flatten()
                target_stats[name] = {
                    'mean': float(np.mean(target_data)),
                    'std': float(np.std(target_data)),
                    'min': float(np.min(target_data)),
                    'max': float(np.max(target_data)),
                    'median': float(np.median(target_data)),
                    'q25': float(np.percentile(target_data, 25)),
                    'q75': float(np.percentile(target_data, 75))
                }
        stats['target_statistics'] = target_stats
        
        return stats
    
    def plot_feature_distributions(self, save_path: str = 'reports/feature_distributions.png'):
        """Plot distributions of input features."""
        logging.info("Creating feature distribution plots...")
        
        if not self.feature_names:
            logging.warning("No feature names available, skipping feature distribution plot")
            return
            
        X_combined = np.concatenate([self.X_train, self.X_val, self.X_test], axis=0)
        
        # Create subplots
        n_features = len(self.feature_names)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, (name, ax) in enumerate(zip(self.feature_names, axes)):
            feature_data = X_combined[:, :, i].flatten()
            ax.hist(feature_data, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'{name}\nμ={np.mean(feature_data):.3f}, σ={np.std(feature_data):.3f}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_target_distributions(self, save_path: str = 'reports/target_distributions.png'):
        """Plot distributions of target variables."""
        logging.info("Creating target distribution plots...")
        
        if not self.target_names:
            logging.warning("No target names available, skipping target distribution plot")
            return
            
        y_combined = np.concatenate([self.y_train, self.y_val, self.y_test], axis=0)
        
        n_targets = len(self.target_names)
        fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5))
        if n_targets == 1:
            axes = [axes]
        
        for i, (name, ax) in enumerate(zip(self.target_names, axes)):
            target_data = y_combined[:, :, i].flatten()
            ax.hist(target_data, bins=50, alpha=0.7, edgecolor='black', color='orange')
            ax.set_title(f'{name}\nμ={np.mean(target_data):.3f}, σ={np.std(target_data):.3f}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sample_sequences(self, n_samples: int = 5, save_path: str = 'reports/sample_sequences.png'):
        """Plot sample input and output sequences."""
        logging.info("Creating sample sequence plots...")
        
        fig, axes = plt.subplots(n_samples, 2, figsize=(15, 4 * n_samples))
        
        for i in range(n_samples):
            # Sample from training data
            idx = np.random.randint(0, self.X_train.shape[0])
            
            # Plot input sequence (first few features)
            input_seq = self.X_train[idx]
            for j in range(min(4, input_seq.shape[1])):  # Plot first 4 features
                axes[i, 0].plot(input_seq[:, j], label=f'{self.feature_names[j]}', alpha=0.7)
            axes[i, 0].set_title(f'Sample {i+1}: Input Sequence (168 hours)')
            axes[i, 0].set_xlabel('Time Steps')
            axes[i, 0].set_ylabel('Feature Values')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot output sequence
            output_seq = self.y_train[idx]
            for j in range(output_seq.shape[1]):
                axes[i, 1].plot(output_seq[:, j], label=f'{self.target_names[j]}', alpha=0.7)
            axes[i, 1].set_title(f'Sample {i+1}: Output Sequence (24 hours)')
            axes[i, 1].set_xlabel('Time Steps')
            axes[i, 1].set_ylabel('Target Values')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self, save_path: str = 'reports/correlation_matrix.png'):
        """Plot correlation matrix of features and targets."""
        logging.info("Creating correlation matrix...")
        
        # Use a sample of data to compute correlations
        sample_size = min(10000, self.X_train.shape[0])
        X_sample = self.X_train[:sample_size].reshape(-1, self.X_train.shape[2])
        y_sample = self.y_train[:sample_size].reshape(-1, self.y_train.shape[2])
        
        # Combine features and targets
        combined_data = np.hstack([X_sample, y_sample])
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(combined_data.T)
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature and Target Correlation Matrix')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_statistics_report(self, stats: dict, save_path: str = 'reports/preprocessing_statistics.json'):
        """Save statistics to JSON file."""
        logging.info(f"Saving statistics report to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        import json
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def print_summary_statistics(self, stats: dict):
        """Print summary statistics to console."""
        print("\n" + "="*60)
        print("PREPROCESSED DATA ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nDataset Information:")
        print(f"  Total windows: {stats['dataset_info']['total_windows']:,}")
        print(f"  Train/Val/Test split: {stats['dataset_info']['train_windows']:,} / {stats['dataset_info']['val_windows']:,} / {stats['dataset_info']['test_windows']:,}")
        print(f"  Input window size: {stats['dataset_info']['input_window_size']} hours")
        print(f"  Output window size: {stats['dataset_info']['output_window_size']} hours")
        print(f"  Number of features: {stats['dataset_info']['num_features']}")
        print(f"  Number of targets: {stats['dataset_info']['num_targets']}")
        
        print(f"\nTarget Variables Summary:")
        for name, target_stats in stats['target_statistics'].items():
            print(f"  {name}:")
            print(f"    Mean: {target_stats['mean']:.3f}")
            print(f"    Std:  {target_stats['std']:.3f}")
            print(f"    Range: [{target_stats['min']:.3f}, {target_stats['max']:.3f}]")
        
        print(f"\nFeature Variables Summary:")
        print(f"  Number of features: {len(stats['feature_statistics'])}")
        feature_ranges = []
        for name, feature_stats in stats['feature_statistics'].items():
            feature_ranges.append((name, feature_stats['max'] - feature_stats['min']))
        
        # Show features with largest ranges
        feature_ranges.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top 5 features by range:")
        for name, range_val in feature_ranges[:5]:
            print(f"    {name}: {range_val:.3f}")

def main():
    """Main analysis function."""
    analyzer = DataAnalyzer()
    
    try:
        # Load data
        analyzer.load_data()
        
        print("\n" + "="*60)
        print("PREPROCESSED DATA ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nDataset Information:")
        print(f"  X_train: {analyzer.X_train.shape}")
        print(f"  y_train: {analyzer.y_train.shape}")
        print(f"  X_val: {analyzer.X_val.shape}")
        print(f"  y_val: {analyzer.y_val.shape}")
        print(f"  X_test: {analyzer.X_test.shape}")
        print(f"  y_test: {analyzer.y_test.shape}")
        
        print(f"\nFeature Names: {analyzer.feature_names}")
        print(f"Target Names: {analyzer.target_names}")
        
        # Create visualizations (using samples for speed)
        print("\nGenerating visualizations...")
        analyzer.plot_feature_distributions()
        analyzer.plot_target_distributions()
        analyzer.plot_sample_sequences()
        analyzer.plot_correlation_matrix()
        
        logging.info("Analysis complete! Check the 'reports/' directory for generated plots.")
        
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        raise

if __name__ == '__main__':
    main() 