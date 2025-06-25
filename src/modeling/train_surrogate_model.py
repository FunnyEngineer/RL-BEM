#!/usr/bin/env python3
"""
Train a surrogate sequence model for EnergyPlus time series prediction.

This script implements Phase 3 of the project plan:
- Preprocess dataset for sequence modeling
- Train a sequence model (LSTM/GRU/TCN)
- Evaluate the model on validation data
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import argparse
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class TimeSeriesDataset(Dataset):
    """Dataset for time series sequence modeling."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 168):
        """
        Initialize dataset.
        
        Args:
            X: Input features [n_samples, sequence_length, n_features] or [n_samples, n_features]
            y: Target values [n_samples, prediction_horizon, n_targets] or [n_samples, n_targets]
            sequence_length: Length of input sequences (default: 168 hours = 1 week)
        """
        # Convert to float arrays first
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        # Check if data is already in sequence format
        if len(X.shape) == 3:
            # Data is already in sequence format [n_samples, seq_len, features]
            self.X = torch.FloatTensor(X)
            print(f"Input data already in sequence format: {X.shape}")
        else:
            # Data needs to be converted to sequences
            self.X = torch.FloatTensor(X)
            print(f"Converting input data to sequences: {X.shape}")
        
        if len(y.shape) == 3:
            # Multi-step prediction targets [n_samples, horizon, targets]
            self.y = torch.FloatTensor(y)
            print(f"Target data in multi-step format: {y.shape}")
        else:
            # Single-step targets
            self.y = torch.FloatTensor(y)
            print(f"Target data in single-step format: {y.shape}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMSurrogateModel(pl.LightningModule):
    """LSTM-based surrogate model for time series prediction."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        prediction_horizon: int = 24,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        sequence_length: int = 168,
        lstm_bidirectional: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.prediction_horizon = prediction_horizon
        self.learning_rate = learning_rate
        self.lstm_bidirectional = lstm_bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=lstm_bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers for multi-step prediction
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * (2 if lstm_bidirectional else 1), output_size)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        """Forward pass."""
        # x shape: [batch_size, sequence_length, input_size]
        lstm_out, (hidden, cell) = self.lstm(x)
        # Apply dropout and final linear layer
        output = self.dropout(lstm_out) # [batch_size, sequence_length, hidden_size * 2]
        output = self.fc(output)  # [batch_size, prediction_horizon,  output_size]
        
        return output
    
    def predict(self, state, action):
        """
        Predict next state given current state window and action.
        This method is designed for RL environment integration.
        Args:
            state: np.ndarray, current state window (shape: [window, features] or [features,])
            action: np.ndarray, action vector (shape: [features,])
        Returns:
            np.ndarray: predicted next state (shape: [features,])
        """
        import torch
        # Convert to torch tensors
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action)
        # If state is 1D, treat as a single time step
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        # If state is 2D (window, features), concatenate action to last time step
        # For RL, we typically want to append action to the last time step
        if len(state.shape) == 2:
            # Option 1: Concatenate action to last time step only
            input_seq = state.clone()
            input_seq[-1] = torch.cat([state[-1], action], dim=-1)
            input_seq = input_seq.unsqueeze(0)  # [1, window, features]
        else:
            # Fallback: treat as [1, window, features]
            input_seq = state.unsqueeze(0)
        # Pad or truncate to expected sequence length
        expected_seq_len = self.sequence_length
        current_seq_len = input_seq.shape[1]
        feature_dim = input_seq.shape[2]
        if current_seq_len < expected_seq_len:
            padding = torch.zeros(1, expected_seq_len - current_seq_len, feature_dim)
            input_seq = torch.cat([input_seq, padding], dim=1)
        elif current_seq_len > expected_seq_len:
            input_seq = input_seq[:, -expected_seq_len:, :]
        self.eval()
        with torch.no_grad():
            output = self(input_seq)
            # Take the last output as the next state prediction
            if len(output.shape) == 3:
                predicted_state = output[0, -1, :]
            else:
                predicted_state = output[0, :]
            # If output shape doesn't match state[-1], fallback to state + action + noise
            if predicted_state.shape[0] != state.shape[1]:
                predicted_state = state[-1] + action + torch.randn_like(state[-1]) * 0.01
            return predicted_state.cpu().numpy()
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'y_true': y, 'y_pred': y_hat}
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        return {'test_loss': loss, 'y_true': y, 'y_pred': y_hat}
    
    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

def load_preprocessed_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed data."""
    print("Loading preprocessed data...")
    
    X_train = np.load(data_dir / 'X_train.npy', allow_pickle=True)
    X_val = np.load(data_dir / 'X_val.npy', allow_pickle=True)
    X_test = np.load(data_dir / 'X_test.npy', allow_pickle=True)
    y_train = np.load(data_dir / 'y_train.npy', allow_pickle=True)
    y_val = np.load(data_dir / 'y_val.npy', allow_pickle=True)
    y_test = np.load(data_dir / 'y_test.npy', allow_pickle=True)
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    # Check data types and handle mixed types
    print(f"X_train dtype: {X_train.dtype}")
    print(f"y_train dtype: {y_train.dtype}")
    
    # If data contains objects, we need to handle it differently
    if X_train.dtype == 'object':
        print("Warning: X_train contains object dtype, attempting to extract numeric data...")
        # Try to extract only numeric columns
        sample = X_train[0, 0, :]  # Get first sample, first timestep, all features
        print(f"Sample features: {sample}")
        
        # Find numeric indices
        numeric_indices = []
        for i, val in enumerate(sample):
            try:
                float(val)
                numeric_indices.append(i)
            except (ValueError, TypeError):
                print(f"Skipping non-numeric feature at index {i}: {val}")
        
        print(f"Found {len(numeric_indices)} numeric features out of {len(sample)}")
        
        # Extract only numeric features
        X_train = X_train[:, :, numeric_indices].astype(np.float32)
        X_val = X_val[:, :, numeric_indices].astype(np.float32)
        X_test = X_test[:, :, numeric_indices].astype(np.float32)
    
    # Data scaling check (simple mean/std check)
    train_mean, train_std = X_train.mean(), X_train.std()
    val_mean, val_std = X_val.mean(), X_val.std()
    test_mean, test_std = X_test.mean(), X_test.std()
    print(f"Train mean/std: {train_mean:.4f}/{train_std:.4f}")
    print(f"Val mean/std: {val_mean:.4f}/{val_std:.4f}")
    print(f"Test mean/std: {test_mean:.4f}/{test_std:.4f}")
    if not (np.isclose(train_mean, val_mean, atol=1e-2) and np.isclose(train_std, val_std, atol=1e-2)):
        print("Warning: Validation set scaling differs from training set! Check your preprocessing.")
    if not (np.isclose(train_mean, test_mean, atol=1e-2) and np.isclose(train_std, test_std, atol=1e-2)):
        print("Warning: Test set scaling differs from training set! Check your preprocessing.")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_data_loaders(
    X_train: np.ndarray, 
    X_val: np.ndarray, 
    X_test: np.ndarray,
    y_train: np.ndarray, 
    y_val: np.ndarray, 
    y_test: np.ndarray,
    sequence_length: int = 168,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training."""
    print(f"Creating data loaders with sequence length: {sequence_length}")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train, sequence_length)
    val_dataset = TimeSeriesDataset(X_val, y_val, sequence_length)
    test_dataset = TimeSeriesDataset(X_test, y_test, sequence_length)
    
    print(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader

def evaluate_model(model: LSTMSurrogateModel, test_loader: DataLoader, device: str) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model on test data."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            y_pred = model(x)
            
            all_predictions.append(y_pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Flatten the arrays for metrics
    predictions_flat = predictions.flatten()
    targets_flat = targets.flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(targets_flat, predictions_flat)
    mse = mean_squared_error(targets_flat, predictions_flat)
    rmse = np.sqrt(mse)
    
    # Calculate R²
    ss_res = np.sum((targets_flat - predictions_flat) ** 2)
    ss_tot = np.sum((targets_flat - np.mean(targets_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2)
    }
    
    return metrics, predictions_flat, targets_flat

def plot_predictions(predictions: np.ndarray, targets: np.ndarray, save_path: Path):
    """Plot predictions vs targets in a 2x2 grid as in the provided screenshot."""
    import matplotlib.gridspec as gridspec
    plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])

    # Top left: Predictions vs True Values
    ax1 = plt.subplot(gs[0, 0])
    ax1.scatter(targets, predictions, alpha=0.5)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predictions')
    ax1.set_title('Predictions vs True Values')

    # Top right: Residuals Plot
    ax2 = plt.subplot(gs[0, 1])
    residuals = predictions - targets
    ax2.scatter(predictions, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predictions')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals Plot')

    # Bottom left: Two vertically stacked time series (Summer and Winter)
    gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], hspace=0.3)
    # Summer
    summer_start, summer_end = 4400, 4520
    ax3a = plt.subplot(gs_left[0])
    ax3a.plot(targets[summer_start:summer_end], label='True (Summer)', color='red', alpha=0.7)
    ax3a.plot(predictions[summer_start:summer_end], label='Predicted (Summer)', color='orange', alpha=0.7)
    ax3a.set_title('Summer Comparison')
    ax3a.set_ylabel('Values')
    ax3a.legend(fontsize=8)
    # Winter
    winter_start, winter_end = 100, 220
    ax3b = plt.subplot(gs_left[1])
    ax3b.plot(targets[winter_start:winter_end], label='True (Winter)', color='blue', alpha=0.7)
    ax3b.plot(predictions[winter_start:winter_end], label='Predicted (Winter)', color='cyan', alpha=0.7)
    ax3b.set_title('Winter Comparison')
    ax3b.set_xlabel('Time Steps')
    ax3b.set_ylabel('Values')
    ax3b.legend(fontsize=8)

    # Bottom right: Distribution Comparison (histogram)
    ax4 = plt.subplot(gs[1, 1])
    ax4.hist(targets, bins=40, alpha=0.6, label='True', density=True)
    ax4.hist(predictions, bins=40, alpha=0.6, label='Predicted', density=True)
    ax4.set_xlabel('Values')
    ax4.set_ylabel('Density')
    ax4.set_title('Distribution Comparison')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train surrogate sequence model')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory')
    parser.add_argument('--sequence_length', type=int, default=168, help='Sequence length (hours)')
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum epochs')
    parser.add_argument('--no-wandb', dest='use_wandb', action='store_false', help='Disable Weights & Biases logging')
    parser.add_argument('--version', type=str, required=True, help='Experiment version identifier (e.g., 0.0.0_lstm)')
    parser.set_defaults(use_wandb=True)
    
    args = parser.parse_args()
    
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir / args.version
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data(data_dir)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size
    )
    
    # Get input and output dimensions
    input_size = X_train.shape[2]  # Features dimension
    if len(y_train.shape) == 3:
        prediction_horizon = y_train.shape[1]  # Time steps to predict
        output_size = y_train.shape[2]  # Number of targets
    else:
        prediction_horizon = 1
        output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1
    
    print(f"Model configuration:")
    print(f"  Input size: {input_size}")
    print(f"  Output size: {output_size}")
    print(f"  Prediction horizon: {prediction_horizon}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Sequence length: {args.sequence_length}")
    
    # Initialize model
    model = LSTMSurrogateModel(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=output_size,
        prediction_horizon=prediction_horizon,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        sequence_length=args.sequence_length
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='surrogate-model-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        mode='min',
        verbose=True
    )
    
    callbacks = [checkpoint_callback, early_stopping]
    
    # Set up logger
    logger = None
    if args.use_wandb:
        logger = WandbLogger(
            project="rl-bem-surrogate",
            name=args.version,
            save_dir=str(project_root / "wandb")
        )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator='auto',
        devices='auto',
        log_every_n_steps=50,
        val_check_interval=0.25,
        gradient_clip_val=1.0
    )
    
    # Train model
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best model from: {best_model_path}")
    model = LSTMSurrogateModel.load_from_checkpoint(best_model_path)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    metrics, predictions, targets = evaluate_model(model, test_loader, device)
    
    print("\nTest Results:")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    
    # Save results
    results = {
        'metrics': metrics,
        'model_config': {
            'input_size': input_size,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'output_size': output_size,
            'sequence_length': args.sequence_length,
            'dropout': args.dropout,
            'learning_rate': args.learning_rate
        },
        'training_config': {
            'batch_size': args.batch_size,
            'max_epochs': args.max_epochs,
            'best_epoch': trainer.current_epoch
        }
    }
    
    with open(output_dir / 'training_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Plot results
    plot_predictions(predictions, targets, output_dir / 'prediction_plots.png')
    
    # Save final model
    final_model_path = output_dir / 'surrogate_model_final.ckpt'
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Log to wandb if enabled
    if args.use_wandb:
        wandb.log(metrics)
        wandb.log({"prediction_plot": wandb.Image(str(output_dir / 'prediction_plots.png'))})
        wandb.finish()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
