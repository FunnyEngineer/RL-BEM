"""
Time Series Surrogate Model Training

This script trains a sequence model (LSTM/GRU) to predict indoor temperature and energy consumption
from building characteristics and weather data using the preprocessed time series dataset.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import pickle
import json
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeSeriesDataset(Dataset):
    """Time Series Dataset for building energy prediction."""
    
    def __init__(self, X_path: str, y_path: str):
        """
        Args:
            X_path: Path to input features numpy file
            y_path: Path to target variables numpy file
        """
        self.X = np.load(X_path, allow_pickle=True)
        self.y = np.load(y_path, allow_pickle=True)
        
        # Ensure data is float32 and convert to torch tensors
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)
        
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        
        logging.info(f"Loaded dataset: X shape {self.X.shape}, y shape {self.y.shape}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TimeSeriesDataModule(L.LightningDataModule):
    """Lightning DataModule for time series data."""
    
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = TimeSeriesDataset(
                os.path.join(self.data_dir, 'X_train.npy'),
                os.path.join(self.data_dir, 'y_train.npy')
            )
            self.val_dataset = TimeSeriesDataset(
                os.path.join(self.data_dir, 'X_val.npy'),
                os.path.join(self.data_dir, 'y_val.npy')
            )
        if stage == 'test' or stage is None:
            self.test_dataset = TimeSeriesDataset(
                os.path.join(self.data_dir, 'X_test.npy'),
                os.path.join(self.data_dir, 'y_test.npy')
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            persistent_workers=True
        )

class LSTMSurrogateModel(L.LightningModule):
    """LSTM-based surrogate model for time series prediction."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 output_dim: int = 3,
                 dropout: float = 0.2,
                 learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_dim * 2)
        
        # Take the last output for prediction
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        # Project to output dimension
        output = self.output_projection(last_output)  # (batch_size, output_dim)
        
        # Repeat for each time step in the output window
        # This is a simple approach - in practice you might want a more sophisticated decoder
        output = output.unsqueeze(1).repeat(1, 24, 1)  # (batch_size, 24, output_dim)
        
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_losses.append(loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.val_losses.append(loss.item())
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        self.log('test_loss', loss, on_epoch=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

class GRUSurrogateModel(L.LightningModule):
    """GRU-based surrogate model for time series prediction."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 output_dim: int = 3,
                 dropout: float = 0.2,
                 learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        
        # GRU forward pass
        gru_out, _ = self.gru(x)
        # gru_out shape: (batch_size, sequence_length, hidden_dim * 2)
        
        # Take the last output for prediction
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        # Project to output dimension
        output = self.output_projection(last_output)  # (batch_size, output_dim)
        
        # Repeat for each time step in the output window
        output = output.unsqueeze(1).repeat(1, 24, 1)  # (batch_size, 24, output_dim)
        
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss, on_epoch=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

def main():
    """Main training function."""
    L.seed_everything(42)
    
    # --- Configuration ---
    DATA_DIR = 'data/processed'
    MODEL_OUTPUT_DIR = 'models'
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    MODEL_TYPE = 'lstm'  # 'lstm' or 'gru'
    
    # Load data to determine dimensions
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'), allow_pickle=True)
    
    input_dim = X_train.shape[2]  # Number of features
    output_dim = y_train.shape[2]  # Number of targets
    
    logging.info(f"Input dimension: {input_dim}")
    logging.info(f"Output dimension: {output_dim}")
    logging.info(f"Sequence length: {X_train.shape[1]}")
    logging.info(f"Prediction horizon: {y_train.shape[1]}")
    
    # --- Setup ---
    data_module = TimeSeriesDataModule(data_dir=DATA_DIR, batch_size=BATCH_SIZE)
    
    # Choose model type
    if MODEL_TYPE == 'lstm':
        model = LSTMSurrogateModel(
            input_dim=input_dim,
            output_dim=output_dim,
            learning_rate=LEARNING_RATE
        )
    else:
        model = GRUSurrogateModel(
            input_dim=input_dim,
            output_dim=output_dim,
            learning_rate=LEARNING_RATE
        )
    
    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_OUTPUT_DIR,
        filename=f'{MODEL_TYPE}-surrogate-{{epoch:02d}}-{{val_loss:.4f}}',
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )
    
    # --- Logger ---
    logger = CSVLogger('logs', name=f'{MODEL_TYPE}_surrogate')
    
    # --- Training ---
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator='auto',
        devices='auto',
        precision='16-mixed' if torch.cuda.is_available() else '32',
        gradient_clip_val=1.0,
        accumulate_grad_batches=2
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Test the model
    trainer.test(model, datamodule=data_module)
    
    # Save final model
    final_model_path = os.path.join(MODEL_OUTPUT_DIR, f'{MODEL_TYPE}_surrogate_final.ckpt')
    trainer.save_checkpoint(final_model_path)
    logging.info(f"Final model saved to {final_model_path}")
    
    # Save model configuration
    config = {
        'model_type': MODEL_TYPE,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'sequence_length': X_train.shape[1],
        'prediction_horizon': y_train.shape[1],
        'hyperparameters': model.hparams
    }
    
    config_path = os.path.join(MODEL_OUTPUT_DIR, f'{MODEL_TYPE}_surrogate_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logging.info(f"Model configuration saved to {config_path}")

if __name__ == '__main__':
    main() 