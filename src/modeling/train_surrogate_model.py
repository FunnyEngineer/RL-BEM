import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

# 1. Define the PyTorch Dataset
class EnergyDataset(Dataset):
    """Energy Consumption Dataset."""

    def __init__(self, parquet_path, target_column='Total_Annual_Energy_Consumption'):
        """
        Args:
            parquet_path (string): Path to the parquet file with data.
            target_column (string): The name of the target variable column.
        """
        df = pd.read_parquet(parquet_path)
        self.target_column = target_column
        self.features = torch.tensor(df.drop(columns=[target_column]).values, dtype=torch.float32)
        self.labels = torch.tensor(df[target_column].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 2. Define the Lightning DataModule
class EnergyDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = EnergyDataset(os.path.join(self.data_dir, 'train_pool.parquet'))
            self.val_dataset = EnergyDataset(os.path.join(self.data_dir, 'validation.parquet'))
        if stage == 'test' or stage is None:
            self.test_dataset = EnergyDataset(os.path.join(self.data_dir, 'test.parquet'))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

# 3. Define the Surrogate Model using Lightning
class SurrogateModel(L.LightningModule):
    def __init__(self, input_dim, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.layer_1 = torch.nn.Linear(input_dim, 128)
        self.layer_2 = torch.nn.Linear(128, 64)
        self.layer_3 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

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

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# 4. Main function to run the training
def main():
    L.seed_everything(42)

    # --- Configuration ---
    DATA_DIR = 'data/processed'
    MODEL_OUTPUT_DIR = 'models'
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    EPOCHS = 50

    # Determine input dimension from the dataset
    sample_df = pd.read_parquet(os.path.join(DATA_DIR, 'train_pool.parquet'))
    input_dim = len(sample_df.columns) - 1

    # --- Setup ---
    data_module = EnergyDataModule(data_dir=DATA_DIR, batch_size=BATCH_SIZE)
    model = SurrogateModel(input_dim=input_dim, learning_rate=LEARNING_RATE)
    
    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_OUTPUT_DIR,
        filename='surrogate-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=False,
        mode='min'
    )

    # --- Logger ---
    logger = CSVLogger('logs')

    # --- Training ---
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator='auto' # Uses GPU or MPS if available
    )
    
    trainer.fit(model, data_module)

    # --- Testing ---
    trainer.test(model, datamodule=data_module)

    # --- Save Final Model ---
    final_model_path = os.path.join(MODEL_OUTPUT_DIR, 'surrogate_model_final.ckpt')
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == '__main__':
    main() 