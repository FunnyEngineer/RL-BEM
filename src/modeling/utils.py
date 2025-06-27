"""Reusable utilities for surrogate-model training & evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Any, Type

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .datasets import create_data_loaders

__all__ = [
    "load_preprocessed_data",
    "evaluate_model",
    "plot_predictions",
    "train_model_arrays",
]


# -----------------------------------------------------------------------------
# Data loading helpers (ported from legacy script)
# -----------------------------------------------------------------------------

def load_preprocessed_data(data_dir: Path):
    """Load numpy arrays saved by the data-preparation pipeline."""
    X_train = np.load(data_dir / "X_train.npy", allow_pickle=True)
    X_val = np.load(data_dir / "X_val.npy", allow_pickle=True)
    X_test = np.load(data_dir / "X_test.npy", allow_pickle=True)
    y_train = np.load(data_dir / "y_train.npy", allow_pickle=True)
    y_val = np.load(data_dir / "y_val.npy", allow_pickle=True)
    y_test = np.load(data_dir / "y_test.npy", allow_pickle=True)
    return X_train, X_val, X_test, y_train, y_val, y_test


# -----------------------------------------------------------------------------
# Evaluation utilities
# -----------------------------------------------------------------------------

def evaluate_model(model, test_loader: DataLoader, device: str):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            all_preds.append(y_hat.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    mae = mean_absolute_error(targets.flatten(), preds.flatten())
    mse = mean_squared_error(targets.flatten(), preds.flatten())
    rmse = np.sqrt(mse)
    ss_res = ((targets.flatten() - preds.flatten()) ** 2).sum()
    ss_tot = ((targets.flatten() - targets.flatten().mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }, preds, targets


# -----------------------------------------------------------------------------
# Plotting helper (same as legacy)
# -----------------------------------------------------------------------------

def plot_predictions(predictions: np.ndarray, targets: np.ndarray, save_path: Path):
    plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])

    # Scatter
    ax1 = plt.subplot(gs[0, 0])
    ax1.scatter(targets, predictions, alpha=0.5)
    min_val, max_val = min(targets.min(), predictions.min()), max(targets.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    ax1.set_xlabel("True Values"); ax1.set_ylabel("Predictions"); ax1.set_title("Predictions vs True Values")

    # Residuals
    ax2 = plt.subplot(gs[0, 1])
    residuals = predictions - targets
    ax2.scatter(predictions, residuals, alpha=0.5)
    ax2.axhline(0, color="r", ls="--")
    ax2.set_xlabel("Predictions"); ax2.set_ylabel("Residuals"); ax2.set_title("Residuals Plot")

    # Summer & winter windows
    gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], hspace=0.3)
    summer, winter = (4400, 4520), (100, 220)
    ax3 = plt.subplot(gs_left[0])
    ax3.plot(targets[summer[0]:summer[1]], label="True (Summer)", color="red", alpha=0.7)
    ax3.plot(predictions[summer[0]:summer[1]], label="Pred (Summer)", color="orange", alpha=0.7)
    ax3.set_title("Summer Comparison"); ax3.legend(fontsize=8)
    ax4 = plt.subplot(gs_left[1])
    ax4.plot(targets[winter[0]:winter[1]], label="True (Winter)", color="blue", alpha=0.7)
    ax4.plot(predictions[winter[0]:winter[1]], label="Pred (Winter)", color="cyan", alpha=0.7)
    ax4.set_title("Winter Comparison"); ax4.set_xlabel("Time Steps"); ax4.legend(fontsize=8)

    # Histograms
    ax5 = plt.subplot(gs[1, 1])
    ax5.hist(targets, bins=40, alpha=0.6, label="True", density=True)
    ax5.hist(predictions, bins=40, alpha=0.6, label="Pred", density=True)
    ax5.set_title("Distribution Comparison"); ax5.set_xlabel("Values"); ax5.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Training helper for in-memory numpy arrays (used by ActiveLearningLoop)
# -----------------------------------------------------------------------------

def train_model_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_dir: str | Path,
    model_name: str,
    model_cls: Type[L.LightningModule],
    max_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    use_wandb: bool = False,
    **model_kwargs: Any,
):
    """Train a model given in-memory numpy arrays.

    Returns the trained model instance.
    """
    save_dir = Path(save_dir)
    run_dir = save_dir / model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # DataLoaders
    train_loader, val_loader, _ = create_data_loaders(
        X_train, y_train, X_val, y_val, batch_size=batch_size
    )

    # Model dimensions
    input_size = X_train.shape[2] if X_train.ndim == 3 else X_train.shape[1]
    if y_train.ndim == 3:
        prediction_horizon = y_train.shape[1]
        output_size = y_train.shape[2]
    else:
        prediction_horizon = 1
        output_size = y_train.shape[1] if y_train.ndim > 1 else 1

    model = model_cls(
        input_size=input_size,
        output_size=output_size,
        prediction_horizon=prediction_horizon,
        learning_rate=learning_rate,
        **model_kwargs,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=run_dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    logger = WandbLogger(project="rl-bem-surrogate", name=model_name, save_dir=str(run_dir.parent)) if use_wandb else None

    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_cb, early_stopping],
        logger=logger,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        val_check_interval=0.25,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_loader, val_loader)

    best_path = checkpoint_cb.best_model_path
    if best_path:
        model = model_cls.load_from_checkpoint(best_path)

    trainer.save_checkpoint(run_dir / "model_final.ckpt")
    if logger:
        import wandb
        wandb.finish()

    return model 