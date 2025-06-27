from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    """PyTorch dataset wrapping 3-D numpy arrays: (N, T, F)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        self.X = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
        self.y = torch.from_numpy(y) if isinstance(y, np.ndarray) else y

    # ------------------------------------------------------------------
    # Required Dataset API
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):  # type: ignore
        return self.X[idx], self.y[idx]


# --------------------------------------------------------------------------
# Data loader helpers
# --------------------------------------------------------------------------

def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Utility generating training / validation / test DataLoaders."""

    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)
    test_ds = TimeSeriesDataset(X_test, y_test) if X_test is not None and y_test is not None else val_ds

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader 