"""Generic training entry-point for surrogate sequence models.

Example usages:

    # Train an LSTM (default)
    python -m src.modeling.train_sequence_model --version exp01

    # Train a GRU (once implemented)
    python -m src.modeling.train_sequence_model --model gru --version exp02

The script is deliberately lightweight – all model-specific logic lives in
``src.modeling.models`` while data loading utilities can be found in
``src.modeling.datasets``.
"""
# coding: utf-8
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# -----------------------------------------------------------------------------
# Ensure project root is on PYTHONPATH so that `src.*` imports resolve correctly
# -----------------------------------------------------------------------------

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# -----------------------------------------------------------------------------
# Local imports – kept after path manipulation to avoid circular dependencies
# -----------------------------------------------------------------------------
from src.modeling.datasets import create_data_loaders
from src.modeling.models import get_model_class
from src.modeling.utils import load_preprocessed_data, evaluate_model, plot_predictions


# -----------------------------------------------------------------------------
# Argument parsing helper
# -----------------------------------------------------------------------------

def parse_model_kwargs(unknown: list[str]) -> Dict[str, Any]:
    """Parse additional CLI arguments into a dictionary.

    Any argument of the form ``--key value`` will be stored as ``key: value`` in
    the resulting dictionary. Values are automatically converted to ``int``,
    ``float`` or left as ``str``.
    """
    key = None
    kwargs: Dict[str, Any] = {}
    for token in unknown:
        if token.startswith("--"):
            key = token.lstrip("-")
        elif key is not None:
            # Attempt dtype conversion
            if token.isdigit():
                value: Any = int(token)
            else:
                try:
                    value = float(token)
                except ValueError:
                    value = token
            kwargs[key] = value
            key = None
    return kwargs


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train surrogate sequence model")

    # Generic options
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory containing numpy arrays")
    parser.add_argument("--output_dir", type=str, default="models", help="Where to save checkpoints & logs")
    parser.add_argument("--version", type=str, required=True, help="Experiment version identifier (e.g., 0.1.0-gru)")
    parser.add_argument("--model", type=str, default="lstm", help="Model architecture to use (registered key)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false", help="Disable Weights & Biases logging")
    parser.set_defaults(use_wandb=True)

    args, unknown = parser.parse_known_args()
    model_kwargs = parse_model_kwargs(unknown)

    # Resolve paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / args.data_dir
    run_dir = project_root / args.output_dir / args.version
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data(data_dir)
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=args.batch_size
    )

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    ModelCls = get_model_class(args.model)
    input_size = X_train.shape[2] if X_train.ndim == 3 else X_train.shape[1]
    if y_train.ndim == 3:
        prediction_horizon = y_train.shape[1]
        output_size = y_train.shape[2]
    else:
        prediction_horizon = 1
        output_size = y_train.shape[1] if y_train.ndim > 1 else 1

    model = ModelCls(
        input_size=input_size,
        output_size=output_size,
        prediction_horizon=prediction_horizon,
        learning_rate=args.learning_rate,
        **model_kwargs,
    )

    # ------------------------------------------------------------------
    # Trainer & callbacks
    # ------------------------------------------------------------------
    checkpoint_cb = ModelCheckpoint(
        dirpath=run_dir,
        filename="{model}-{epoch:02d}-{val_loss:.4f}".format(model=args.model),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=15)
    callbacks = [checkpoint_cb, early_stopping]

    logger = None
    if args.use_wandb:
        logger = WandbLogger(
            project="rl-bem-surrogate",
            name=args.version,
            save_dir=str(project_root / "wandb"),
        )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        val_check_interval=0.25,
        gradient_clip_val=1.0,
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    trainer.fit(model, train_loader, val_loader)

    # Use best checkpoint for evaluation
    best_path = checkpoint_cb.best_model_path or None
    if best_path:
        model = ModelCls.load_from_checkpoint(best_path)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    metrics, preds, targets = evaluate_model(model, test_loader, device)

    print("\nTest results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Persist artefacts
    plot_predictions(preds, targets, run_dir / "prediction_plots.png")

    trainer.save_checkpoint(run_dir / "model_final.ckpt")

    if args.use_wandb and logger is not None:
        import wandb

        wandb.log(metrics)
        wandb.log({"prediction_plot": wandb.Image(str(run_dir / "prediction_plots.png"))})
        wandb.finish()


if __name__ == "__main__":
    main() 