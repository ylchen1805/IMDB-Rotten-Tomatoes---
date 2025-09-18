"""Training logic for sentiment analysis models."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """Trainer class for sentiment analysis models."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        gradient_clip: Optional[float] = 1.0
    ):
        """Initialize trainer.

        Args:
            model: PyTorch model to train
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            gradient_clip: Maximum gradient norm for clipping
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip

        # Initialize optimizer and loss
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "valid_loss": [],
            "valid_acc": []
        }

        # Best model tracking
        self.best_valid_loss = float("inf")
        self.best_model_path = None

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int = 0
    ) -> Tuple[float, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Average loss and accuracy for the epoch
        """
        self.model.train()
        epoch_losses = []
        epoch_accs = []

        progress_bar = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch}",
            leave=False
        )

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["ids"].to(self.device)
            labels = batch["label"].to(self.device)
            attention_mask = batch.get("attention_mask", None)

            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Forward pass
            outputs = self.model(input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs, labels)

            # Calculate accuracy
            accuracy = self._calculate_accuracy(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            self.optimizer.step()

            # Track metrics
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy)

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{accuracy:.4f}"
            })

        return np.mean(epoch_losses), np.mean(epoch_accs)

    def evaluate(
        self,
        valid_loader: DataLoader,
        desc: str = "Evaluating"
    ) -> Tuple[float, float]:
        """Evaluate model on validation set.

        Args:
            valid_loader: Validation data loader
            desc: Description for progress bar

        Returns:
            Average loss and accuracy
        """
        self.model.eval()
        epoch_losses = []
        epoch_accs = []

        with torch.no_grad():
            progress_bar = tqdm(valid_loader, desc=desc, leave=False)

            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["ids"].to(self.device)
                labels = batch["label"].to(self.device)
                attention_mask = batch.get("attention_mask", None)

                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs, labels)

                # Calculate accuracy
                accuracy = self._calculate_accuracy(outputs, labels)

                # Track metrics
                epoch_losses.append(loss.item())
                epoch_accs.append(accuracy)

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{accuracy:.4f}"
                })

        return np.mean(epoch_losses), np.mean(epoch_accs)

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        num_epochs: int = 3,
        save_dir: str = "./models",
        early_stopping_patience: int = 3
    ) -> Dict[str, List[float]]:
        """Train model for multiple epochs.

        Args:
            train_loader: Training data loader
            valid_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save best model
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history dictionary
        """
        os.makedirs(save_dir, exist_ok=True)
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # Validation phase
            valid_loss, valid_acc = self.evaluate(valid_loader)
            self.history["valid_loss"].append(valid_loss)
            self.history["valid_acc"].append(valid_acc)

            # Print epoch results
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

            # Save best model
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.best_model_path = Path(save_dir) / "best_model.pt"
                self.save_checkpoint(self.best_model_path)
                print(f"✓ Saved best model (valid_loss: {valid_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

            # Clean GPU cache periodically
            if epoch % 5 == 0:
                torch.cuda.empty_cache()

        return self.history

    def save_checkpoint(self, path: Path):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "best_valid_loss": self.best_valid_loss
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        # Security: Use safe loading
        try:
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=False  # Need to load optimizer state
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        self.best_valid_loss = checkpoint.get("best_valid_loss", float("inf"))

    def _calculate_accuracy(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """Calculate accuracy for predictions.

        Args:
            predictions: Model predictions (logits)
            labels: True labels

        Returns:
            Accuracy as float
        """
        predicted_classes = predictions.argmax(dim=-1)
        correct = predicted_classes.eq(labels).sum()
        accuracy = correct.float() / labels.size(0)
        return accuracy.item()