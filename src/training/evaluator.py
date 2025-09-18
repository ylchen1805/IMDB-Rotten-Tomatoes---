"""Evaluation utilities for sentiment analysis models."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    return_predictions: bool = False
) -> Dict:
    """Evaluate model and compute metrics.

    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to use
        return_predictions: Whether to return raw predictions

    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    model = model.to(device)

    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["ids"].to(device)
            attention_mask = batch.get("attention_mask", None)

            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Get predictions
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs, dim=-1)
            predictions = outputs.argmax(dim=-1)

            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if "label" in batch:
                labels = batch["label"]
                all_labels.extend(labels.cpu().numpy())

    # Calculate metrics if labels are available
    metrics = {}
    if all_labels:
        metrics = {
            "accuracy": accuracy_score(all_labels, all_predictions),
            "f1_score": f1_score(all_labels, all_predictions, average="weighted"),
            "precision": precision_score(all_labels, all_predictions, average="weighted"),
            "recall": recall_score(all_labels, all_predictions, average="weighted"),
            "confusion_matrix": confusion_matrix(all_labels, all_predictions).tolist(),
            "classification_report": classification_report(
                all_labels,
                all_predictions,
                target_names=["negative", "positive"]
            )
        }

    if return_predictions:
        metrics["predictions"] = all_predictions
        metrics["probabilities"] = all_probs

    return metrics


def predict_sentiment(
    model: nn.Module,
    texts: List[str],
    tokenizer,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 512
) -> List[Dict]:
    """Predict sentiment for a list of texts.

    Args:
        model: Trained model
        texts: List of text strings
        tokenizer: Tokenizer to use
        device: Device to use
        batch_size: Batch size for inference
        max_length: Maximum sequence length

    Returns:
        List of prediction dictionaries
    """
    model.eval()
    model = model.to(device)
    predictions = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize batch
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # Move to device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Get predictions
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs, dim=-1)
            preds = outputs.argmax(dim=-1)

        # Convert to readable format
        for j, text in enumerate(batch_texts):
            predictions.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "sentiment": "positive" if preds[j].item() == 1 else "negative",
                "confidence": probs[j].max().item(),
                "probabilities": {
                    "negative": probs[j][0].item(),
                    "positive": probs[j][1].item()
                }
            })

    return predictions