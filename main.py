#!/usr/bin/env python3
"""Main entry point for sentiment analysis training and inference."""

import os
import sys
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import torch
import transformers
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.configs import Config, load_config
from src.data import SentimentDataset, TestDataset, create_data_loader, preprocess_data
from src.data.datasets import load_data
from src.models import BertSentimentModel, RobertaSentimentModel
from src.training import Trainer, evaluate_model
from src.training.evaluator import predict_sentiment
from src.utils import get_device, set_seed


@click.group()
@click.version_option(version="0.2.0")
def cli():
    """IMDB/Rotten Tomatoes Sentiment Analysis CLI."""
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file (YAML)"
)
@click.option(
    "--model-type",
    type=click.Choice(["bert", "roberta"]),
    default="bert",
    help="Model type to use"
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    help="Number of training epochs"
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    help="Training batch size"
)
@click.option(
    "--learning-rate",
    "-lr",
    type=float,
    help="Learning rate"
)
@click.option(
    "--device",
    "-d",
    type=str,
    help="Device to use (cuda/cpu)"
)
def train(
    config: Optional[str],
    model_type: Optional[str],
    epochs: Optional[int],
    batch_size: Optional[int],
    learning_rate: Optional[float],
    device: Optional[str]
):
    """Train a sentiment analysis model."""
    # Load configuration
    cfg = load_config(config)

    # Override config with command line arguments
    if model_type:
        cfg.model.model_type = model_type
    if epochs:
        cfg.training.num_epochs = epochs
    if batch_size:
        cfg.training.batch_size = batch_size
    if learning_rate:
        cfg.training.learning_rate = learning_rate
    if device:
        cfg.device = device

    # Set up device and seed
    device = get_device(cfg.device)
    set_seed(cfg.training.seed)

    print("\n" + "=" * 50)
    print("SENTIMENT ANALYSIS TRAINING")
    print("=" * 50)
    print(f"\nConfiguration:")
    print(f"  Model: {cfg.model.model_type} ({cfg.model.model_name})")
    print(f"  Epochs: {cfg.training.num_epochs}")
    print(f"  Batch Size: {cfg.training.batch_size}")
    print(f"  Learning Rate: {cfg.training.learning_rate}")
    print(f"  Device: {device}")

    # Load and preprocess data
    print("\n📊 Loading data...")
    try:
        train_data = load_data(cfg.data.train_path)
        print(f"  Loaded {len(train_data)} training samples")
    except Exception as e:
        click.echo(f"❌ Error loading training data: {e}", err=True)
        sys.exit(1)

    # Initialize tokenizer based on model type
    print("\n🔤 Initializing tokenizer...")
    if cfg.model.model_type == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained(cfg.model.model_name)
        model_class = BertSentimentModel
    elif cfg.model.model_type == "roberta":
        cfg.model.model_name = "roberta-base"
        tokenizer = transformers.RobertaTokenizer.from_pretrained(cfg.model.model_name)
        model_class = RobertaSentimentModel
    else:
        click.echo(f"❌ Unknown model type: {cfg.model.model_type}", err=True)
        sys.exit(1)

    # Preprocess data
    print("\n⚙️  Preprocessing data...")
    train_data = preprocess_data(
        train_data,
        tokenizer,
        label_column=cfg.data.label_column,
        text_column=cfg.data.text_column,
        remove_duplicates=cfg.data.remove_duplicates
    )

    # Split into train and validation
    print(f"\n✂️  Splitting data (validation ratio: {cfg.data.validation_split})...")
    train_df, valid_df = train_test_split(
        train_data,
        test_size=cfg.data.validation_split,
        random_state=cfg.training.seed,
        stratify=train_data[cfg.data.label_column]
    )
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    print(f"  Train: {len(train_df)} samples")
    print(f"  Valid: {len(valid_df)} samples")

    # Create datasets and dataloaders
    print("\n📦 Creating datasets...")
    train_dataset = SentimentDataset(train_df)
    valid_dataset = SentimentDataset(valid_df)

    train_loader = create_data_loader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        pad_token_id=tokenizer.pad_token_id,
        shuffle=True,
        include_attention_mask=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory
    )
    valid_loader = create_data_loader(
        valid_dataset,
        batch_size=cfg.training.batch_size,
        pad_token_id=tokenizer.pad_token_id,
        shuffle=False,
        include_attention_mask=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory
    )

    # Initialize model
    print("\n🤖 Initializing model...")
    model = model_class(
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        dropout_rate=cfg.model.dropout_rate,
        freeze_backbone=cfg.model.freeze_backbone
    )
    print(f"  Total parameters: {model.get_num_parameters():,}")
    print(f"  Trainable parameters: {model.get_num_parameters(trainable_only=True):,}")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        gradient_clip=cfg.training.gradient_clip
    )

    # Train model
    print("\n🚀 Starting training...")
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=cfg.training.num_epochs,
        save_dir=cfg.training.save_dir,
        early_stopping_patience=cfg.training.early_stopping_patience
    )

    print("\n✅ Training completed!")
    print(f"  Best validation loss: {trainer.best_valid_loss:.4f}")
    print(f"  Best model saved to: {trainer.best_model_path}")

    # Save configuration
    config_save_path = Path(cfg.training.save_dir) / "config.yaml"
    cfg.save(config_save_path)


@cli.command()
@click.option(
    "--model-path",
    "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained model checkpoint"
)
@click.option(
    "--test-path",
    "-t",
    type=click.Path(exists=True),
    help="Path to test data CSV"
)
@click.option(
    "--output",
    "-o",
    type=str,
    default="submission.csv",
    help="Output file path for predictions"
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file"
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=32,
    help="Inference batch size"
)
def predict(
    model_path: str,
    test_path: Optional[str],
    output: str,
    config: Optional[str],
    batch_size: int
):
    """Generate predictions using a trained model."""
    # Load configuration
    cfg = load_config(config)

    # Set up device
    device = get_device(cfg.device)

    print("\n" + "=" * 50)
    print("SENTIMENT ANALYSIS PREDICTION")
    print("=" * 50)

    # Determine model type from config or checkpoint
    model_dir = Path(model_path).parent
    config_file = model_dir / "config.yaml"
    if config_file.exists():
        cfg = load_config(str(config_file))
        print(f"  Loaded config from: {config_file}")

    # Initialize tokenizer
    print("\n🔤 Initializing tokenizer...")
    if cfg.model.model_type == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained(cfg.model.model_name)
        model_class = BertSentimentModel
    else:
        tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
        model_class = RobertaSentimentModel

    # Load model
    print(f"\n🤖 Loading model from: {model_path}")
    model = model_class(
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # Load test data
    test_path = test_path or cfg.data.test_path
    print(f"\n📊 Loading test data from: {test_path}")
    test_data = load_data(test_path)
    print(f"  Loaded {len(test_data)} test samples")

    # Preprocess test data
    print("\n⚙️  Preprocessing test data...")
    tqdm.pandas()
    test_data["input_ids"] = test_data[cfg.data.text_column].progress_apply(
        lambda x: tokenizer(x, truncation=True)["input_ids"]
    )

    # Create test dataset and loader
    test_dataset = TestDataset(test_data)
    test_loader = create_data_loader(
        test_dataset,
        batch_size=batch_size,
        pad_token_id=tokenizer.pad_token_id,
        shuffle=False,
        include_attention_mask=True
    )

    # Generate predictions
    print("\n🔮 Generating predictions...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch["ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = outputs.argmax(dim=-1)
            predictions.extend(preds.cpu().numpy())

    # Create submission dataframe
    submission = pd.DataFrame({
        "id": test_data["id"] if "id" in test_data.columns else range(len(predictions)),
        "sentiment": ["positive" if p == 1 else "negative" for p in predictions]
    })

    # Save predictions
    submission.to_csv(output, index=False)
    print(f"\n✅ Predictions saved to: {output}")
    print(f"  Total predictions: {len(submission)}")
    print(f"  Positive: {(submission['sentiment'] == 'positive').sum()}")
    print(f"  Negative: {(submission['sentiment'] == 'negative').sum()}")


@cli.command()
@click.option(
    "--model-path",
    "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained model checkpoint"
)
@click.option(
    "--data-path",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Path to evaluation data CSV"
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file"
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=32,
    help="Evaluation batch size"
)
def evaluate(
    model_path: str,
    data_path: str,
    config: Optional[str],
    batch_size: int
):
    """Evaluate a trained model on labeled data."""
    # Load configuration
    cfg = load_config(config)

    # Set up device
    device = get_device(cfg.device)

    print("\n" + "=" * 50)
    print("SENTIMENT ANALYSIS EVALUATION")
    print("=" * 50)

    # Load model configuration
    model_dir = Path(model_path).parent
    config_file = model_dir / "config.yaml"
    if config_file.exists():
        cfg = load_config(str(config_file))

    # Initialize tokenizer
    print("\n🔤 Initializing tokenizer...")
    if cfg.model.model_type == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained(cfg.model.model_name)
        model_class = BertSentimentModel
    else:
        tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
        model_class = RobertaSentimentModel

    # Load model
    print(f"\n🤖 Loading model from: {model_path}")
    model = model_class(
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes
    )

    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)

    # Load and preprocess data
    print(f"\n📊 Loading data from: {data_path}")
    eval_data = load_data(data_path)
    print(f"  Loaded {len(eval_data)} samples")

    # Preprocess data
    print("\n⚙️  Preprocessing data...")
    eval_data = preprocess_data(
        eval_data,
        tokenizer,
        label_column=cfg.data.label_column,
        text_column=cfg.data.text_column
    )

    # Create dataset and loader
    eval_dataset = SentimentDataset(eval_data)
    eval_loader = create_data_loader(
        eval_dataset,
        batch_size=batch_size,
        pad_token_id=tokenizer.pad_token_id,
        shuffle=False,
        include_attention_mask=True
    )

    # Evaluate model
    print("\n📈 Evaluating model...")
    metrics = evaluate_model(model, eval_loader, device)

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"\n📊 Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")

    print(f"\n📋 Classification Report:")
    print(metrics['classification_report'])

    print(f"\n🔢 Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  TN: {cm[0][0]:5d}  FP: {cm[0][1]:5d}")
    print(f"  FN: {cm[1][0]:5d}  TP: {cm[1][1]:5d}")


@cli.command()
@click.argument("text", nargs=-1, required=True)
@click.option(
    "--model-path",
    "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained model checkpoint"
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file"
)
def analyze(text: tuple, model_path: str, config: Optional[str]):
    """Analyze sentiment of provided text."""
    # Join text arguments
    text = " ".join(text)

    # Load configuration
    cfg = load_config(config)
    device = get_device(cfg.device)

    # Load model configuration
    model_dir = Path(model_path).parent
    config_file = model_dir / "config.yaml"
    if config_file.exists():
        cfg = load_config(str(config_file))

    # Initialize tokenizer
    if cfg.model.model_type == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained(cfg.model.model_name)
        model_class = BertSentimentModel
    else:
        tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
        model_class = RobertaSentimentModel

    # Load model
    model = model_class(
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes
    )

    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Predict sentiment
    results = predict_sentiment(
        model=model,
        texts=[text],
        tokenizer=tokenizer,
        device=device
    )

    # Display results
    result = results[0]
    print("\n" + "=" * 50)
    print("SENTIMENT ANALYSIS")
    print("=" * 50)
    print(f"\n📝 Text: {result['text']}")
    print(f"\n🎭 Sentiment: {result['sentiment'].upper()}")
    print(f"🎯 Confidence: {result['confidence']:.2%}")
    print(f"\n📊 Probabilities:")
    print(f"  Positive: {result['probabilities']['positive']:.2%}")
    print(f"  Negative: {result['probabilities']['negative']:.2%}")


if __name__ == "__main__":
    cli()