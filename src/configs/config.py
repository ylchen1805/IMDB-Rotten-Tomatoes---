"""Configuration management for sentiment analysis."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataConfig:
    """Data configuration."""
    train_path: str = "data/train.csv"
    test_path: str = "data/test.csv"
    validation_split: float = 0.05
    remove_duplicates: bool = False
    max_length: int = 512
    text_column: str = "review"
    label_column: str = "sentiment"


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = "bert"  # bert, roberta
    model_name: str = "bert-base-uncased"
    num_classes: int = 2
    dropout_rate: float = 0.1
    freeze_backbone: bool = False


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    gradient_clip: float = 1.0
    early_stopping_patience: int = 3
    save_dir: str = "./models"
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config object
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create config from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config object
        """
        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))

        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            device=config_dict.get("device")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            "data": {
                "train_path": self.data.train_path,
                "test_path": self.data.test_path,
                "validation_split": self.data.validation_split,
                "remove_duplicates": self.data.remove_duplicates,
                "max_length": self.data.max_length,
                "text_column": self.data.text_column,
                "label_column": self.data.label_column
            },
            "model": {
                "model_type": self.model.model_type,
                "model_name": self.model.model_name,
                "num_classes": self.model.num_classes,
                "dropout_rate": self.model.dropout_rate,
                "freeze_backbone": self.model.freeze_backbone
            },
            "training": {
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "num_epochs": self.training.num_epochs,
                "gradient_clip": self.training.gradient_clip,
                "early_stopping_patience": self.training.early_stopping_patience,
                "save_dir": self.training.save_dir,
                "seed": self.training.seed,
                "num_workers": self.training.num_workers,
                "pin_memory": self.training.pin_memory
            },
            "device": self.device
        }

    def save(self, yaml_path: str):
        """Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

        print(f"Configuration saved to: {yaml_path}")


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or use defaults.

    Args:
        config_path: Optional path to config file

    Returns:
        Config object
    """
    if config_path and Path(config_path).exists():
        return Config.from_yaml(config_path)

    # Check for default config file
    default_config = Path("config.yaml")
    if default_config.exists():
        return Config.from_yaml(default_config)

    # Return default configuration
    return Config()