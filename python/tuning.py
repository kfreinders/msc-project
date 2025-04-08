import itertools
import json
import logging

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from logging_config import setup_logging
from model import NeuralNetwork
from utils import train_model, load_data, split_data


def generate_paramsets(params: dict[str, list[float]]) -> list[dict]:
    """Generate all hyperparameter combinations."""
    keys, values = zip(*params.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations


def build_dataloaders(
        dataset: torch.utils.data.TensorDataset, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    """Split dataset and create DataLoaders."""
    train, val, _ = split_data(
        dataset, ptrain=0.8, pval=0.2, ptest=0.0, batch_size=batch_size
    )
    return train, val


def build_model(config: dict, device: torch.device) -> torch.nn.Module:
    """Build a model based on the hyperparameters."""
    model = NeuralNetwork(
        input_dim=26,
        output_dim=6,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout_rate=config["dropout_rate"],
    )
    return model.to(device)


def train_and_evaluate(
    model: torch.nn.Module,
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device
) -> float:
    """Train and evaluate a single configuration."""
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()

    _, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=100,
        patience=5,
    )

    final_val_loss = min(history["avg_val_loss"])
    return final_val_loss


def main() -> None:
    # Set up logger
    setup_logging()
    logger = logging.getLogger(__name__)

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading dataset...")
    dataset = load_data("data/nosoi/merged.csv")

    # Define the hyperparameter and architecture search space
    search_space: dict[str, list[float]] = {
        "learning_rate": [1e-2, 1e-3, 1e-4],
        "hidden_size": [32, 64, 128, 256],
        "num_layers": [2, 3, 4, 5],
        "dropout_rate": [0.1, 0.2, 0.3],
        "batch_size": [32, 64, 128],
    }

    # Generate all possible combinations for a full grid search
    hyperparameter_combinations = generate_paramsets(search_space)
    logger.info(
        f"Total configurations to try: {len(hyperparameter_combinations)}"
    )

    # Try to resume from existing results
    results = []
    try:
        with open("data/dnn/tuning_results.json", "r") as f:
            loaded = json.load(f)
            results = loaded.get("results", [])
            logger.info(
                f"Loaded {len(results)} previously saved configurations."
            )
    except FileNotFoundError:
        logger.info("No previous results found. Starting new tuning run.")

    # Build set of already tried configurations
    tried_configs = {json.dumps(r[0], sort_keys=True) for r in results}

    # Start full grid search
    for idx, config in enumerate(hyperparameter_combinations, start=1):
        # Skip this configuration if we've tried it before
        config_serialized = json.dumps(config, sort_keys=True)
        if config_serialized in tried_configs:
            logger.info(
                f"Skipping already tried configuration "
                f"{idx}/{len(hyperparameter_combinations)}"
            )
            continue

        logger.info(
            f"Training configuration {idx}/{len(hyperparameter_combinations)}"
        )
        logger.debug(f"Configuration details: {config}")

        # Create dataloaders with specified batch size
        train, val = build_dataloaders(dataset, config["batch_size"])

        # Build model dynamically
        model = build_model(config, device)

        # Train and pick best val loss during training
        final_val_loss = train_and_evaluate(
            model, config, train, val, device
        )

        # Save the config and its performance
        logger.debug(
            f"Final validation loss for current config: {final_val_loss:.4f}"
        )
        results.append((config, final_val_loss))

        # Save a checkpoint after every configuration
        with open("data/dnn/tuning_results.json", "w") as f:
            json.dump({"results": results}, f, indent=4)

    # Sort by lowest validation loss
    results.sort(key=lambda x: x[1])
    best_config, best_loss = results[0]

    # Print best result
    logger.info(
        f"Best hyperparameters found: {best_config} "
        f"with validation loss {best_loss}"
    )

    # Save results
    with open("tuning_results.json", "w") as f:
        json.dump(
            {"best_config": best_config, "best_loss": best_loss}, f, indent=4
        )


if __name__ == "__main__":
    main()
