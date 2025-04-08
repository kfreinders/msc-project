import torch
import itertools
from model import NeuralNetwork
from utils import train_model, load_data, split_data
from torch import nn, optim
import logging
from logging_config import setup_logging


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
    search_space = {
        "learning_rate": [1e-2, 1e-3, 1e-4],
        "hidden_size": [32, 64, 128, 256],
        "num_layers": [2, 3, 4, 5],
        "dropout_rate": [0.1, 0.2, 0.3],
        "batch_size": [32, 64, 128],
    }

    # Generate all possible combinations for a full grid search
    keys, values = zip(*search_space.items())
    hyperparameter_combinations = [
        dict(zip(keys, v)) for v in itertools.product(*values)
    ]
    logger.info(
        f"Total configurations to try: {len(hyperparameter_combinations)}"
    )

    # Start full grid search
    results = []
    for idx, config in enumerate(hyperparameter_combinations, start=1):
        logger.info(
            f"Training configuration {idx}/{len(hyperparameter_combinations)}"
        )
        logger.debug(f"Configuration details: {config}")

        # Create dataloaders with specified batch size
        train, val, _ = split_data(
            dataset,
            ptrain=0.8,
            pval=0.2,
            ptest=0.0,
            batch_size=config["batch_size"],
        )

        # Build model dynamically
        model = NeuralNetwork(
            input_dim=26,
            output_dim=6,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout_rate=config["dropout_rate"],
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        criterion = nn.MSELoss()

        # Train
        _, history = train_model(
            model,
            train,
            val,
            criterion,
            optimizer,
            device,
            epochs=100,
            patience=5
        )

        final_val_loss = min(
            history["avg_val_loss"]
        )  # Pick best val loss during training

        # Save the config and its performance
        results.append((config, final_val_loss))
        logger.debug(
            f"Final validation loss for current config: {final_val_loss:.4f}"
        )

    # Sort by lowest validation loss
    results.sort(key=lambda x: x[1])

    # Print best result
    logger.info(
        f"Best hyperparameters found: {results[0][0]} with validation loss {results[0][1]:.4f}"
    )


if __name__ == "__main__":

    main()
