import numpy as np
import logging
from logging_config import setup_logging
from model import NeuralNetwork
import os
import torch
from torch import nn, optim
from utils import (
    evaluate_model,
    load_data,
    split_data_with_meta,
    merge_summary_and_parameters,
    train_model,
    predict_parameters,
    plot_predictions,
)


def log_transform(x: np.ndarray) -> np.ndarray:
    """Wrapper for np.log to please the type checker."""
    return np.log(x)


def sqrt_transform(x: np.ndarray) -> np.ndarray:
    """Wrapper for np.sqrt to further please the type checker."""
    return np.sqrt(x)

# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------


def main() -> None:
    # Set up logger
    setup_logging("training")
    logger = logging.getLogger(__name__)

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Merge summary statistic and nosoi parameter files
    logger.info("Merging summary statistic and nosoi parameter files...")
    merge_summary_and_parameters(
        "data/nosoi/summary_stats_export.csv",
        "data/nosoi/master.csv",
        "data/nosoi/merged.csv",
        filter_fn=lambda df: df["SS_11"] > 2000,
        filter_fn_desc="SS_11 > 2000"
    )

    # Read the merged csv file and also retrieve summary statistic 11, which is
    # the total no. hosts at the end of the simulation.
    logger.info("Generating dataset from csv...")

    # Which transforms to apply to the data for training
    transform_map = {
        "p_fatal": log_transform
    }

    dataset, meta, log_idxs = load_data(
        "data/nosoi/merged.csv",
        extract_columns=["SS_11"],
        target_transforms=transform_map,
        use_infectivity=True
    )

    # Split dataset into training, validation and testing sets
    logger.info(
        "Splitting dataset into training, validation and testing sets..."
    )
    train, val, test, _, val_meta, _ = split_data_with_meta(
        dataset,
        meta,
        ptrain=0.7,
        pval=0.15,
        ptest=1 - 0.7 - 0.15,
        batch_size=32
    )

    # Initialize model
    logger.info("Initializing model...")
    model = NeuralNetwork(
        input_dim=26,
        output_dim=5,
        hidden_size=256,
        num_layers=4,
        dropout_rate=0.1
    ).to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    logger.info("Training model...")
    trained_model, _ = train_model(
        model,
        train,
        val,
        criterion,
        optimizer,
        device,
        epochs=100,
        patience=5,
    )

    # Save the trained model
    os.makedirs("data/dnn", exist_ok=True)
    torch.save(model.state_dict(), "data/dnn/regressor.pt")
    logger.info("Model saved to 'data/dnn/regressor.pt'.")

    # Evaluate model
    test_loss = evaluate_model(trained_model, test, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f}")

    # Load model (redundant in this case since already in memory)
    model.load_state_dict(
        torch.load("data/dnn/regressor.pt", map_location=device)
    )
    model.to(device)

    # Predict
    preds, trues = predict_parameters(model, val, device)

    # Apply inverse transform for plotting
    for i in log_idxs:
        preds[:, i] = np.exp(preds[:, i])
        trues[:, i] = np.exp(trues[:, i])

    # Plot
    param_names = [
        "mean_t_incub",
        "stdv_t_incub",
        "infectivity",
        "p_trans",
        "p_fatal",
        "t_recovery",
    ]

    fig = plot_predictions(preds, trues, param_names, val_meta["SS_11"])
    fig.savefig("predicted_vs_true.png", dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    main()
