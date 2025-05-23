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
    train_model,
    predict_nosoi_parameters,
    plot_predictions,
)
from nosoi_data_manger import NosoiDataManager


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

    # Which transforms to apply to the data for training
    transform_map = {
        "PAR_p_fatal": log_transform
    }

    manager = NosoiDataManager(
        "data/nosoi/summary_stats_export.csv",
        "data/nosoi/master.csv"
    )

    # Drop simulations with less than 2000 individuals. Below this size,
    # transmission chains seem to not have enough signal for the DNN to infer
    # the nosoi parameters
    manager.drop_by_filter(lambda df: df["SST_11"] > 2000, "SST_11 > 2000")

    # Replace mean_nContact and p_trans by their product, which is the
    # infectivity.
    manager.apply_infectivity()

    # Apply transforms to data. Currently, only predictions of p_fatal benefit
    # from a log transform but do so significantly
    manager.apply_target_transforms(transform_map)

    dataset, meta, log_idxs = load_data(
        "data/nosoi/merged.csv",
        extract_columns=["SS_11"],
        target_transforms=transform_map,
        use_infectivity=True
    )

    # Get input and output data dimensions from the dataset
    input_dim = dataset[0][0].shape[0]   # First sample’s input tensor shape
    output_dim = dataset[0][1].shape[0]  # First sample’s target tensor shape
    logger.info(
        f"DNN model will have {input_dim} inputs and {output_dim} outputs"
    )

    # Split dataset into training, validation and testing sets
    logger.info(
        "Splitting dataset into training, validation and testing sets..."
    )
    train, val, test, _, _, test_meta = split_data_with_meta(
        dataset,
        meta,
        ptrain=0.7,
        pval=0.15,
        batch_size=32
    )

    # Initialize model
    logger.info("Initializing model...")
    model = NeuralNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_size=256,
        num_hidden_layers=4,
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
    preds, trues = predict_nosoi_parameters(model, test, device)

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

    fig = plot_predictions(preds, trues, param_names, test_meta["SS_11"])
    fig.savefig("predicted_vs_true.png", dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    main()
