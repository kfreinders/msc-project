"""
Train and evaluate a deep neural network for nosoi parameter inference.

This script handles the full pipeline of loading and preprocessing nosoi
simulation data, training a DNN model to predict key epidemiological parameters
from summary statistics, evaluating the model on a held-out test set, and
visualizing the predictions.

Steps:
- Load and preprocess simulation data from CSV files
- Apply transformations (e.g. log-transform of p_fatal) The data preprocessing
  can be changed in python/data/nosoi_data_manger)
- Train a DNN on summary statistics using PyTorch
- Save the trained model
- Evaluate the model on the test set
- Generate prediction vs. ground-truth plots

Requirements:
- The summary statistics and master CSVs must be located in `data/nosoi/`
- Output files are saved under `data/splits/` and `data/dnn/`
"""

import logging
from utils.logging_config import setup_logging
from models.model import NeuralNetwork
import os
import torch
from torch import nn, optim
from utils.utils import (
    evaluate_model,
    train_model,
    predict_nosoi_parameters,
    plot_predictions,
)
from dataproc.nosoi_data_manger import prepare_nosoi_data


# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------


def main() -> None:
    # Set up logger
    setup_logging("training")
    logger = logging.getLogger(__name__)

    # Preprocess split datasets
    train_tensor, val_tensor, test_tensor = prepare_nosoi_data(
        summary_stats_csv="data/nosoi/summary_stats_export.csv",
        master_csv="data/nosoi/master.csv",
        output_dir="data/splits",
        overwrite=False
    )

    # Build dataloaders
    train_loader = train_tensor.make_dataloader(shuffle=True)
    val_loader = val_tensor.make_dataloader(shuffle=True)
    test_loader = test_tensor.make_dataloader()

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(
        f"DNN model will have {train_tensor.input_dim} inputs and "
        f"{train_tensor.output_dim} outputs"
    )

    # Initialize model
    logger.info("Initializing model...")
    model = NeuralNetwork(
        input_dim=train_tensor.input_dim,
        output_dim=train_tensor.output_dim,
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
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=100,
        patience=5,
    )

    # Save the trained model
    # TODO: save existing regressor.pt as "regressor-n.pt" and save current
    os.makedirs("data/dnn", exist_ok=True)
    torch.save(model.state_dict(), "data/dnn/regressor.pt")
    logger.info("Model saved to 'data/dnn/regressor.pt'.")

    # Evaluate model
    test_loss = evaluate_model(trained_model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f}")

    # # Load model (redundant in this case since already in memory)
    model.load_state_dict(
        torch.load("data/dnn/regressor.pt", map_location=device)
    )
    model.to(device)

    # Predict
    # TODO: write preds and trues for later usage
    preds, trues = predict_nosoi_parameters(model, test_loader, device)

    # Plot
    param_names = [
        "mean_t_incub",
        "stdv_t_incub",
        "infectivity",
        "p_trans",
        "p_fatal",
        "t_recovery",
    ]

    # TODO: also add a plot of model trained to predict mean_nContact and
    # p_trans instead of infectivity, and plot both the predictions for these
    # as well as their product (which is the infectivity)

    # TODO: extract this functionality
    sst_11_values = test_tensor.get_raw_feature("SST_11")
    fig = plot_predictions(preds, trues, param_names, sst_11_values)

    fig.savefig("predicted_vs_true.png", dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    main()
