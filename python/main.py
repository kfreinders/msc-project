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

# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------


def main() -> None:
    # Set up logger
    setup_logging()
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
    dataset, meta = load_data("data/nosoi/merged.csv", ["SS_11"])

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
        output_dim=6,
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

    # Plot
    param_names = [
        "mean_nContact",
        "mean_t_incub",
        "mean_p_trans",
        "p_fatal",
        "p_trans",
        "t_recovery",
    ]

    fig = plot_predictions(preds, trues, param_names, val_meta["SS_11"])
    fig.savefig("predicted_vs_true.png", dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    main()
