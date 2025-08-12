import argparse
import logging
import numpy as np
from pathlib import Path
import torch
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json

from models.tuning import HyperParams, model_factory
from dataproc.nosoi_split import NosoiSplit
from utils.logging_config import setup_logging


def plot_histogram(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    param_names: list[str],
    output_path: Path
):
    """
    Plot and save per-parameter error histograms with KDE overlays.

    This function computes the prediction errors for each parameter and creates
    a histogram with a kernel density estimate (KDE) to visualize the
    distribution of errors. Each plot is saved to file under the given output
    directory.

    Parameters
    ----------
    y_pred : np.ndarray
        Array of predicted values from the DNN, shape (n_samples,
        n_parameters).
    y_true : np.ndarray
        Array of true values from the dataset, shape (n_samples, n_parameters).
    param_names : list of str
        List of parameter names corresponding to the columns in y_pred and
        y_true.
    output_path : Path
        Directory in which the resulting histogram images will be saved.
    """
    for i, name in enumerate(param_names):
        param_errors = y_pred[:, i] - y_true[:, i]
        plt.figure(figsize=(6, 4))
        sns.histplot(param_errors, kde=True, color="skyblue")
        plt.title(f"Prediction Error for {name}")
        plt.xlabel("Prediction Error")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path / f"dnn_error_{name}.png", dpi=300)
        plt.close()


def run_predicted_vs_true(
    splits_path: Path,
    model_path: Path,
    output_path: Path,
    make_plots: bool
) -> None:
    """
    Evaluate a trained DNN on the test split and export prediction diagnostics.

    Loads the test data and a saved model, performs inference, computes
    prediction errors, and exports mean absolute error (MAE) per parameter.
    Optionally, per-parameter error histograms are plotted and saved.

    Parameters
    ----------
    splits_path : Path
        Path to the directory containing the pickled `NosoiSplit` test split.
    model_path : Path
        Path to the directory containing the trained DNN (`regressor.pt`) and
        its hyperparameter configuration (`best_config.json`).
    output_path : Path
        Directory where results such as error parquet, MAE values, and plots
        will be saved.
    make_plots : bool
        If True, generates and saves histogram plots of the prediction errors
        for each target parameter.
    """
    # Set up logger
    setup_logging("training")
    logger = logging.getLogger(__name__)

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data splits from disk
    test_split = NosoiSplit.load("test", splits_path)
    logger.info(f"Loaded saved data splits from {splits_path}")

    # Retrieve hyperparameters
    with open(model_path / "best_config.json") as handle:
        data = json.load(handle)
        cfg = HyperParams.from_dict(data)

    logger.info(f"HyperParams: {cfg}")

    # Prepare model architecture from loaded hyperparameters
    model = model_factory(
        test_split.input_dim,
        test_split.output_dim,
        cfg,
        device
    )

    # Load model
    model.load_state_dict(
        torch.load(model_path / "regressor.pt", map_location=device)
    )
    model.to(device)
    logger.info(f"Loaded saved DNN from: {model_path / 'regressor.pt'}")

    X = test_split.X.to(device)
    y_true = test_split.y
    param_names = test_split.y_colnames

    # Forward pass to get predictions
    with torch.no_grad():
        y_pred = model(X).cpu().numpy()

    # Undo log transform for p_fatal
    logger.info("Undoing log transform on p_fatal...")
    y_true = y_true.clone().cpu().numpy()
    y_pred = y_pred.copy()
    i_p_fatal = test_split.y_colnames.index("p_fatal")
    y_true[:, i_p_fatal] = np.exp(y_true[:, i_p_fatal])
    y_pred[:, i_p_fatal] = np.exp(y_pred[:, i_p_fatal])

    # Compute prediction errors and MAE
    logger.info("Computing MAE per parameter...")
    mae_values = mean_absolute_error(
        y_true,
        y_pred,
        multioutput='raw_values'
    )
    mae_per_param = {
        param: float(mae) for param, mae in zip(param_names, mae_values)
    }
    logging.info(mae_per_param)

    with open(output_path / "dnn_mae.json", 'w') as handle:
        json.dump(mae_per_param, handle)
    logger.info(f"Saved MAE values to {output_path / 'dnn_mae.json'}")

    # Compute per-parameter prediction errors
    df_errors = pd.DataFrame()
    for i, param in enumerate(param_names):
        errors = y_pred[:, i] - y_true[:, i]
        df_errors[param] = errors

    df_errors.to_parquet(output_path / "dnn_data.parquet", index=False)
    logger.info(
        f"Saved prediction errors to {output_path / 'dnn_data.parquet'}"
    )

    if make_plots:
        plot_histogram(y_pred, y_true, test_split.y_colnames, output_path)
        logger.info(f"Exported plots to: {output_path}")


def cli_main():
    parser = argparse.ArgumentParser(
        description="Compute MAE in predictions for a trained DNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--splits-path", type=str, default="data/splits/scarce_0.00",
        help="Path to the directory containing pickled NosoiSplit object."
    )
    parser.add_argument(
        "--model-path", type=str, default="data/dnn/scarce_0.00",
        help=(
            "Path to the directory containing the pickled DNN (regressor.pt) "
            "and hyperparameter configuration (best_config.json)"
        )
    )
    parser.add_argument(
        "--output-path", type=str, default="data/benchmarks",
        help="Directory to save output data."
    )

    parser.add_argument(
        "--make_plots", type=bool, default=False,
        help="Make and export prediction error distribution plots."
    )

    args = parser.parse_args()
    run_predicted_vs_true(
        splits_path=Path(args.splits_path),
        model_path=Path(args.model_path),
        output_path=Path(args.output_path),
        make_plots=args.make_plots
    )


if __name__ == "__main__":
    cli_main()
