import argparse
import logging
import numpy as np
from pathlib import Path
import torch
import json
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from models.interfaces import TrainableModel
from models.tuning import HyperParams, model_factory
from dataproc.nosoi_split import NosoiSplit
from utils.logging_config import setup_logging
from utils.utils import predict_nosoi_parameters


def load_model_and_data(
    splits_path: Path,
    model_path: Path,
    device: torch.device
) -> tuple[NosoiSplit, TrainableModel, HyperParams]:
    """
    Load the trained model, hyperparameters, and test data split.

    This function restores the test dataset split, reads the best
    hyperparameter configuration, and loads the corresponding trained neural
    network model onto the specified device.

    Parameters
    ----------
    splits_path : Path
        Path to the directory containing serialized NosoiSplit objects.
    model_path : Path
        Path to the directory containing the trained model and its
        hyperparameter configuration.
    device : torch.device
        The device (CPU or CUDA) on which to load the model.

    Returns
    -------
    tuple[NosoiSplit, TrainableModel, HyperParams]
        The test split, the trained model ready for inference, and the
        hyperparameter configuration used for training.
    """
    test_split = NosoiSplit.load("test", splits_path)

    with open(model_path / "best_config.json") as handle:
        cfg = HyperParams.from_dict(json.load(handle))

    model = model_factory(
        test_split.input_dim,
        test_split.output_dim,
        cfg,
        device
    )
    model.load_state_dict(
        torch.load(model_path / "regressor.pt", map_location=device)
    )
    model.to(device)

    return test_split, model, cfg


def compute_baseline_r2(
    model: TrainableModel,
    test_split: NosoiSplit,
    cfg: HyperParams,
    device: torch.device
):
    """
    Compute baseline predictive performance of the trained model.

    Uses the provided test split to generate predictions and calculates the
    coefficient of determination for each predicted parameter.

    Parameters
    ----------
    model : TrainableModel
        The trained deep neural network model.
    test_split : NosoiSplit
        The test dataset split containing inputs and true outputs.
    cfg : HyperParams
        Hyperparameter configuration, including batch size.
    device : torch.device
        The device (CPU or CUDA) to perform inference on.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Baseline R squared scores, predicted parameter values, and true values.
    """

    test_loader = test_split.make_dataloader(cfg.batch_size)
    preds, trues = predict_nosoi_parameters(model, test_loader, device)
    baseline_r2 = r2_score(trues, preds, multioutput="raw_values")
    return baseline_r2, preds, trues


def perturb_features(
    model: TrainableModel,
    test_split: NosoiSplit,
    y_true: np.ndarray,
    baseline_r2: np.ndarray,
    cfg: HyperParams,
    device: torch.device,
    repeats: int
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """
    Perform perturbation analysis to assess feature importance.

    Each input feature is independently shuffled while keeping all other
    features intact. Predictions are made on the perturbed dataset, and the
    resulting drop in R squared compared to the baseline is recorded. This
    process is repeated multiple times per feature to reduce the impact of
    randomness.

    Parameters
    ----------
    model : TrainableModel
        The trained DNN model used for inference.
    test_split : NosoiSplit
        The test dataset split containing features and metadata.
    y_true : np.ndarray
        True parameter values for the test split.
    baseline_r2 : np.ndarray
        Baseline R squared scores for the unperturbed data.
    cfg : HyperParams
        Hyperparameter configuration, including batch size.
    device : torch.device
        Device (CPU or CUDA) on which inference is performed.
    repeats : int
        Number of times to repeat the shuffling per feature.

    Returns
    -------
    list[tuple[str, np.ndarray, np.ndarray]]
        A list where each element corresponds to a feature and contains its
        name, the mean change in R squared across repeats, and the standard
        deviation.
    """
    X = test_split.X.clone().detach().to(device)
    results = []

    for i, feature in enumerate(test_split.x_colnames):
        diffs = []
        for _ in range(repeats):
            X_shuffled = X.clone()
            idx = torch.randperm(X_shuffled.size(0))
            X_shuffled[:, i] = X_shuffled[idx, i]

            ds = TensorDataset(X_shuffled, torch.zeros(len(X_shuffled)))
            loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

            preds_shuffled, _ = predict_nosoi_parameters(model, loader, device)
            r2_shuffled = r2_score(
                y_true, preds_shuffled, multioutput="raw_values"
            )
            diffs.append(r2_shuffled - baseline_r2)

        results.append(
            (feature, np.mean(diffs, axis=0), np.std(diffs, axis=0))
        )

    return results


def plot_perturbation(
    results: list[tuple[str, np.ndarray, np.ndarray]],
    y_colnames: list[str],
    output_path: Path
) -> None:
    """
    Plot perturbation analysis results.

    Generates a line plot with error bars showing the average change in R
    squared after shuffling each feature, compared to the baseline. A baseline
    case is included for reference at index -1.

    Parameters
    ----------
    results : list[tuple[str, np.ndarray, np.ndarray]]
        Feature importance results with mean and std R squared values.
    y_colnames : list[str]
        Names of the output parameters predicted by the model.
    output_path : Path
        Destination file path for saving the plot.
    """
    _, ax = plt.subplots(figsize=(12, 6))

    # Add baseline at index -1
    x = np.arange(-1, len(results))
    xticklabels = ["Baseline"] + [res[0].replace("SST_", "") for res in results]

    for param_idx, param_name in enumerate(y_colnames):
        # Baseline at delta R squared = 0, then the shuffled means
        means = [0.0] + [res[1][param_idx] for res in results]
        stds = [0.0] + [res[2][param_idx] for res in results]

        # Plot baseline + shuffled as one continuous line
        ax.errorbar(
            x,
            means,
            yerr=stds,
            marker="o",
            linestyle="-",
            capsize=6,
            capthick=1.5,
            label=param_name,
        )

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.set_ylabel("Δ R² (shuffled – baseline)")
    ax.set_xlabel("Shuffled Summary Statistic")
    ax.set_title("Perturbation Analysis of Feature Importance")

    # Place legend inside plot bottom right
    ax.legend(title="Predicted Parameters", loc="lower right")

    # Export plot
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def run_perturbation(
    splits_path: Path,
    model_path: Path,
    repeats: int,
    output_path: Path
) -> None:
    """
    Run perturbation analysis for a trained DNN.

    Loads the test dataset and model, computes the baseline R², performs
    repeated shuffling of each feature, and generates a plot summarizing the
    impact of each feature on model performance.

    Parameters
    ----------
    splits_path : Path
        Path to the directory containing the test NosoiSplit.
    model_path : Path
        Path to the directory containing the trained model.
    repeats : int
        Number of times each feature is shuffled for averaging.
    output_path : Path
        File path to save the perturbation analysis plot.
    """
    # Set up logger
    setup_logging("predicted_vs_true")
    logger = logging.getLogger(__name__)

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    test_split, model, cfg = load_model_and_data(
        splits_path, model_path, device
    )

    logger.info("Making predictions...")
    test_loader = test_split.make_dataloader(cfg.batch_size)
    preds, trues = predict_nosoi_parameters(model, test_loader, device)

    baseline_r2 = r2_score(trues, preds, multioutput='raw_values')

    print(type(baseline_r2))

    results = perturb_features(
        model,
        test_split,
        trues,
        baseline_r2,
        cfg,
        device,
        repeats
    )

    plot_perturbation(results, test_split.y_colnames, output_path)


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Perform a perturbation analysis to assess feature importance in "
            "a trained DNN. Each summary statistic is independently shuffled "
            "multiple times, and the change in R squared compared to the "
            "baseline is recorded to quantify its contribution to model "
            "performance."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--splits-path", type=Path, default=Path("data/splits/scarce_0.00"),
        help="Path to the directory containing pickled NosoiSplit object."
    )
    parser.add_argument(
        "--model-path", type=Path, default=Path("data/dnn/scarce_0.00"),
        help=(
            "Path to the directory containing the pickled DNN (regressor.pt) "
            "and hyperparameter configuration (best_config.json)"
        )
    )
    parser.add_argument(
        "--repeats", type=int, default=10,
        help=(
            "Number of times each summary statistic is shuffled to average "
            "out randomness in R squared estimates."
        )
    )
    parser.add_argument(
        "--output-path", type=str, default="perturbation_analysis.pdf",
        help="Where to save the plot."
    )

    args = parser.parse_args()
    run_perturbation(
        args.splits_path,
        args.model_path,
        args.repeats,
        args.output_path
    )


if __name__ == "__main__":
    cli_main()
