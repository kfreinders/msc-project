#!/usr/bin/env python
import argparse
from pathlib import Path
import json
import logging
from sklearn.metrics import r2_score
import torch

from torch.utils.data import DataLoader

from dataproc.nosoi_data_manager import NosoiDataProcessor
from dataproc.nosoi_split import NosoiSplit
from models.interfaces import TrainableModel
from utils.logging_config import setup_logging
from utils.utils import predict_nosoi_parameters, save_torch_with_versioning
from models.tuning import (
    HyperParams,
    set_seed,
    model_factory,
    optuna_study,
    train_single_config
)


def prepare_paths(name: str) -> tuple[Path, Path, Path, Path]:
    """
    Prepare the output directory and file paths for a given model run.

    This function ensures the output directory for a DNN run exists and
    constructs standardized paths for saving the trained model, the best
    hyperparameter configuration, and the Optuna study database.

    Parameters
    ----------
    name : str
        Unique name for the run, typically derived from the input file stem.

    Returns
    -------
    tuple[Path, Path, Path, Path]
        Paths for the run root directory, trained model, best config, and
        Optuna study database, respectively.
    """
    dnn_root = Path("data/dnn") / name
    dnn_root.mkdir(parents=True, exist_ok=True)
    return (
        dnn_root,
        dnn_root / "regressor.pt",
        dnn_root / "best_config.json",
        dnn_root / "optuna_study.db",
    )


def load_all_splits(path: Path):
    """
    Load all available Nosoi data splits (train, validation, test).

    This function reconstructs `NosoiSplit` objects from serialized files
    stored in the given directory.

    Parameters
    ----------
    path : Path
        Directory containing the saved split files.

    Returns
    -------
    tuple[NosoiSplit, NosoiSplit, NosoiSplit]
        The training, validation, and test splits.
    """
    train_split = NosoiSplit.load("train", path)
    val_split = NosoiSplit.load("val", path)
    test_split = NosoiSplit.load("test", path)
    return train_split, val_split, test_split


def load_search_space(
    path: Path = Path("search_space.json")
) -> dict[str, list[int | float]]:
    """
    Load the hyperparameter search space from a JSON file.

    Parameters
    ----------
    path : Path, optional
        Path to the JSON file defining the hyperparameter search space.
        Defaults to "search_space.json".

    Returns
    -------
    dict[str, list[int | float]]
        A dictionary where keys are hyperparameter names and values are lists
        of candidate values to explore during tuning.
    """
    with path.open("r") as f:
        return json.load(f)


def load_model(
    model_path: Path,
    input_dim: int,
    output_dim: int,
    best_cfg_path: Path,
    device: torch.device
):
    """
    Load a trained DNN model and its configuration from disk.

    The method restores the model architecture from a saved hyperparameter
    configuration and loads its trained weights. The model is then mapped onto
    the specified device.

    Parameters
    ----------
    model_path : Path
        Path to the file containing the trained model weights (.pt).
    input_dim : int
        Number of input features for the model.
    output_dim : int
        Number of output parameters predicted by the model.
    best_cfg_path : Path
        Path to the JSON file storing the best hyperparameter configuration.
    device : torch.device
        Device to load the model onto (CPU or CUDA).

    Returns
    -------
    tuple[TrainableModel, HyperParams]
        The loaded model and its associated hyperparameter configuration.
    """
    with best_cfg_path.open("r") as f:
        best_cfg = HyperParams.from_dict(json.load(f))

    trained_model = model_factory(
        input_dim,
        output_dim,
        best_cfg,
        device
    )
    trained_model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    return trained_model.to(device), best_cfg


def compute_mse_on_test(
    model: TrainableModel,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate the trained model on the test dataset using Mean Squared Error.

    This function computes the average MSE loss of the model predictions on the
    provided test dataset. It disables gradient tracking to speed up evaluation
    and reduce memory usage.

    Parameters
    ----------
    model : TrainableModel
        The trained PyTorch model used to generate predictions.
    test_loader : DataLoader
        DataLoader object providing the test dataset in mini-batches.
    device : torch.device
        The device (CPU or CUDA) on which computation should be performed.

    Returns
    -------
    float
        The average MSE loss over all batches in the test set.
    """
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            total_loss += criterion(model(X), y).item()
    return total_loss / len(test_loader)


def compute_r2_per_param(
    model: TrainableModel,
    test_split: NosoiSplit,
    test_loader: DataLoader,
    device: torch.device
) -> dict[str, float]:
    """
    Compute the coefficient of determination for each predicted parameter.

    This function evaluates the predictive performance of the model on the test
    dataset by calculating the R^2 score for each output dimension. It uses the
    true and predicted parameter values for each sample and returns a
    dictionary mapping each parameter name to its corresponding R^2 score.

    Parameters
    ----------
    model : TrainableModel
        The trained model to be evaluated.
    test_split : NosoiSplit
        The test split containing metadata such as output column names.
    test_loader : DataLoader
        DataLoader providing batched test data.
    device : torch.device
        The device (CPU or CUDA) to use.

    Returns
    -------
    dict[str, float]
        A dictionary where keys are parameter names and values are R² scores.
    """
    r2_values: dict[str, float] = {}
    preds, trues = predict_nosoi_parameters(
        model,
        test_loader,
        device
    )
    n_params = preds.shape[1]
    for i in range(n_params):
        parameter = (
            test_split.y_colnames[i]
            if test_split.y_colnames else str(i)
        )
        r2 = r2_score(trues[:, i], preds[:, i])
        r2_values[parameter] = float(r2)
    return r2_values


def evaluate_model_and_log(
    model: TrainableModel,
    test_split: NosoiSplit,
    best_cfg: HyperParams,
    device: torch.device
) -> None:
    """
    Evaluate a trained model on the test split and log its performance.

    This function computes the test loss (MSE) and R-squared values for each
    parameter, logging them for inspection. It uses the batch size defined in
    the best hyperparameter configuration.

    Parameters
    ----------
    model : TrainableModel
        The trained PyTorch model to be evaluated.
    test_split : NosoiSplit
        The test split containing both features and ground truth labels.
    best_cfg : HyperParams
        The best hyperparameter configuration obtained via tuning.
    device : torch.device
        Device used for evaluation.
    """
    test_loader = test_split.make_dataloader(best_cfg.batch_size)
    test_loss = compute_mse_on_test(model, test_loader, device)
    r2_values = compute_r2_per_param(model, test_split, test_loader, device)
    logging.info(f"Test loss: {test_loss:.4f}")
    logging.info(f"R-squared values: {r2_values}")


def main_pipeline(
    csv_file: Path,
    csv_master: Path,
    splits_path: Path,
    hparamspace_path: Path,
    n_trials: int,
    max_epochs,
    seed: int
) -> None:
    """
    Train or load a DNN to infer nosoi parameters from summary statistics.

    The pipeline checks whether a trained model and configuration already exist
    for the given scarcity level. If so, they are loaded and evaluated.
    Otherwise, the pipeline runs hyperparameter optimization with Optuna,
    retrains the model using the best configuration, saves the results, and
    evaluates its predictive accuracy.

    Parameters
    ----------
    csv_file : Path
        Path to the CSV file containing summary statistics at a given scarcity
        level.
    csv_master : Path
        Path to the CSV file containing the ground truth nosoi parameters.
    splits_path : Path
        Path to the directory containing precomputed NosoiSplit objects.
    hparamspace_path : Path
        Path to the JSON file specifying the hyperparameter search space.
    n_trials : int
        Number of Optuna trials to run for hyperparameter optimization.
    max_epochs : int
        Maximum number of epochs to train the DNN for.
    seed : int
        Random seed for reproducibility.
    """
    # Set up logging
    setup_logging("train-on-scarcity")
    logger = logging.getLogger(__name__)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    search_space = load_search_space(hparamspace_path)
    logger.info(f"Hyperparameter search space: {search_space}")

    name = csv_file.stem
    dnn_root, model_path, best_config_path, study_path = prepare_paths(name)

    logger.info(f"Generating data splits for {csv_file} ...")

    if splits_path.exists():
        logger.info(f"Using existing splits from {splits_path}")
    else:
        logger.info(f"Generating data splits for {csv_file} ...")
        splits_path = NosoiDataProcessor.prepare_for_scarcity(csv_file, csv_master)
    train_split, val_split, test_split = load_all_splits(splits_path)

    (dnn_root / "results.json").parent.mkdir(parents=True, exist_ok=True)

    if best_config_path.exists() and model_path.exists():
        logger.info(
            f"Found existing model and config for {name}, loading "
            "instead of training."
        )
        trained_model, best_cfg = load_model(
            model_path,
            train_split.input_dim,
            train_split.output_dim,
            best_config_path,
            device
        )
        evaluate_model_and_log(trained_model, test_split, best_cfg, device)
        return

    logger.info(
        f"No existing model found for {name}, starting training..."
    )

    best_cfg, _ = optuna_study(
        train_split,
        val_split,
        device,
        max_epochs=max_epochs,
        n_trials=n_trials,
        study_name=f"study_{name}",
        storage_path=study_path,
        search_space=search_space
    )

    logger.info(f"Best config for {name}: {best_cfg}")

    # Save best config to JSON
    with best_config_path.open("w") as f:
        json.dump(best_cfg.as_dict(), f, indent=4)
    logger.info(
        f"Exported best hyperparameter config to {best_config_path}"
    )

    # Retrain model
    logger.info(
        "Now training model with best hyperparameters. "
        "This may take a while..."
    )
    trained_model, _ = train_single_config(
        best_cfg,
        model_factory,
        train_split,
        val_split,
        device
    )

    save_torch_with_versioning(
        trained_model, model_path
    )
    logger.info(f"Saved model to {model_path}")

    evaluate_model_and_log(trained_model, test_split, best_cfg, device)


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a DNN on a scarcified data set."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--scarce-data-path",
        type=Path,
        default=Path("data/scarce_stats/scarce_0.00.csv"),
        help="Path to csv output from create_scarce_data.py."
    )
    parser.add_argument(
        "--master-path",
        type=Path,
        default=Path("data/nosoi/master.csv"),
        help=("Path to file with all original nosoi simulation parameters.")
    )
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=Path("data/splits/scarce_0.00"),
        help="Path to the directory containing pickled NosoiSplit object."
    )
    parser.add_argument(
        "--hparamspace-path",
        type=Path,
        default=Path("python/pipelines/hparamspace.json"),
        help="Path to the hyperparameter search space definition."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials to run for hyperparameter optimization."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum number of epochs to train the DNN for."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()
    main_pipeline(
        csv_file=args.scarce_data_path,
        csv_master=args.master_path,
        splits_path=args.splits_path,
        hparamspace_path=args.hparamspace_path,
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        seed=args.seed
    )


if __name__ == "__main__":
    cli_main()
