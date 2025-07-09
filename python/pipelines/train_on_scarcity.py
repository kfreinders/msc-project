#!/usr/bin/env python
from pathlib import Path
import csv
import json
import logging
from sklearn.metrics import r2_score
import time
import torch
from typing import Sequence

from torch.utils.data import DataLoader

from dataproc.nosoi_data_manger import NosoiDataProcessor
from dataproc.nosoi_split import NosoiSplit
from models.interfaces import TrainableModel
from utils.logging_config import setup_logging
from utils.utils import predict_nosoi_parameters, save_torch_with_versioning
from models.tuning import (
    set_seed,
    model_factory,
    optuna_study,
    train_single_config
)


def evaluate_model(
    model: TrainableModel,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
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
    r2_values: dict[str, float] = {}
    preds, trues = predict_nosoi_parameters(
        model,
        test_loader,
        device
    )
    n_params = preds.shape[1]
    for i in range(n_params):
        parameter = test_split.y_colnames[i] if test_split.y_colnames else str(i)
        r2 = r2_score(trues[:, i], preds[:, i])
        r2_values[parameter] = float(r2)
    return r2_values


def main() -> None:
    # Set up logging
    setup_logging("train-on-scarcity")
    logger = logging.getLogger(__name__)

    rows = [("scarcity", "test_loss")]

    dir_master_csv = Path("data/nosoi/master.csv")
    dir_scarce_csv = Path("data/scarce_stats")
    metrics_csv = Path("data/metrics/scarcity_performance.csv")
    all_csv_files = sorted(dir_scarce_csv.glob("scarce_*.csv"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    set_seed(seed)

    logger.info(
        f"Found {len(all_csv_files)} summary statistic files for degraded "
        f"graphs in: {dir_scarce_csv}"
    )
    logger.info(f"Reading nosoi parameters from: {dir_master_csv}")
    logger.info(f"Saving output metrics to: {metrics_csv}")
    logger.info(f"Set run seed to {seed}")
    logger.info(f"Using device: {device}")

    search_space: dict[str, Sequence[int | float]] = {
        "learning_rate": [1e-2, 1e-3, 1e-4, 1e-5],
        "hidden_size": [16, 32, 64, 128, 256],
        "num_layers": [1, 2, 3, 4, 5],
        "dropout_rate": [0.1, 0.2, 0.3],
        "batch_size": [16, 32, 64, 128],
    }
    logger.info(f"Hyperparameter search space: {search_space}")

    for csv_file in all_csv_files:
        level = csv_file.stem
        model_dir = Path("data/dnn") / level
        model_dir.mkdir(parents=True, exist_ok=True)
        root_path = Path("data/dnn") / level
        metrics_path = model_dir / "metrics.json"

        if metrics_path.exists():
            logger.info(f"Skipping {level}: already completed.")
            continue

        start_time = time.time()

        logger.info(f"Generating data splits for {csv_file} ...")
        split_dir = NosoiDataProcessor.prepare_for_scarcity(
            csv_file, dir_master_csv
        )

        train_split = NosoiSplit.load("train", split_dir)
        val_split = NosoiSplit.load("val", split_dir)
        test_split = NosoiSplit.load("test", split_dir)

        json_path = root_path / "results.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)

        best_cfg, _ = optuna_study(
            train_split,
            val_split,
            device,
            n_trials=50,
            study_name=f"study_{level}",
            storage_path=root_path / "optuna_study.db"
        )

        logger.info(f"Best config for {level}: {best_cfg}")

        # Save best config to JSON
        best_config_path = model_dir / "best_config.json"
        with best_config_path.open("w") as f:
            json.dump(best_cfg.as_dict(), f, indent=4)

        logger.info(
            f"Exported best hyperparameter config to {best_config_path}"
        )

        # Retrain model
        trained_model, _ = train_single_config(
            best_cfg,
            model_factory,
            train_split,
            val_split,
            device
        )

        # Save the trained model if it doesn't exist yet.
        model_path = root_path / "regressor.pt"
        if model_path.is_file():
            logger.info(f"{model_path} already exists: not overwriting")
        else:
            torch.save(trained_model.state_dict(), model_path)
            save_torch_with_versioning(
                trained_model, model_path
            )

        # Make the test set dataloader
        test_loader = test_split.make_dataloader(best_cfg.batch_size)

        # Evaluate model on test set
        test_loss = evaluate_model(
            trained_model,
            test_loader,
            device,
        )

        # Save test loss to metrics file
        with metrics_path.open("w") as f:
            json.dump({"test_loss": test_loss}, f, indent=4)

        logger.info(
            f"Finished {level} in {time.time() - start_time:.1f} seconds."
            f"Test loss: {test_loss:.4f}"
        )
        rows.append((level, f"{test_loss:.4f}"))

        # Get R-squared values for each parameter
        r2_values = compute_r2_per_param(
            trained_model,
            test_split,
            test_loader,
            device
        )
        logger.info(r2_values)

    # Save summary CSV
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with metrics_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
