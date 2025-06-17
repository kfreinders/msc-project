#!/usr/bin/env python
from pathlib import Path
import csv
import json
import logging
import time
import torch
from typing import Sequence

from dataproc.nosoi_data_manger import NosoiDataProcessor
from dataproc.nosoi_split import NosoiSplit
from utils.logging_config import setup_logging
from models.tuning import (
    set_seed,
    model_factory,
    optuna_study,
    train_single_config
)


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
        "learning_rate": [1e-2, 1e-3, 3e-4, 1e-4],
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
        output_path = Path("data/tuning") / level
        metrics_path = model_dir / "metrics.json"

        if metrics_path.exists():
            logger.info(f"Skipping {level} — already completed.")
            continue

        start_time = time.time()

        logger.info(f"Generating data splits for {csv_file}")
        split_dir = NosoiDataProcessor.prepare_for_scarcity(
            csv_file, dir_master_csv)

        train_split = NosoiSplit.load("train", split_dir)
        val_split = NosoiSplit.load("val", split_dir)
        test_split = NosoiSplit.load("test", split_dir)

        json_path = output_path / "results.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)

        best_cfg, _ = optuna_study(
            train_split,
            val_split,
            device,
            n_trials=100,
            study_name=f"study_{level}",
            storage_path=output_path / "optuna_study.db"
        )

        logger.info(f"Best config for {level}: {best_cfg}")

        # Retrain model
        trained_model, _ = train_single_config(
            best_cfg,
            model_factory,
            train_split,
            val_split,
            device
        )

        # Evaluate on test set
        test_loader = test_split.make_dataloader(best_cfg.batch_size)
        trained_model.eval()
        criterion = torch.nn.MSELoss()
        test_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                test_loss += criterion(trained_model(X), y).item()
        test_loss /= len(test_loader)

        # Save test loss to metrics file
        with metrics_path.open("w") as f:
            json.dump({"test_loss": test_loss}, f, indent=4)

        logger.info(
            f"Finished {level} in {time.time() - start_time:.1f} seconds."
            f"Test loss: {test_loss:.4f}"
        )
        rows.append((level, f"{test_loss:.4f}"))

    # Save summary CSV
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with metrics_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
