import argparse
import logging
import numpy as np
from pathlib import Path
import torch
import json

from models.tuning import HyperParams, model_factory
from dataproc.nosoi_split import NosoiSplit
from utils.logging_config import setup_logging
from utils.utils import plot_predictions, predict_nosoi_parameters


def compute_predicted_vs_true(
    splits_path: Path,
    model_path: Path,
    output_path: Path
) -> None:
    # Set up logger
    setup_logging("predicted_vs_true")
    logger = logging.getLogger(__name__)

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    test_split = NosoiSplit.load("test", splits_path)

    # Retrieve hyperparameters
    with open(model_path / "best_config.json") as handle:
        data = json.load(handle)
        cfg = HyperParams.from_dict(data)

    # Set up model architecture
    model = model_factory(
        test_split.input_dim,
        test_split.output_dim,
        cfg,
        device
    )

    # Load model (redundant in case since already in memory)
    model_path = model_path / "regressor.pt"
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)

    logger.info("Making predictions...")
    test_loader = test_split.make_dataloader(cfg.batch_size)
    preds, trues = predict_nosoi_parameters(model, test_loader, device)

    # Undo log transform for p_fatal (index 3)
    i_p_fatal = test_split.y_colnames.index("p_fatal")
    trues[:, i_p_fatal] = np.exp(trues[:, i_p_fatal])
    preds[:, i_p_fatal] = np.exp(preds[:, i_p_fatal])

    logger.info("Generating predicted vs true plot...")
    n_hosts = test_split.get_raw_feature("SST_06")
    fig = plot_predictions(preds, trues, test_split.y_colnames, n_hosts)
    fig.savefig(str(output_path), dpi=400, bbox_inches="tight")
    logger.info("Done")


def cli_main():
    parser = argparse.ArgumentParser(
        description="Plot predicted versus true values for a trained DNN.",
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
        "--output-path", type=str, default="./predicted_vs_true.png",
        help="Where to save the plot."
    )

    args = parser.parse_args()
    compute_predicted_vs_true(
        args.splits_path,
        args.model_path,
        args.output_path
    )


if __name__ == "__main__":
    cli_main()
