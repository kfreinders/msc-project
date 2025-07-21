import logging
import numpy as np
from pathlib import Path
import torch
import json

from models.tuning import HyperParams, model_factory
from dataproc.nosoi_split import NosoiSplit
from utils.logging_config import setup_logging
from utils.utils import plot_predictions, predict_nosoi_parameters


def main() -> None:
    # Set up logger
    setup_logging("predicted_vs_true")
    logger = logging.getLogger(__name__)

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set root path
    root_path = Path("data/splits/scarce_0.00")

    test_split = NosoiSplit.load("test", root_path)
    logger.info(f"Loaded saved data splits from {root_path}")

    # Retrieve hyperparameters
    with open(root_path / "best_config.json") as handle:
        data = json.load(handle)
        cfg = HyperParams.from_dict(data)

    logger.info(f"HyperParams: {cfg}")

    # Set up model architecture
    model = model_factory(
        test_split.input_dim,
        test_split.output_dim,
        cfg,
        device
    )

    # Load model (redundant in case since already in memory)
    model_path = root_path / "regressor.pt"
    logger.info(f"Loading saved model from: {model_path}")
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
    fig.savefig("predicted_vs_true.png", dpi=400, bbox_inches="tight")
    logger.info("Done")


if __name__ == "__main__":
    main()
