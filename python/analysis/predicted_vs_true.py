import logging
import numpy as np
from pathlib import Path
import torch

from models.tuning import HyperParams, model_factory
from dataproc.nosoi_split import NosoiSplit
from utils.logging_config import setup_logging
from utils.utils import plot_predictions, predict_nosoi_parameters


def main() -> None:
    # Set up logger
    setup_logging("training")
    logger = logging.getLogger(__name__)

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data splits from disk
    splits_path = Path("data/splits/scarce_0.00")
    train_split = NosoiSplit.load("train", splits_path)
    test_split = NosoiSplit.load("test", splits_path)
    logger.info(f"Loaded saved data splits from {splits_path}")

    cfg = HyperParams(
        learning_rate=0.0005186374528320235,
        hidden_size=256,
        num_layers=2,
        dropout_rate=0.10448580769582116,
        batch_size=16
    )
    logger.info(f"HyperParams: {cfg}")

    model = model_factory(
        train_split.input_dim,
        train_split.output_dim,
        cfg,
        device
    )

    # Load model (redundant in case since already in memory)
    model_path = Path("data/dnn/scarce_0.00/regressor.pt")
    logger.info(f"Loading saved model from: {model_path}")
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)

    logger.info(f"Making predictions...")
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
