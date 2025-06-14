from dataclasses import asdict, dataclass
from itertools import product
import json
import logging
from random import sample
from re import search
from typing import Callable, Dict, Iterable, Sequence

import numpy as np
from pathlib import Path
import torch
from torch import nn, optim

from utils.logging_config import setup_logging
from .model import NeuralNetwork
from .interfaces import TrainableModel
from .training import train
from dataproc.nosoi_split import NosoiSplit


@dataclass(frozen=True, slots=True)
class HyperParams:
    """Immutable, hashable bundle of hyper-parameters."""

    learning_rate: float
    hidden_size: int
    num_layers: int
    dropout_rate: float
    batch_size: int

    @classmethod
    def from_dict(cls, d: Dict[str, float | int]) -> "HyperParams":
        """
        Construct a `HyperParams` instance from a dictionary.

        This class method parses a dictionary containing hyperparameter
        values and returns a `HyperParams` object with properly cast types.

        Parameters
        ----------
        cls : type
            The `HyperParams` class itself.
        d : Dict[str, float | int]
            A dictionary containing keys for each hyperparameter
            (e.g., "learning_rate", "hidden_size", etc.).

        Returns
        -------
        HyperParams
            A new `HyperParams` instance initialized with values from the
            dictionary.
        """
        return cls(
            learning_rate=float(d["learning_rate"]),
            hidden_size=int(d["hidden_size"]),
            num_layers=int(d["num_layers"]),
            dropout_rate=float(d["dropout_rate"]),
            batch_size=int(d["batch_size"]),
        )

    def as_dict(self) -> Dict[str, float | int]:
        """
        Convert the `HyperParams` instance to a dictionary.

        This method returns the internal fields of the `HyperParams` object
        as a plain dictionary for logging or serialization.

        Returns
        -------
        Dict[str, float | int]
            A dictionary representation of the hyperparameter bundle.
        """
        return asdict(self)


def set_seed(seed: int = 42) -> None:
    """
    Make all libraries deterministic.

    Set a seed for numpy and torch libraries to make results reproducable.

    Parameters
    ----------
    seed : int
        Seed to use. Default is 42.
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make cuDNN only use deterministic convolution algorithms
    torch.backends.cudnn.deterministic = True

    # Disables cuDNN to automatically benchmark multiple convolution algorithms
    # and select the fastest one.
    torch.backends.cudnn.benchmark = False


def full_grid(
    space: Dict[str, Sequence[float | int]]
) -> Iterable[HyperParams]:
    """
    Generate all combinations of hyperparameter settings.

    This function creates a list of dictionaries, where each dictionary
    represents one unique combination of hyperparameter values from the
    provided search space.

    Parameters
    ----------
    params : Dict[str, Sequence[float | int]]
        Dictionary where keys are hyperparameter names and values are
        lists of possible values to try.

    Returns
    -------
    HyperParams
        Immutable, hashable bundle of hyperparameters.
    """
    keys, values = zip(*space.items())
    for combo in product(*values):
        yield HyperParams(**dict(zip(keys, combo)))


def random_search(
    search_space: Dict[str, Sequence[float | int]],
    k=150
) -> Iterable[HyperParams]:
    """
    Randomly sample k combinations from the hyperparameter search space.

    Parameters
    ----------
    params : Dict[str, Sequence[float | int]]
        Dictionary where keys are hyperparameter names and values are
        lists of possible values to try.

    Returns
    -------
    HyperParams
        Immutable, hashable bundle of hyperparameters.
    """
    subset = sample(list(full_grid(search_space)), k=k)
    for config in subset:
        yield config


def model_factory(
    input_dim: int,
    output_dim: int,
    cfg: HyperParams,
    device: torch.device
) -> TrainableModel:
    """
    Build a neural network model based on a given hyperparameter configuration.

    This function initializes a fully connected feedforward neural network
    (DNN) using the specified configuration and moves it to the target device.

    Parameters
    ----------
    config : HyperParams
        HyperParams class containing the model hyperparameters.
    device : torch.device
        The device (CPU or CUDA) on which to place the model.

    Returns
    -------
    TrainableModel
        The constructed neural network model.
    """
    return NeuralNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_layers,
        dropout_rate=cfg.dropout_rate,
    ).to(device)


def train_single_config(
    cfg: HyperParams,
    model_factory: Callable[
        [int, int, HyperParams, torch.device], TrainableModel
    ],
    train_split: NosoiSplit,
    val_split: NosoiSplit,
    device: torch.device,
    max_epochs: int = 100,
    patience: int = 5,
) -> float:
    """
    Train a model with a single hyperparameter configuration.

    Parameters
    ----------
    cfg : HyperParams
        The set of hyperparameters (learning rate, number of layers, etc.)
        used to configure the model.
    model_factory : Callable[[int, int, HyperParams, torch.device], TrainableModel]
        A function that constructs the model given the input/output dimensions,
        a hyperparameter configuration, and the target device.
    train_split : NosoiSplit
        The training dataset and associated metadata.
    val_split : NosoiSplit
        The validation dataset and associated metadata.
    device : torch.device
        Device on which the model is trained (CPU or CUDA).
    max_epochs : int, optional
        Maximum number of training epochs (default is 100).
    patience : int, optional
        Number of epochs with no improvement after which training is stopped
        early (default is 5).

    Returns
    -------
    float
        The lowest validation loss observed after training.
    """
    # Get input and output dimensions
    input_dim = train_split.input_dim
    output_dim = train_split.output_dim

    model = model_factory(input_dim, output_dim, cfg, device)
    optimiser = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()

    batch_size = cfg.batch_size
    train_loader = train_split.make_dataloader(batch_size)
    val_loader = val_split.make_dataloader(batch_size)

    model, hist = train(
        model, train_loader, val_loader, criterion, optimiser, device,
        epochs=max_epochs, patience=patience
    )
    return float(min(hist["val_loss"]))


def backup_json(path: Path) -> None:
    """
    Naïve versioning: rename *path* → *stem-1.json*, *stem-2.json*, ...

    Parameters
    ----------
    path
        Path to write the json file to.
    """
    if not path.exists():
        return
    i = 1
    while True:
        candidate = path.with_stem(f"{path.stem}-{i}")
        if not candidate.exists():
            path.rename(candidate)
            break
        i += 1


def tune_model(
    train_split: NosoiSplit,
    val_split: NosoiSplit,
    model_factory: Callable[
        [int, int, HyperParams, torch.device], TrainableModel
    ],
    search_space: Dict[str, Sequence[float | int]],
    device: torch.device,
    output_path: Path,
    max_epochs: int = 100,
    patience: int = 5,
) -> tuple[HyperParams, float]:
    """
    Perform a full grid search over all hyperparameter combinations and return
    the best one.

    Parameters
    ----------
    train_split : NosoiSplit
        Training data split.
    val_split : NosoiSplit
        Validation data split.
    model_factory : Callable
        Function that returns a model when given input/output dims,
        hyperparams, and device.
    search_space : Dict[str, Sequence[float | int]]
        Hyperparameter search space.
    device : torch.device
        Training device (CPU or CUDA).
    output_path : Path
        Path to save tuning results.
    max_epochs : int, optional
        Maximum number of epochs to train. Default is 100.
    patience : int, optional
        Early stopping patience. Default is 5.

    Returns
    -------
    tuple[HyperParams, float]
        Best-performing hyperparameter config and its validation loss.
    """
    setup_logging("tuning")
    logger = logging.getLogger(__name__)
    logger.info("Starting hyperparameter tuning...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    backup_json(output_path)

    results: list[tuple[Dict[str, float | int], float]] = []
    best_loss = float("inf")
    best_config = None

    combinations = list(full_grid(search_space))
    logger.info(f"Testing {len(combinations)} hyperparameter combinations.")

    # Loop over and test all hyperparameter combinations
    for i, cfg in enumerate(combinations, start=1):
        logger.info(f"Training configuration {i}/{len(combinations)}: {cfg}")
        val_loss = train_single_config(
            cfg, model_factory, train_split, val_split,
            device, max_epochs=max_epochs, patience=patience
        )
        logger.info(f"Validation loss: {val_loss:.4f}")

        results.append((cfg.as_dict(), val_loss))

        if val_loss < best_loss:
            best_loss = val_loss
            best_config = cfg

        # Save checkpoint after each config
        with open(output_path, "w") as f:
            json.dump(
                {
                    "results": results,
                    "best": best_config.as_dict() if best_config else None,
                    "loss": best_loss
                },
                f,
                indent=4
            )

    if best_config is None:
        raise RuntimeError(
            "No model configuration was successfully evaluated."
        )

    logger.info(f"Best config: {best_config} (loss: {best_loss:.4f})")
    return best_config, best_loss


def main() -> None:
    # Set up logger
    setup_logging("tuning")
    logger = logging.getLogger(__name__)

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set the seed
    seed = 42
    set_seed(seed)
    logger.info(f"Set hyperparameter tuning seed to: {seed}")

    # Define the hyperparameter search space
    search_space: dict[str, Sequence[int | float]] = {
        "learning_rate": [1e-2, 1e-3, 3e-4, 1e-4],
        "hidden_size": [16, 32, 64, 128, 256],
        "num_layers": [1, 2, 3, 4, 5],
        "dropout_rate": [0.1, 0.2, 0.3],
        "batch_size": [16, 32, 64, 128],
    }
    logger.info(f"Hyperparameter search space: {search_space}")

    # Load data splits from disk
    splits_path = Path("data/splits/scarce_0.05")
    train_split = NosoiSplit.load("train", splits_path)
    val_split = NosoiSplit.load("val", splits_path)
    logger.info(f"Loaded saved data splits from {splits_path}")

    output_path = Path("data/tuning")
    logger.info(f"Saving tuning results to: {output_path}")

    # Start tuning
    tune_model(
        train_split,
        val_split,
        model_factory,
        search_space,
        device,
        output_path
    )


if __name__ == "__main__":
    main()
