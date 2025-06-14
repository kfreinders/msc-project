from dataclasses import asdict, dataclass
from itertools import product
import json
import logging
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


def all_param_combinations(
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

    combinations = list(all_param_combinations(search_space))
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

    # Load data
    logger.info("Loading dataset...")
    dataset, *_ = load_data("data/nosoi/merged.csv")

    # Define the hyperparameter and architecture search space
    search_space: dict[str, list[float]] = {
        "learning_rate": [1e-2, 1e-3, 3e-4, 1e-4],
        "hidden_size": [16, 32, 64, 128, 256],
        "num_layers": [1, 2, 3, 4, 5],
        "dropout_rate": [0.1, 0.2, 0.3],
        "batch_size": [16, 32, 64, 128],
    }

    # Generate all possible combinations for a full grid search
    hyperparameter_combinations = all_param_combinations(search_space)
    logger.info(
        f"Total configurations to try: {len(hyperparameter_combinations)}"
    )

    # TODO: more intelligent resuming logic

    # Try to resume from existing results
    results = []
    try:
        with open("data/dnn/tuning_results.json", "r") as f:
            loaded = json.load(f)
            results = loaded.get("results", [])
            logger.info(
                f"Loaded {len(results)} previously saved configurations."
            )
    except FileNotFoundError:
        logger.info("No previous results found. Starting new tuning run.")

    # Build set of already tried configurations
    tried_configs = {json.dumps(r[0], sort_keys=True) for r in results}

    # FIXME: if there are .json files from a previous run, the script should
    # be aware that this is a new run. Currently, it might cause issues with
    # resuming logic.

    # Start full grid search
    for idx, config in enumerate(hyperparameter_combinations, start=1):
        # Skip this configuration if we've tried it before
        config_serialized = json.dumps(config, sort_keys=True)
        if config_serialized in tried_configs:
            logger.info(
                f"Skipping already tried configuration "
                f"{idx}/{len(hyperparameter_combinations)}"
            )
            continue

        logger.info(
            f"Training configuration {idx}/{len(hyperparameter_combinations)}"
        )
        logger.info(f"Configuration details: {config}")

        # Create dataloaders with specified batch size
        train, val = build_dataloaders(dataset, config["batch_size"])

        # Build model dynamically
        model = model_factory(config, device)

        # Train and pick best val loss during training
        final_val_loss = train_and_evaluate(
            model, config, train, val, device
        )

        # Save the config and its performance
        logger.info(
            f"Final validation loss for current config: {final_val_loss:.4f}"
        )
        results.append((config, final_val_loss))

        # Save a checkpoint after every configuration
        with open("data/dnn/tuning_results.json", "w") as f:
            json.dump({"results": results}, f, indent=4)

    # Sort by lowest validation loss
    results.sort(key=lambda x: x[1])
    best_config, best_loss = results[0]

    # Print best result
    logger.info(
        f"Best hyperparameters found: {best_config} "
        f"with validation loss {best_loss}"
    )

    # Save results
    with open("tuning_results.json", "w") as f:
        json.dump(
            {"best_config": best_config, "best_loss": best_loss}, f, indent=4
        )


if __name__ == "__main__":
    main()
