"""
tuning.py
---------

This module provides a complete framework for hyperparameter tuning of DNN
models trained on epidemiological summary statistics.

Core functionality includes:
- Definition of immutable `HyperParams` dataclass to manage model
  configurations
- Grid and random search methods to generate hyperparameter combinations
- A factory interface (`model_factory`) to construct models from
  hyperparameters
- Utilities for deterministic behaviour
- Functions to train models (`train_single_config`) and evaluate them
- A `tune_model` routine that tracks and saves performance metrics and best
  configurations

The module is designed to work with models implementing the `TrainableModel`
interface and data formatted as `NosoiSplit` objects. It supports both full
grid and random search strategies, and outputs results in a versioned JSON
format.

Typical usage:
>>> best_cfg, best_loss = tune_model(
>>>     train_split, val_split, model_factory,
>>>     search_space, random_search, device, output_path
>>> )

All training includes early stopping and performance tracking for each
configuration.
"""

from dataclasses import asdict, dataclass
from itertools import product
import json
import logging
import optuna
from random import sample
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


def full_grid_search(
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
    subset = sample(list(full_grid_search(search_space)), k=k)
    for config in subset:
        yield config


def optuna_objective(
    trial: optuna.Trial,
    train_split: NosoiSplit,
    val_split: NosoiSplit,
    device: torch.device,
    max_epochs: int,
    patience: int,
) -> float:
    """
    Objective function for Optuna hyperparameter optimization.

    This function defines the hyperparameter search space, constructs a model
    configuration using suggested values from the trial, trains the model on
    the provided training split, and returns the validation loss as the
    optimization objective.

    Parameters
    ----------
    trial : optuna.Trial
        The Optuna trial object used to suggest hyperparameter values.
    train_split : NosoiSplit
        Training data split used for model training.
    val_split : NosoiSplit
        Validation data split used to evaluate the model during tuning.
    device : torch.device
        Device on which to train the model (e.g., "cpu" or "cuda").
    max_epochs : int
        Maximum number of training epochs.
    patience : int
        Number of epochs to wait for improvement before early stopping.

    Returns
    -------
    float
        Validation loss of the trained model for the given hyperparameter
        configuration.
    """
    cfg = HyperParams(
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        hidden_size=trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256]),
        num_layers=trial.suggest_int("num_layers", 1, 5),
        dropout_rate=trial.suggest_float("dropout_rate", 0.1, 0.3),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
    )

    _, val_loss = train_single_config(
        cfg,
        model_factory,
        train_split,
        val_split,
        device,
        max_epochs,
        patience
    )
    return val_loss


def optuna_study(
    train_split: NosoiSplit,
    val_split: NosoiSplit,
    device: torch.device,
    max_epochs: int = 30,
    patience: int = 2,
    n_trials: int = 50,
    study_name: str = "nosoi_hyperparameter_tuning",
    storage_path: Path = Path("optuna_studies/nosoi_study.db")
) -> tuple[HyperParams, float]:
    """
    Run Optuna study to find best hyperparameters.

    Parameters
    ----------
    train_split : NosoiSplit
        Training data.
    val_split : NosoiSplit
        Validation data.
    device : torch.device
        Device for training.
    max_epochs : int, optional
        Maximum number of training epochs (default is 30).
    patience : int, optional
        Number of epochs with no improvement after which training is stopped
        early (default is 5).
    n_trials : int
        Number of Optuna trials.
    study_name : str
        Name of the Optuna study.
    storage_path : str
        Path to SQLite DB file for saving the study persistently.

    Returns
    -------
    tuple[HyperParams, float]
        Best hyperparameters and validation loss.
    """
    # Ensure directory exists and use SQLite storage
    Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: optuna_objective(
            trial,
            train_split,
            val_split,
            device,
            max_epochs,
            patience
        ),
        n_trials=n_trials
    )

    best_cfg_dict = study.best_trial.params
    best_cfg = HyperParams.from_dict(best_cfg_dict)
    best_loss = study.best_value

    return best_cfg, best_loss


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
) -> tuple[TrainableModel, float]:
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
    tuple[TrainableModel, float]
        The trained NeuralNetwork and according lowest validation loss observed
        after training.
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
    return model, float(min(hist["val_loss"]))


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


# TODO: return best model also for ease of use.
def tune_model(
    train_split: NosoiSplit,
    val_split: NosoiSplit,
    model_factory: Callable[
        [int, int, HyperParams, torch.device], TrainableModel
    ],
    search_space: Dict[str, Sequence[float | int]],
    search_method: Callable[[Dict[str, Sequence[float | int]]], Iterable[HyperParams]],
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
    search_method : Callable[[Dict[str, Sequence[float | int]]], Iterable[HyperParams]]
        Which search method to explore the hyperparameter space. Can be
        full_grid_search or random_search.
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

    logger.info(f"Using search method: {search_method.__name__}")
    combinations = list(search_method(search_space))
    logger.info(f"Testing {len(combinations)} hyperparameter combinations.")

    # Loop over and test all hyperparameter combinations
    for i, cfg in enumerate(combinations, start=1):
        logger.info(f"Training configuration {i}/{len(combinations)}: {cfg}")
        _, val_loss = train_single_config(
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
