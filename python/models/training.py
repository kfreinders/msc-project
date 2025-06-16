import copy
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
import logging
from utils.logging_config import setup_logging

from .interfaces import TrainableModel


def get_logger():
    return logging.getLogger(__name__)


# TODO: implement patience tuning: optimize tradeoff between loss gain and
# waiting longer because of higher patience settings.
def train(
    model: TrainableModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: _Loss,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int = 100,
    patience: int = 5
) -> Tuple[TrainableModel, dict[str, list[float]]]:
    """
    Train a neural network model with early stopping.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to train.
    train_loader : DataLoader
        DataLoader for the training dataset.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    criterion : torch.nn.modules.loss._Loss
        Loss function used for training.
    optimizer : Optimizer
        Optimizer used to update model weights.
    device : torch.device
        Device on which the model is trained (CPU or CUDA).
    epochs : int, optional
        Maximum number of training epochs (default is 100).
    patience : int, optional
        Number of epochs with no improvement after which training is stopped
        early (default is 5).

    Returns
    -------
    TrainableModel
        The trained model with the best weights loaded (based on validation
        loss).
    dict[str, list[float]]
        Dictionary with training and validation loss curves per epoch.
        Keys are 'train_loss' and 'val_loss'.

    Notes
    -----
    The best model (lowest validation loss) is tracked and loaded at the end
    of training. Early stopping halts training if the validation loss does
    not improve for `patience` consecutive epochs.
    """
    # Set up logger
    setup_logging("training")
    logger = logging.getLogger(__name__)

    best_loss = float("inf")
    best_model_state = None
    trigger = 0

    # Keep track of training history
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item()
        val_loss = val_loss / len(val_loader)

        if torch.isnan(torch.tensor(val_loss)):
            logger.warning(
                "NaN encountered in validation loss. Stopping early."
            )
            break

        # Log each epoch when debugging
        logger.debug(
            f"Epoch {epoch+1}: "
            f"Train Loss = {train_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}"
        )

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            trigger = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            trigger += 1
            if trigger >= patience:
                logger.debug("Early stopping triggered.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history
