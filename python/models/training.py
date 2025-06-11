from typing import Protocol, Tuple
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
import logging
from utils.logging_config import setup_logging


def get_logger():
    return logging.getLogger(__name__)


class TrainableModel(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def parameters(self):
        ...

    def train(self, mode: bool = True):
        ...

    def eval(self):
        ...

    def state_dict(self):
        ...

    def load_state_dict(self, state_dict):
        ...


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
    torch.nn.Module
        The trained model with the best weights loaded.
    dict[str, list[float]]
        A dictionary containing lists of average training and validation losses
        across epochs. Keys are 'avg_train_loss' and 'avg_val_loss'.

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
        "avg_train_loss": [],
        "avg_val_loss": [],
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

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item()
        avg_val_loss = val_loss / len(val_loader)

        # Log each epoch when debugging
        logger.debug(
            f"Epoch {epoch+1}: "
            f"Train Loss = {avg_train_loss:.4f}, "
            f"Val Loss = {avg_val_loss:.4f}"
        )

        # Record history
        history["avg_train_loss"].append(avg_train_loss)
        history["avg_val_loss"].append(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            trigger = 0
            best_model_state = model.state_dict()
        else:
            trigger += 1
            if trigger >= patience:
                logger.debug("Early stopping triggered.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history
