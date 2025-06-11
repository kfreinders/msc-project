import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import os
import torch
from typing import Optional
import logging

# -----------------------------------------------------------------------------
#  Configure logging
# -----------------------------------------------------------------------------

# Set up logger
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
#  Functions
# -----------------------------------------------------------------------------


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    device: torch.device,
) -> float:
    """
    Evaluate a trained model on a test set.

    Parameters
    ----------
    model : torch.nn.Module
        The trained neural network model.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    criterion : torch.nn.modules.loss._Loss
        Loss function used for evaluation.
    device : torch.device
        Device on which the model is evaluated (CPU or CUDA).

    Returns
    -------
    float
        Average loss on the test dataset.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            total_loss += criterion(model(X), y).item()

    avg_test_loss = total_loss / len(test_loader)
    return avg_test_loss


# Disable gradient calculation (important for inference only, saves memory)
@torch.no_grad()
def predict_nosoi_parameters(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict outputs for a dataset using a trained model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained neural network model.
    loader : torch.utils.data.DataLoader
        DataLoader providing input features and true labels.
    device : torch.device
        Device on which prediction is performed.

    Returns
    -------
    tuple of numpy.ndarray
        Tuple containing predicted values and true labels.
    """
    # Set the model to evaluation mode (disables i.a. dropout)
    model.eval()
    preds_list = []
    trues_list = []

    # Loop over batches from the loader
    for X_batch, y_batch in loader:
        # Move both features and labels to the device (GPU or CPU)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Make predictions
        y_pred = model(X_batch)

        # Move predictions and true labels back to CPU
        preds_list.append(y_pred.cpu())
        trues_list.append(y_batch.cpu())

    # Convert to a single numpy arrays
    preds_np = torch.cat(preds_list, dim=0).numpy()
    trues_np = torch.cat(trues_list, dim=0).numpy()

    return preds_np, trues_np


def plot_predictions(
    preds: np.ndarray,
    trues: np.ndarray,
    param_names: Optional[list[str]] = None,
    color_by: Optional[np.ndarray] = None,
    color_label: str = "Color scale",
    n_cols: int = 3,
    figsize: tuple[int, int] = (18, 12),
) -> matplotlib.figure.Figure:
    """
    Plot predicted versus true values for each output parameter.

    Parameters
    ----------
    preds : np.ndarray
        Predicted values from the model (N samples × M parameters).
    trues : np.ndarray
        True target values.
    param_names : list of str, optional
        Names of the output parameters. If None, parameters are labeled by
        index.
    color_by : np.ndarray, optional
        Optional variable to color points by (e.g., simulation size or
        duration). Must be the same length as `preds` and `trues`.
    color_label : str, optional
        Label to show next to the colorbar. Default is "Color scale".
    n_cols : int
        Number of columns in the subplot grid.
    figsize : tuple[int, int]
        Overall size of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure with a grid of all the predicted versus true value
        plots.
    """
    n_params = preds.shape[1]
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, constrained_layout=True
    )
    axes = axes.flatten()

    scatter = None  # Store for colorbar

    for i in range(n_params):
        ax = axes[i]

        if color_by is not None:
            scatter = ax.scatter(
                trues[:, i], preds[:, i],
                c=color_by, cmap="viridis", alpha=0.5
            )
        else:
            ax.scatter(trues[:, i], preds[:, i], alpha=0.3)

        min_val = min(trues[:, i].min(), preds[:, i].min())
        max_val = max(trues[:, i].max(), preds[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)

        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(param_names[i] if param_names else f"Parameter {i}")

    # Hide unused axes
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    # Shared colorbar
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes.tolist(), shrink=0.95)
        cbar.set_label(color_label)

    return fig


def save_torch_with_versioning(
    model: torch.nn.Module,
    path: str,
    logger: logging.Logger
) -> None:
    """
    Save a PyTorch model to `path`, renaming any existing file by appending a
    version number.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to save.
    path : str
        The full path to save the model to (e.g., '../data/dnn/regressor.pt').
    logger : logging.Logger
        A configured logger to report actions.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # If the file exists, move it to a versioned filename
    if os.path.exists(path):
        base, ext = os.path.splitext(path)
        i = 1
        while True:
            backup_path = f"{base}-{i}{ext}"
            if not os.path.exists(backup_path):
                os.rename(path, backup_path)
                logger.info(f"Existing model renamed to '{backup_path}'.")
                break
            i += 1

    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to '{path}'.")
