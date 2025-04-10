import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import logging
from logging_config import setup_logging

# -----------------------------------------------------------------------------
#  Configure logging
# -----------------------------------------------------------------------------

# Set up logger
setup_logging()
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
#  Functions
# -----------------------------------------------------------------------------


def merge_summary_and_parameters(
    summary_stats_csv: str,
    master_csv: str,
    output_csv: str,
    drop_trivial: bool = False,
) -> pd.DataFrame:
    """
    Inner join summary statistics and parameters on the seed and save to a CSV.

    Parameters
    ----------
    summary_stats_csv : str
        Path to the CSV file containing summary statistics.
    master_csv : str
        Path to the CSV file containing nosoi simulation parameters.
    output_csv : str
        Path where the merged CSV file will be saved.
    drop_trivial : bool, optional
        Whether to preserve simulations with only one infected host (default is
        False).

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame containing both summary statistics and parameters.
    """
    # Load summary statistics and parameters
    summary_df = pd.read_csv(summary_stats_csv)
    master_df = pd.read_csv(master_csv)

    # Merge them on the 'seed' column
    merged_df = pd.merge(summary_df, master_df, on="seed", how="inner")

    # Optionally drop trivial simulations
    if drop_trivial:
        if "SS_11" not in merged_df.columns:
            raise ValueError(
                "SS_11 (total hosts) column not found in summary statistics."
            )
        n_before = len(merged_df)
        merged_df = merged_df[merged_df["SS_11"] > 1]
        n_after = len(merged_df)
        n_dropped = n_before - n_after
        logger.info(
            f"Dropped {n_dropped:,} trivial simulations "
            f"(only patient zero infected)."
        )

    # Save the merged dataset
    merged_df.to_csv(output_csv, index=False)

    return merged_df


def load_data(csv_path: str) -> torch.utils.data.TensorDataset:
    """
    Load and normalize the dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the merged CSV file containing summary statistics and
        parameters.

    Returns
    -------
    torch.utils.data.TensorDataset
        A dataset containing normalized summary statistics as features and
        parameters as labels.
    """
    df = pd.read_csv(csv_path)

    # Drop the 'seed' column
    df = df.drop(columns=["seed"])

    # Find where the summary statistics end
    ss_cols = [col for col in df.columns if col.startswith("SS_")]
    n_ss = len(ss_cols)

    X = df.iloc[:, :n_ss].values  # Summary stats
    y = df.iloc[:, n_ss:].values  # Parameters

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)

    return dataset


def split_data(
    dataset: torch.utils.data.TensorDataset,
    ptrain: float,
    pval: float,
    ptest: float,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split a dataset into training, validation, and testing sets.

    Parameters
    ----------
    dataset : torch.utils.data.TensorDataset
        The full dataset to split.
    ptrain : float
        Proportion of the dataset to use for training.
    pval : float
        Proportion of the dataset to use for validation.
    ptest : float
        Proportion of the dataset to use for testing.
    batch_size : int, optional
        Number of samples per batch (default is 64).

    Returns
    -------
    tuple of torch.utils.data.DataLoader
        DataLoaders for training, validation, and testing datasets.
    """
    total = ptrain + pval + ptest
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Split proportions must sum to 1.0 (got {total:.4f})"
        )

    # Ensure that the split sizes always exactly sum up
    n = len(dataset)
    train_size = round(ptrain * n)
    val_size = round(pval * n)
    test_size = n - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size]
    )

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
    )


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 100,
    patience: int = 5,
) -> tuple[torch.nn.Module, dict[str, list[float]]]:
    """
    Train a neural network model with early stopping.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to train.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    criterion : torch.nn.modules.loss._Loss
        Loss function used for training.
    optimizer : torch.optim.Optimizer
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
def predict_parameters(
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
    n_cols: int = 3,
    figsize: tuple[int, int] = (18, 12),
) -> None:
    """
    Plot predicted versus true values for each output parameter.

    Parameters
    ----------
    preds : numpy.ndarray
        Predicted values from the model.
    trues : numpy.ndarray
        True target values.
    param_names : list of str, optional
        Names of the output parameters. If None, parameters will be
        labeled by index.
    n_cols : int, optional
        Number of columns in the plot grid (default is 3).
    figsize : tuple of int, optional
        Figure size (default is (18, 12)).

    Returns
    -------
    None
    """
    # Calculate how many rows we need
    n_params = preds.shape[1]
    n_rows = (n_params + n_cols - 1) // n_cols

    # Construct each scatterplot
    plt.figure(figsize=figsize)
    for i in range(n_params):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.scatter(trues[:, i], preds[:, i], alpha=0.3)

        # Draw the diagonal
        min_val = min(trues[:, i].min(), preds[:, i].min())
        max_val = max(trues[:, i].max(), preds[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)

        # Set plot labels
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        if param_names:
            plt.title(param_names[i])
        else:
            plt.title(str(i))

    plt.tight_layout()
    plt.show()
