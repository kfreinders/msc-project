import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import pandas as pd
import torch
from typing import Callable, Optional
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
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
    filter_fn: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
    filter_fn_desc: Optional[str] = None,
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
    filter_fn : Callable, optional
        A function that takes the merged DataFrame and returns a Boolean mask
        indicating which rows to keep.

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame containing both summary statistics and parameters.

    Examples
    --------
    Merge the master and summary statistics files into `merged.csv`, keeping
    only rows where the total number of hosts (SS_11) exceeds 2000.

    merge_summary_and_parameters(
        "data/nosoi/summary_stats_export.csv",
        "data/nosoi/master.csv",
        "data/nosoi/merged.csv",
        filter_fn=lambda df: df["SS_11"] > 2000,
        filter_fn_desc="SS_11 > 2000"
    )

    """
    # Load summary statistics and parameters
    summary_df = pd.read_csv(summary_stats_csv)
    master_df = pd.read_csv(master_csv)

    # Merge them on the 'seed' column
    merged_df = pd.merge(summary_df, master_df, on="seed", how="inner")

    # Optional filtering logic
    if filter_fn is not None:
        n_before = len(merged_df)
        mask = filter_fn(merged_df)
        merged_df = merged_df[mask]
        n_after = len(merged_df)
        logger.info(
            f"Filtered {n_before - n_after:,} rows"
        )

    # Print filter function description for logging purposes
    if filter_fn_desc is not None:
        logger.info(
            f"Row filtering condition: {filter_fn_desc}"
        )

    # Save the merged dataset
    merged_df.to_csv(output_csv, index=False)

    return merged_df


def load_data(
    csv_path: str,
    extract_columns: Optional[list[str]] = None
) -> tuple[TensorDataset, dict[str, np.ndarray]]:
    """
    Load and normalize the dataset from a CSV file.

    Optionally extract specific summary statistic columns (unnormalized)
    for later analysis (e.g., simulation length, transmission chain size).

    Parameters
    ----------
    csv_path : str
        Path to the merged CSV file containing summary statistics and
        parameters.
    extract_columns : list of str, optional
        List of summary statistic column names to extract separately
        before normalization.

    Returns
    -------
    dataset : torch.utils.data.TensorDataset
        Dataset containing normalized summary statistics as features
        and raw parameters as labels.
    extracted : dict of str → np.ndarray
        Dictionary mapping extracted column names to their unnormalized values.
        Empty if no columns were extracted.
    """
    df = pd.read_csv(csv_path)

    # Drop non-feature columns
    df = df.drop(columns=["seed"])

    # Identify summary statistic columns
    ss_cols = [col for col in df.columns if col.upper().startswith("SS_")]
    n_ss = len(ss_cols)

    # Input features and target parameters
    X = df.iloc[:, :n_ss].values  # Summary stats
    y = df.iloc[:, n_ss:].values  # Parameters

    # Extract unnormalized columns
    extracted = {}
    if extract_columns is not None:
        missing = [col for col in extract_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"Column(s) not found in dataset: {', '.join(missing)}"
            )
        # Build mapping of column name to unnormalized values
        extracted = {
            col: np.asarray(df[col].values.copy()) for col in extract_columns
        }

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Build PyTorch dataset
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)

    return dataset, extracted


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


def split_data_with_meta(
    dataset: TensorDataset,
    meta: dict[str, np.ndarray],
    ptrain: float,
    pval: float,
    ptest: float,
    batch_size: int = 64,
    seed: Optional[int] = None,
) -> tuple[
    DataLoader, DataLoader, DataLoader,
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    """
    Shuffle and split dataset and its associated metadata.

    Parameters
    ----------
    dataset : TensorDataset
        The full dataset of features and targets.
    meta : dict[str, np.ndarray]
        Dictionary of metadata arrays (e.g., unnormalized simulation lengths),
        with one value per sample.
    ptrain : float
        Proportion of samples used for training.
    pval : float
        Proportion used for validation.
    ptest : float
        Proportion used for testing.
    batch_size : int, optional
        Batch size for DataLoaders.
    seed : int, optional
        If provided, controls the random shuffling for reproducibility.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
        DataLoaders for each split.
    train_meta, val_meta, test_meta : dict[str, np.ndarray]
        Sliced metadata dictionaries corresponding to each split.
    """
    total = ptrain + pval + ptest
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Split proportions must sum to 1.0 (got {total:.4f})"
        )

    n = len(dataset)
    indices = np.arange(n)

    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n, generator=g).numpy()
    else:
        np.random.shuffle(indices)

    # Split indices
    train_size = round(ptrain * n)
    val_size = round(pval * n)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    # Split dataset
    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    test_ds = Subset(dataset, test_idx.tolist())

    # Split meta
    def slice_meta(idxs: np.ndarray) -> dict[str, np.ndarray]:
        return {k: v[idxs] for k, v in meta.items()}

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
        slice_meta(train_idx),
        slice_meta(val_idx),
        slice_meta(test_idx),
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
