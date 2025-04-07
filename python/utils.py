import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler


def merge_summary_and_parameters(
    summary_stats_csv: str, master_csv: str, output_csv: str
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

    train_size = int(ptrain * len(dataset))
    val_size = int(pval * len(dataset))
    test_size = int(ptest * len(dataset))
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
) -> torch.nn.Module:
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
        Trained model with the best weights loaded.
    """
    best_loss = float("inf")
    best_model_state = None
    trigger = 0

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

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss = {avg_train_loss:.4f}, "
            f"Val Loss = {avg_val_loss:.4f}"
        )

        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            trigger = 0
            best_model_state = model.state_dict()
        else:
            trigger += 1
            if trigger >= patience:
                print("Early stopping triggered.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


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
    print(f"Test Loss: {avg_test_loss:.4f}")
    return avg_test_loss
