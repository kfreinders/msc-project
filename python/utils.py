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

