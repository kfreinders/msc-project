import os
import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from logging_config import setup_logging


# Set up logger
logger = logging.getLogger(__name__)


@dataclass
class NosoiSplit:
    X: torch.Tensor
    y: torch.Tensor
    x_raw: np.ndarray
    y_raw: np.ndarray

    def input_dim(self) -> int:
        return self.X.shape[1]

    def output_dim(self) -> int:
        return self.y.shape[1]

    def make_dataloader(
            self,
            batch_size: int = 32,
            shuffle: bool = True
    ) -> DataLoader:
        dataset = TensorDataset(self.X, self.y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def save(self, name: str, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.X, os.path.join(output_dir, f"{name}_x.pt"))
        torch.save(self.y, os.path.join(output_dir, f"{name}_y.pt"))
        np.savez(
            os.path.join(output_dir, f"{name}_raw.npz"),
            x_raw=self.x_raw,
            y_raw=self.y_raw
        )

    @classmethod
    def load(
        cls,
        name: str,
        input_dir: str,
        device: Optional[torch.device] = None
    ) -> "NosoiSplit":
        if device is None:
            device = torch.device("cpu")

        X = torch.load(
            os.path.join(input_dir, f"{name}_x.pt"), map_location=device
        )
        y = torch.load(
            os.path.join(input_dir, f"{name}_y.pt"), map_location=device
        )
        raw = np.load(os.path.join(input_dir, f"{name}_raw.npz"))
        return cls(X, y, raw["x_raw"], raw["y_raw"])


class NosoiDataProcessor:
    """
    Manages loading, merging, filtering, and transforming data for nosoi
    simulations.
    """
    def __init__(
        self,
        summary_stats_csv: str,
        master_csv: str,
    ) -> None:
        self.summary_stats_csv = summary_stats_csv
        self.master_csv = master_csv

        # Core dataframe
        self.df: pd.DataFrame

        # Keep track of which column are transformed and how
        self.transformed_cols: dict[str, Callable] = {}

        # Automatically load and merge data
        self._join_master_and_summary()

    @property
    def x(self) -> np.ndarray:
        """
        Return the input features (SST_* columns) as a NumPy array.
        """
        self._assert_data_loaded()
        sst_cols = [col for col in self.df.columns if col.startswith("SST_")]
        return self.df[sst_cols].to_numpy()

    @property
    def x_raw(self) -> np.ndarray:
        """Return raw input features (RAW_SST_ columns) as a NumPy Array."""
        self._assert_data_loaded()
        raw_cols = [
            col for col in self.df.columns if col.startswith("RAW_SST_")
        ]
        return self.df[raw_cols].to_numpy()

    @property
    def y(self) -> np.ndarray:
        """
        Return the target variables (PAR_* columns) as a NumPy array.
        """
        self._assert_data_loaded()
        par_cols = [col for col in self.df.columns if col.startswith("PAR_")]
        return self.df[par_cols].to_numpy()

    @property
    def y_raw(self) -> np.ndarray:
        """Return raw target parameters (RAW_PAR_ columns) as a NumPy Array."""
        self._assert_data_loaded()
        raw_cols = [
            col for col in self.df.columns if col.startswith("RAW_PAR_")
        ]
        return self.df[raw_cols].to_numpy()

    @property
    def input_dim(self) -> int:
        """
        Return the number of input features (columns starting with 'SST_').
        """
        return self.x.shape[1]

    @property
    def output_dim(self) -> int:
        """
        Return the number of target parameters (columns starting with 'PAR_').
        """
        return self.y.shape[1]

    # TODO: add docstring explanation of the protected prefixes and raw copy
    def _join_master_and_summary(self) -> None:
        """
        Load and inner-join summary statistics and parameter files on 'seed'.

        This function populates self.df with the joined dataset, containing
        both summary statistics (e.g., SS_*) and simulation parameters from
        nosoi.
        """
        # Load summary statistics and parameters
        summary_df = pd.read_csv(self.summary_stats_csv)
        master_df = pd.read_csv(self.master_csv)

        # Inner join the dfs on the simulation seed
        merged_df = pd.merge(summary_df, master_df, on="seed", how="inner")

        # Make a copy of all the raw data
        for col in merged_df.columns:
            if "RAW_" in col:
                raise ValueError(
                    f"Column '{col}' uses the reserved prefix 'RAW_'. This "
                    "prefix is reserved for internal use to preserve raw "
                    "data. Please rename it in the source file."
                )
            # Don't copy the seed column as it's never going to be transformed
            if col == "seed":
                continue

            merged_df[f"RAW_{col}"] = merged_df[col]

        self.df = merged_df

    def _assert_data_loaded(self) -> None:
        """
        Verify that the main DataFrame is loaded and not empty.

        This internal helper method is used to ensure that `self.df` has been
        populated before proceeding with any operations that depend on it.

        Raises
        ------
        RuntimeError
            If `self.df` has not been initialized.
        RuntimeError
            If `self.df` exists but contains no rows.
        """
        if self.df is None:
            raise RuntimeError("DataFrame is not loaded.")

        if self.df.empty:
            raise RuntimeError("DataFrame is empty.")

    def drop_by_filter(
        self,
        filter_fn: Callable[[pd.DataFrame], pd.Series],
        filter_fn_desc: Optional[str] = None,
    ) -> None:
        """
        Drop values from the merged dataset using a row-wise Boolean mask.

        Parameters
        ----------
        filter_fn : Callable[[pd.DataFrame], pd.Series]
            A function that takes the full merged DataFrame and returns a
            Boolean mask of rows to retain (e.g., `df["PAR_SS_11"] > 2000`).

        filter_fn_desc : str, optional
            A description of the filtering condition for logging purposes.
            Useful for tracking pipeline behavior.
        """
        self._assert_data_loaded()

        n_before = len(self.df)
        mask = filter_fn(self.df)
        self.df = self.df[mask]
        n_after = len(self.df)
        logger.info(
            f"Dropped {n_before - n_after:,} rows out of {n_before:,} based "
            f"on filter"
        )

        # Print filter function description for logging purposes
        if filter_fn_desc is not None:
            logger.info(f"Row filtering condition: {filter_fn_desc}")

    def apply_infectivity(self) -> None:
        """
        Replace 'mean_nContact' and 'p_trans' with their product: infectivity.

        This method computes a new column 'infectivity' as the product of
        mean_nContact and p_trans, drops the originals, and reorders the
        resulting columns for consistency.

        Raises
        ------
        ValueError
            If 'infectivity' already exists, indicating it has already
            been applied.
        ValueError
            If either 'mean_nContact' or 'p_trans' is missing from the dataset.
        """
        self._assert_data_loaded()

        # Ensure infectivity is not already in the dataset
        if "PAR_infectivity" in self.df.columns:
            raise ValueError(
                "Column 'PAR_infectivity' already exists in self.df."
            )

        # Ensure both mean_nContact and p_trans are present
        if {"PAR_mean_nContact", "PAR_p_trans"} - set(self.df.columns):
            raise ValueError(
                "Both 'PAR_mean_nContact' and 'PAR_p_trans' must be in the"
                "dataset."
            )

        logger.info("Applying infectivity (mean_nContact x p_trans)...")

        self.df["PAR_infectivity"] = (
            self.df["PAR_mean_nContact"] * self.df["PAR_p_trans"]
        )
        self.df.drop(
            columns=["PAR_mean_nContact", "PAR_p_trans"], inplace=True
        )

        # Keep column order consistent
        order = [
            "PAR_mean_t_incub",
            "PAR_stdv_t_incub",
            "PAR_infectivity",
            "PAR_p_fatal",
            "PAR_t_recovery"
        ]

        # Include columns not specified in `order`
        remaining = [col for col in self.df.columns if col not in order]
        self.df = self.df[order + remaining]

    def apply_target_transforms(self, transforms: dict[str, Callable]) -> None:
        """
        Apply column-wise transformations (e.g., log, sqrt) to target columns.

        The transformations are stored in `self.transformed_cols` for reference
        or downstream inverse application.

        Parameters
        ----------
        transforms : dict[str, Callable]
            Dictionary mapping column names to transformation functions to
            apply. For example: {"p_fatal": np.log, "t_recovery": np.sqrt}

        Raises
        ------
        ValueError
            If a specified column is missing in the DataFrame.
        """
        self._assert_data_loaded()

        self.transformed_cols = {}
        for col, fn in transforms.items():
            if col not in self.df.columns:
                raise ValueError(
                    f"Target column '{col}' not found in merged dataset."
                )

            logger.debug(f"Applying {fn.__name__} to '{col}'")
            self.df[col] = fn(self.df[col].to_numpy(copy=True))
            self.transformed_cols[col] = fn

    def load_data(self) -> TensorDataset:
        """
        Convert the internal DataFrame into a PyTorch TensorDataset.

        This method extracts columns prefixed with "SST_" as input features and
        "PAR_" as target outputs from the internally stored DataFrame. The
        features are normalized using sklearn's StandardScaler, and the
        resulting tensors are returned in a TensorDataset.

        Returns
        -------
        dataset : torch.utils.data.TensorDataset
            A dataset containing normalized summary statistics as input
            features and simulation parameters as output targets.

        Raises
        ------
        RuntimeError
            If the internal DataFrame is not loaded or empty.
        ValueError
            If no SST_ or PAR_ columns are found in the DataFrame.
        """
        self._assert_data_loaded()

        X = self.x
        y = self.y

        if X.size == 0 or y.size == 0:
            raise ValueError("SST_ or PAR_ columns are missing or empty.")

        X_scaled = StandardScaler().fit_transform(X)

        return TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

    @staticmethod
    def _generate_split_indices(
            n: int,
            ptrain: float,
            pval: float,
            seed: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.arange(n)
        if seed is not None:
            g = torch.Generator().manual_seed(seed)
            indices = torch.randperm(n, generator=g).numpy()
        else:
            np.random.shuffle(indices)

        train_size = round(ptrain * n)
        val_size = round(pval * n)

        return (
            indices[:train_size],
            indices[train_size:train_size + val_size],
            indices[train_size + val_size:]
        )

    def split_data(
        self,
        ptrain: float,
        pval: float,
        seed: Optional[int] = None,
    ) -> tuple[NosoiSplit, NosoiSplit, NosoiSplit]:
        """
        Split processed data into train/val/test sets and preserve alignment
        with raw data.

        Parameters
        ----------
        ptrain : float
            Proportion for training set.
        pval : float
            Proportion for validation set.
        seed : int, optional
            Seed for reproducible shuffling.

        Returns
        -------
        train, val, test : tuple[NosoiSplit, NosoiSplit, NosoiSplit]
            Each NosoiSplit contains:
            - X, y: normalized input features and targets (as torch.Tensor)
            - x_raw, y_raw: raw features and targets (as np.ndarray)
        """
        self._assert_data_loaded()

        # Validate proportions
        if not (0 < ptrain < 1 and 0 < pval < 1):
            raise ValueError(
                "Proportions must be between 0 and 1 (exclusive)."
            )
        if ptrain + pval > 1.0:
            raise ValueError(
                f"ptrain + pval must be ≤ 1.0 (got {ptrain + pval:.4f})"
            )

        # Get and normalize SST features
        X, X_raw = self.x, self.x_raw
        X = StandardScaler().fit_transform(X)

        # Get target parameters
        y, y_raw = self.y, self.y_raw

        train_idx, val_idx, test_idx = self._generate_split_indices(
            len(X), ptrain, pval, seed
        )

        # Slice tensors
        def tensor(array: np.ndarray, idx: np.ndarray) -> torch.Tensor:
            return torch.tensor(array[idx], dtype=torch.float32)

        split_train = NosoiSplit(
            tensor(X, train_idx),
            tensor(y, train_idx),
            X_raw[train_idx],
            y_raw[train_idx]
        )

        split_val = NosoiSplit(
            tensor(X, val_idx),
            tensor(y, val_idx),
            X_raw[val_idx],
            y_raw[val_idx]
        )

        split_test = NosoiSplit(
            tensor(X, test_idx),
            tensor(y, test_idx),
            X_raw[test_idx],
            y_raw[test_idx]
        )

        # Return named splits
        return split_train, split_val, split_test
