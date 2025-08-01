"""
nosoi_split.py
--------------

This module defines the `NosoiSplit` class, which encapsulates aligned
processed (torch.Tensor) and raw (NumPy array) data for a given data split
(e.g., train, validation, test) in nosoi-based simulation studies.

It supports:
- Easy access to input/output dimensions for model initialization
- Creation of PyTorch DataLoaders
- Save/load functionality preserving both processed and raw feature mappings
- Column-aware access to raw features for interpretability or analysis

Typical usage involves preparing splits in a data processing pipeline and
loading them for training and evaluation of neural network models.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class NosoiSplit:
    """
    A data container representing one split (train/val/test) of simulation
    data.

    This class holds both processed input/output tensors (`X`, `y`) for model
    training, and their aligned raw NumPy counterparts (`x_raw`, `y_raw`).
    Optional column names are stored to enable semantic lookup.

    Core features:
    - Generates PyTorch DataLoaders via `make_dataloader()`.
    - Preserves traceability between normalized inputs and raw epidemiological
      parameters after shuffling the full dataset by NosoiDataProcessor.
    - Can be saved to and loaded from disk using `.pt` and `.npz` formats.
    - Allows access to specific raw features and targets.

    Useful for analysis of predictions in the context of their original
    simulation inputs.
    """
    X: torch.Tensor
    y: torch.Tensor
    x_raw: np.ndarray
    y_raw: np.ndarray
    x_raw_colnames: list[str]
    y_raw_colnames: list[str]
    x_colnames: list[str]
    y_colnames: list[str]

    @property
    def input_dim(self) -> int:
        """
        Number of input features in the processed dataset.

        Returns
        -------
        int
            Number of columns in `X`.
        """
        return self.X.shape[1]

    @property
    def output_dim(self) -> int:
        """
        Number of target output parameters in the processed dataset.

        Returns
        -------
        int
            Number of columns in `y`.
        """
        return self.y.shape[1]

    def make_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = False
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader for the processed input and output tensors.

        Warning: If `shuffle=True`, the correspondence between the processed
        data (`X`, `y`) and their raw counterparts (`x_raw`, `y_raw`) is lost.
        Use shuffling only when alignment with raw data is not needed (e.g.,
        during training). For tracing model predictions back to the raw inputs,
        leave shuffle=False to preserve alignment.

        Parameters
        ----------
        batch_size : int, optional
            Number of samples per batch to load. Default is 32.
        shuffle : bool, optional
            Whether to shuffle the data at every epoch. Default is True.
        Returns
        -------
        DataLoader
            A PyTorch DataLoader that yields batches of (X, y) pairs.
        """
        dataset = TensorDataset(self.X, self.y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def save(self, name: str, output_dir: Path) -> None:
        """
        Save the split to disk, including both processed and raw data.

        Saves `X` and `y` as PyTorch `.pt` files and `x_raw` and `y_raw` as
        compressed NumPy `.npz`.

        Parameters
        ----------
        name : str
            Prefix for the output files (e.g., 'train', 'val', 'test').
        output_dir : Path
            Directory where the files will be saved.
        """
        # Make output dir if it doesn't exist already
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the PyTorch tensors (X, y) as .pt
        torch.save(self.X, output_dir / f"{name}_x.pt")
        torch.save(self.y, output_dir / f"{name}_y.pt")

        raw_data = {
            "x_raw": self.x_raw,
            "y_raw": self.y_raw,
        }

        # Save column names if they exist
        if self.x_raw_colnames is not None:
            raw_data["x_raw_columns"] = np.array(self.x_raw_colnames)
        if self.y_raw_colnames is not None:
            raw_data["y_raw_columns"] = np.array(self.y_raw_colnames)
        if self.x_colnames is not None:
            raw_data["x_columns"] = np.array(self.x_colnames)
        if self.y_colnames is not None:
            raw_data["y_columns"] = np.array(self.y_colnames)

        np.savez(
            output_dir / f"{name}_raw.npz",
            **raw_data,
            allow_pickle=True
        )

    @classmethod
    def load(
        cls,
        name: str,
        path: Path,
        device: Optional[torch.device] = None
    ) -> "NosoiSplit":
        """
        Load a previously saved split from disk.

        This method reconstructs a `NosoiSplit` object by loading the
        processed tensors (`X`, `y`) and the raw arrays (`x_raw`, `y_raw`).

        Parameters
        ----------
        name : str
            Prefix of the saved files to load (e.g., 'train', 'val', 'test').
        path : Path
            Directory containing the saved files.
        device : torch.device, optional
            Device to map tensors to (e.g., 'cpu', 'cuda'). Defaults to CPU.

        Returns
        -------
        NosoiSplit
            The reconstructed split containing processed and raw data.
        """
        if device is None:
            device = torch.device("cpu")

        x_file = path / f"{name}_x.pt"
        y_file = path / f"{name}_y.pt"
        raw_file = path / f"{name}_raw.npz"

        if any(not f.exists() for f in [x_file, y_file, raw_file]):
            raise FileNotFoundError(f"Missing files for split '{name}'")

        X = torch.load(x_file, map_location=device)
        y = torch.load(y_file, map_location=device)
        raw = np.load(raw_file)

        x_raw_columns = raw.get("x_raw_columns", None)
        if x_raw_columns is not None:
            x_raw_columns = x_raw_columns.tolist()

        y_raw_columns = raw.get("y_raw_columns", None)
        if y_raw_columns is not None:
            y_raw_columns = y_raw_columns.tolist()

        x_columns = raw.get("x_columns", None)
        if x_columns is not None:
            x_columns = x_columns.tolist()

        y_columns = raw.get("y_columns", None)
        if y_columns is not None:
            y_columns = y_columns.tolist()

        return cls(
            X=X,
            y=y,
            x_raw=raw["x_raw"],
            y_raw=raw["y_raw"],
            x_raw_colnames=x_raw_columns,
            y_raw_colnames=y_raw_columns,
            x_colnames=x_columns,
            y_colnames=y_columns
        )

    def get_raw_feature(self, name: str) -> np.ndarray:
        """
        Get raw input feature values by column name.

        Parameters
        ----------
        name : str
            Column name to retrieve (e.g., 'SST_11').

        Returns
        -------
        np.ndarray
            Values of the requested raw input feature.

        Raises
        ------
        ValueError
            If column names are not available or the name is not found.
        """
        if self.x_raw_colnames is None:
            raise ValueError("x_raw column names are not available.")
        try:
            idx = self.x_raw_colnames.index(name)
        except ValueError:
            raise ValueError(f"Column '{name}' not found in x_raw_columns.")
        return self.x_raw[:, idx]
