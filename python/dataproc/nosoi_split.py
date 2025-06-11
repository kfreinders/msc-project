import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class NosoiSplit:
    X: torch.Tensor
    y: torch.Tensor
    x_raw: np.ndarray
    y_raw: np.ndarray
    x_raw_columns: Optional[list[str]] = None
    y_raw_columns: Optional[list[str]] = None

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
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.X, os.path.join(output_dir, f"{name}_x.pt"))
        torch.save(self.y, os.path.join(output_dir, f"{name}_y.pt"))

        raw_data = {
            "x_raw": self.x_raw,
            "y_raw": self.y_raw,
        }

        if self.x_raw_columns is not None:
            raw_data["x_raw_columns"] = np.array(self.x_raw_columns)
        if self.y_raw_columns is not None:
            raw_data["y_raw_columns"] = np.array(self.y_raw_columns)

        np.savez(
            os.path.join(output_dir, f"{name}_raw.npz"),
            **raw_data,
            allow_pickle=True
        )

    @classmethod
    def load(
        cls,
        name: str,
        input_dir: Path,
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
        input_dir : Path
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

        X = torch.load(
            os.path.join(input_dir, f"{name}_x.pt"), map_location=device
        )
        y = torch.load(
            os.path.join(input_dir, f"{name}_y.pt"), map_location=device
        )
        raw = np.load(os.path.join(input_dir, f"{name}_raw.npz"))

        x_raw_columns = raw["x_raw_columns"].tolist() if "x_raw_columns" in raw else None
        y_raw_columns = raw["y_raw_columns"].tolist() if "y_raw_columns" in raw else None

        return cls(
            X=X,
            y=y,
            x_raw=raw["x_raw"],
            y_raw=raw["y_raw"],
            x_raw_columns=x_raw_columns,
            y_raw_columns=y_raw_columns
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
        if self.x_raw_columns is None:
            raise ValueError("x_raw column names are not available.")
        try:
            idx = self.x_raw_columns.index(name)
        except ValueError:
            raise ValueError(f"Column '{name}' not found in x_raw_columns.")
        return self.x_raw[:, idx]
