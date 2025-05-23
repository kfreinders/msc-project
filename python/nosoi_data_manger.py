from typing import Callable, Optional
import pandas as pd
import logging


# Set up logger
logger = logging.getLogger(__name__)


class NosoiDataManager:
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

        # Automatically load and merge data
        self._join_master_and_summary()

    # TODO: add docstring explanation of the protected prefixes and raw copy
    def _join_master_and_summary(self) -> None:
        """
        Load and inner-join summary statistics and parameter files on 'seed'.

        This function populates self.df_merged with the joined dataset,
        containing both summary statistics (e.g., SS_*) and simulation
        parameters from nosoi.
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
        logger.info(f"Dropped {n_before - n_after:,} rows based on filter")

        # Print filter function description for logging purposes
        if filter_fn_desc is not None:
            logger.info(f"Row filtering condition: {filter_fn_desc}")
