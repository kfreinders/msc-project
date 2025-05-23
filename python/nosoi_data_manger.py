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
                "Column 'PAR_infectivity' already exists in df_merged."
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
