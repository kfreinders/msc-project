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
