import argparse
import pandas as pd
from pathlib import Path


def select_balanced_data(
    file1: Path,
    file2: Path,
    output_path: Path,
    seed: int = 42
):
    """
    Create a balanced dataset by randomly sampling rows from two CSV files
    while ensuring that no selected seeds overlap between them.

    This function loads two datasets that share the same set of seeds and
    yields a random 50/50 balanced selection of rows from file1 and file2
    without overlapping seeds.

    Parameters
    ----------
    file1 : Path
        Path to the first CSV file (e.g., full transmission chains).
    file2 : Path
        Path to the second CSV file (e.g., degraded transmission chains).
    output_path : Path
        Destination file where the balanced dataset will be saved.
    seed : int, optional
        Random seed for reproducibility of the sampling process (default=42).

    Returns
    -------
    pd.DataFrame
        A shuffled DataFrame containing the balanced, non-overlapping rows
        from both input files.
    """
    # Load CSVs
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    assert len(df1) == len(df2), "df1 and df2 must be equal length."

    # Step 1: Randomly select half of df1
    n_select = len(df1) // 2
    df1_sample = df1.sample(n=n_select, random_state=seed)
    selected_seeds = set(df1_sample["seed"])

    # Step 2: Select rows from df2 whose seeds are not in df1_sample
    df2_sample = df2[~df2["seed"].isin(selected_seeds)]

    # Step 3: Ensure both sets have equal size
    max_size = min(len(df1_sample), len(df2_sample))
    df1_sample = df1_sample.head(max_size)
    df2_sample = df2_sample.head(max_size)

    # Combine and shuffle
    df_balanced = pd.concat([df1_sample, df2_sample]).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(output_path, index=False)

    return df_balanced


def cli_main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a balanced from two scarce data CSV files output by "
            "create_scarce_data.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--file1", type=Path, required=True,
        help="Path to first scarce data CSV file"
    )
    parser.add_argument(
        "--file2", type=Path, required=True,
        help="Path to second scarce data CSV file"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Where to save the balanced output CSV"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="What seed to use for reproducing results."
    )

    args = parser.parse_args()
    select_balanced_data(args.file1, args.file2, args.output, args.seed)


if __name__ == "__main__":
    cli_main()
