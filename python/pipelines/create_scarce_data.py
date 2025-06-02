import numpy as np
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
import re
from typing import Iterator

from dataproc.simulation_loader import NosoiSimulation
from dataproc.scarcity_strategies import DataScarcityStrategy, RandomNodeDrop
from dataproc.summary_stats import compute_summary_statistics


def find_parquet_files(root_dir: Path) -> Iterator[Path]:
    """
    Recursively yield all .parquet files under the given root directory.

    Parameters
    ----------
    root_dir : Path
        The root directory to search in.

    Yields
    ------
    Path
        Path to each found .parquet file.
    """
    yield from root_dir.rglob("*.parquet")


def extract_seed(path: Path) -> int:
    """
    Extract the numerical seed from a filename of the form
    'inftable_<seed>_mapped.parquet'.

    Parameters
    ----------
    path : Path
        The file path to extract the seed from.

    Returns
    -------
    int
        The extracted seed.
    """
    match = re.search(r"inftable_(\d+)_mapped\.parquet", path.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract seed from filename: {path.name}")


def apply_single_level(
    root_dir: Path,
    strategy: DataScarcityStrategy,
    level: float,
    output_dir: Path
) -> None:
    """
    Apply a single data scarcity strategy to all .parquet files in a directory.

    Does not overwrite the original .parquet files. Instead, it loads them and
    as NosoiSimulation objects and then applies the strategy to an internal
    graph representation. The summary statistics are then re-computed for the
    degraded graph and saved to disk.

    Parameters
    ----------
    root_dir : Path
        Directory containing .parquet files.
    strategy : DataScarcityStrategy
        Strategy to apply.
    level : float
        The fraction of data to drop, used in output filename.
    output_dir : Path
        Directory to save the output CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"scarce_{level:.2f}.csv"
    results = []

    for file in find_parquet_files(root_dir):
        try:
            seed = extract_seed(file)
            sim = NosoiSimulation.from_parquet(str(file))
            degraded_graph = strategy.apply(sim.as_graph())
            sim._graph = degraded_graph
            stats_df = compute_summary_statistics(sim)

            if not stats_df.empty:
                stats_df.insert(0, "SEED", seed)
                results.append(stats_df)
        except Exception as e:
            print(f"Skipping {file.name}: {e}")

    if results:
        full_df = pd.concat(results, ignore_index=True)
        full_df.to_csv(output_path, index=False)
        print(f"Written to {output_path}")


def _apply_level(args):
    path, level, output_dir = args
    strategy = RandomNodeDrop(level)
    apply_single_level(path, strategy, level, output_dir)


def apply_all_levels(path: Path, levels: np.ndarray, output_dir: Path) -> None:
    """
    Apply multiple levels of data scarcity to all simulation files.

    Parameters
    ----------
    path : Path
        Path to the directory containing simulation .parquet files.
    levels : ArrayLike or Sequence[float]
        The drop percentages to apply (e.g., [0.0, 0.1, 0.2, ...]).
    """
    with Pool() as pool:
        args = [(path, level, output_dir) for level in levels]
        pool.map(_apply_level, args)


def main() -> None:
    input_path = Path("data/nosoi")
    output_path = Path("data/scarce_stats")
    levels = np.linspace(0, 0.5, num=11)  # 0.0, 0.05, ..., 0.50
    apply_all_levels(input_path, levels, output_path)


if __name__ == "__main__":
    main()
