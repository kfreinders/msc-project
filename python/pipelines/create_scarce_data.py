import numpy as np
from pathlib import Path
import logging
from multiprocessing import Pool
import pandas as pd
import re
from typing import Iterator

from dataproc.simulation_loader import NosoiSimulation
from dataproc.scarcity_strategies import DataScarcityStrategy, RandomNodeDrop
from dataproc.summary_stats import compute_summary_statistics
from utils.logging_config import setup_logging


def get_logger():
    return logging.getLogger(__name__)


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
    logger = get_logger()
    logger.debug(f"Searching for .parquet files under: {root_dir}")
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
    logger = get_logger()
    match = re.search(r"inftable_(\d+)_mapped\.parquet", path.name)
    if match:
        seed = int(match.group(1))
        logger.debug(f"Extracted seed {seed} from filename: {path.name}")
        return seed
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
    logger = get_logger()
    logger.info(f"Applying scarcity level {level:.2f} to files in {root_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"scarce_{level:.2f}.csv"
    results = []

    for file in find_parquet_files(root_dir):
        try:
            logger.debug(f"Processing file: {file.name}")
            seed = extract_seed(file)
            sim = NosoiSimulation.from_parquet(str(file))
            degraded_graph = strategy.apply(sim.as_graph())
            sim._graph = degraded_graph
            stats_df = compute_summary_statistics(sim)
            logger.debug(f"Stats computed for seed {seed}")

            if not stats_df.empty:
                stats_df.insert(0, "SEED", seed)
                results.append(stats_df)
            else:
                logger.warning(f"Empty stats for {file.name}, skipping.")
        except Exception as e:
            logger.error(
                f"Skipping {file.name} due to error: {e}", exc_info=True
            )

    if results:
        full_df = pd.concat(results, ignore_index=True)
        full_df.to_csv(output_path, index=False)
        logger.debug(f"Written summary statistics to: {output_path}")
    else:
        logger.warning(
            f"No results written for level {level:.2f}, no successful files."
        )


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
    logger = get_logger()
    logger.info(f"Starting processing with levels: {levels}")
    with Pool() as pool:
        args = [(path, level, output_dir) for level in levels]
        pool.map(_apply_level, args)
    logger.info("Finished applying all scarcity levels.")


def main() -> None:
    setup_logging("training")
    logger = get_logger()
    setup_logging(run_name="scarcity_pipeline")
    logger.info("Starting main pipeline")

    input_path = Path("data/nosoi")
    output_path = Path("data/scarce_stats")
    levels = np.linspace(0.05, 0.5, num=10)

    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Levels: {levels}")

    apply_all_levels(input_path, levels, output_path)
    logger.info("Main pipeline completed")


if __name__ == "__main__":
    main()
