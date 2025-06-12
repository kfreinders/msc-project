import numpy as np
from pathlib import Path
import logging
from multiprocessing import Pool

from dataproc.simulation_loader import NosoiSimulation
from dataproc.scarcity_strategies import DataScarcityStrategy, RandomNodeDrop
from dataproc.summary_stats import compute_summary_statistics
from dataproc.parquet import find_parquet_files, extract_seed, peek_host_count
from utils.logging_config import setup_logging


def get_logger():
    return logging.getLogger(__name__)


def apply_single_level(
    root_dir: Path,
    strategy: DataScarcityStrategy,
    level: float,
    output_dir: Path,
    min_hosts: int = 2_000
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
    min_hosts : in
        Only include transmission chains with at least this minimum number of
        hosts.
    """
    logger = get_logger()
    logger.info(f"Applying scarcity level {level:.2f} to files in {root_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"scarce_{level:.2f}.csv"

    # Track whether to write the header
    first_write = True

    for file in find_parquet_files(root_dir):
        try:
            # Skip simulations with too few hosts
            host_count = peek_host_count(file)
            if host_count < min_hosts:
                logger.info(
                    f"Skipping {file.name} ({host_count} < {min_hosts})."
                )
                continue

            seed = extract_seed(file)
            sim = NosoiSimulation.from_parquet(file)

            # Apply degradation & recompute summary statistics
            sim.graph = strategy.apply(sim.graph)
            stats_df = compute_summary_statistics(sim)

            if stats_df.empty:
                logger.warning("No stats for %s, skipping.", file.name)
                continue

            stats_df.insert(0, "seed", seed)
            stats_df.to_csv(
                output_path, mode="a", header=first_write, index=False
            )
            first_write = False  # Only write header on first iteration
            logger.debug("Appended stats for seed %010d", seed)

        except Exception as e:
            logger.error(
                f"Error processing {file.name}: {e}", exc_info=True
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
    logger = get_logger()
    setup_logging(run_name="create_scarce_sst")

    logger.info("Starting main pipeline")
    input_path = Path("data/nosoi")
    output_path = Path("data/scarce_stats")
    levels = np.linspace(0.05, 0.5, num=10)

    logger.info(f"Input path to Parquet files: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Scarcity levels: {levels}")

    apply_all_levels(input_path, levels, output_path)
    logger.info("Main pipeline completed")


if __name__ == "__main__":
    main()
