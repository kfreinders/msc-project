import argparse
import hashlib
from typing import Optional
import math
import numpy as np
from pathlib import Path
import pandas as pd
import logging
from multiprocessing import Pool

from dataproc.simulation_loader import NosoiSimulation
from dataproc.scarcity_strategies import DataScarcityStrategy, RandomNodeDrop
from dataproc.summary_stats import compute_summary_statistics
from dataproc.parquet import find_parquet_files, extract_seed, peek_host_count
from utils.logging_config import setup_logging


def get_logger():
    return logging.getLogger(__name__)


def deterministic_seed(*args) -> int:
    """
    Generate a deterministic 32-bit seed from input arguments.

    This function concatenates all input arguments as an UTF-8 string from
    which to generate an sha256 hash.

    Returns
    -------
    int
        Deterministic 32-bit seed based on the input arguments
    """
    joined = '|'.join(map(str, args)).encode('utf-8')
    digest = hashlib.sha256(joined).digest()
    return int.from_bytes(digest[:4], 'big') % 2**32


def get_seeds(
    csv_file: Path
) -> tuple[Optional[list[int]], Optional[list[str]]]:
    if csv_file.is_file() and (df := pd.read_csv(csv_file)) is not None:
        seeds = df["seed"].tolist()
        headers = df.columns.values.tolist()
        return seeds, headers
    return None, None


def apply_single_level(
    root_dir: Path,
    strategy: DataScarcityStrategy,
    level: float,
    output_path: Path,
    min_hosts: float,
    max_hosts: float,
    truncate: int
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
    output_path : Path
        Directory to save the output CSV file.
    min_hosts : float
        Only include transmission chains with at least this minimum number of
        hosts.
    max_hosts : float
        Only include transmission chains with at most this maximum number of
        hosts.
    truncate: bool
         If > 0, truncate oversized simulations to this number of hosts.
    """
    logger = get_logger()
    logger.info(f"Applying scarcity level {level:.2f} to files in {root_dir}")
    output_path = output_path / f"scarce_{level:.2f}.csv"

    seeds, headers = get_seeds(output_path)

    # Track whether to write the header
    first_write = True if not headers else False

    for file in find_parquet_files(root_dir):
        sim_seed = extract_seed(file)
        host_count = peek_host_count(file)

        # Skip already finished simulations
        if seeds and sim_seed in seeds:
            logger.info(f"Skipping already finished file: {file.name}")
            continue

        # Skip simulations with too few hosts
        if host_count < min_hosts:
            logger.info(
                f"Level {level}: "
                f"skipping {file.name} ({host_count} < {min_hosts})."
            )
            continue

        # Skip simulations with too many hosts
        if host_count > max_hosts and truncate == 0:
            logger.info(
                f"Level {level}: "
                f"skipping {file.name} ({host_count} > {max_hosts})."
            )
            continue

        try:
            sim = NosoiSimulation.from_parquet(file)

            # Truncate oversized simulations if requested
            if truncate > 0 and sim.n_hosts > max_hosts:
                logger.info(
                    f"Level {level}: truncating {file.name} "
                    f"({host_count} > {max_hosts}), target {truncate}."
                )

            sim = sim.truncate(truncate)

            # Create deterministic degradation seed
            deg_seed = deterministic_seed(sim_seed, level)
            logger.debug(
                f"Level: {level}: "
                f"using degradation seed {deg_seed} for {file.name}"
            )

            # Apply degradation if level > 0
            if math.isclose(level, 0.0):
                logger.info(
                    f"Level {level}: no scarcity applied to {file.name}"
                )
            else:
                sim.graph = strategy.apply(sim.graph, seed=deg_seed)

            # Re-compute summary statistics
            stats_df = compute_summary_statistics(sim)
            if stats_df.empty:
                logger.warning("No stats for %s, skipping.", file.name)
                continue

            stats_df.insert(0, "seed", sim_seed)
            stats_df.to_csv(
                output_path, mode="a", header=first_write, index=False
            )
            first_write = False  # Only write header on first iteration
            logger.info(f"Level {level}: finished {file.name}")

        except Exception as e:
            logger.error(
                f"Error processing {file.name}: {e}", exc_info=True
            )


def _apply_level(args):
    path, level, output_path, min_hosts, max_hosts, truncate = args
    strategy = RandomNodeDrop(level)
    apply_single_level(
        path,
        strategy,
        level,
        output_path,
        min_hosts,
        max_hosts,
        truncate
    )


def apply_all_levels(
    path: Path,
    levels: np.ndarray,
    output_path: Path,
    min_hosts: float,
    max_hosts: float,
    truncate: int
) -> None:
    """
    Apply multiple levels of data scarcity to all simulation files.

    Parameters
    ----------
    path : Path
        Path to the directory containing simulation .parquet files.
    levels : ArrayLike or Sequence[float]
        The drop percentages to apply (e.g., [0.0, 0.1, 0.2, ...]).
    output_path : Path
        Path to output scare summary statistic datasets to.
    min_hosts : float
        Only include transmission chains with at least this minimum number of
        hosts.
    max_hosts : float
        Only include transmission chains with at most this maximum number of
        hosts.
    truncate: int
         If > 0, truncate oversized simulations to this number of hosts.
    """
    logger = get_logger()
    logger.info(f"Starting processing with levels: {levels}")
    with Pool() as pool:
        args = [(path, level, output_path, min_hosts, max_hosts, truncate) for level in levels]
        pool.map(_apply_level, args)
    logger.info("Finished applying all scarcity levels.")


def cli_main():
    parser = argparse.ArgumentParser(
        description=(
            """Apply a data scarcity strategy to nosoi simulations and compute
            the new summary statistics."""
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-path", type=Path, default=Path("data/nosoi"),
        help="Path to directory with input .parquet simulations."
    )
    parser.add_argument(
        "--output-path", type=Path, default=Path("data/scarce_stats"),
        help="Directory to save degraded summary statistics."
    )
    parser.add_argument(
        "--min-hosts", type=int, default=2000,
        help="Minimum number of hosts required."
    )
    parser.add_argument(
        "--max-hosts", type=int, default=math.inf,
        help="Maximum number of hosts allowed."
    )
    parser.add_argument(
        "--truncate", type=int, default=0,
        help="If > 0, truncate oversized simulations to this number of hosts.")
    parser.add_argument(
        "--min-level", type=float, default=0.00,
        help="Lower bound level of scarcity to apply (inclusive)."
    )
    parser.add_argument(
        "--max-level", type=float, default=0.5,
        help="Upper bound level of scarcity to apply (inclusive)."
    )
    parser.add_argument(
        "--steps-level", type=int, default=11,
        help="Number levels to use between the upper and lower bound."
    )

    args = parser.parse_args()

    # Argument validation
    if args.min_hosts < 1:
        parser.error("--min-hosts must be >= 1")

    if args.max_hosts != math.inf and args.max_hosts < args.min_hosts:
        parser.error("--max-hosts must be >= --min-hosts")

    if args.truncate > 0 and not (args.min_hosts <= args.truncate <= args.max_hosts):
        parser.error(
            f"--truncate {args.truncate} is invalid. "
            f"Must be between min-hosts ({args.min_hosts}) "
            f"and max-hosts ({args.max_hosts})."
        )

    if args.min_level < 0.0 or args.max_level > 1.0:
        parser.error("Level boundaries must must all be in [0, 1]")

    if not args.input_path.exists():
        parser.error(
            f"Input path {args.input_path} does not exist."
        )

    if args.steps_level < 1:
        parser.error("--steps-level must be >= 1")

    args.output_path.mkdir(parents=True, exist_ok=True)

    setup_logging(run_name="create_scarce_sst")
    levels = np.linspace(
        args.min_level,
        args.max_level,
        num=args.steps_level
    )

    logger = get_logger()
    logger.info(f"Processing scarcity levels: {levels}")

    apply_all_levels(
        path=args.input_path,
        levels=levels,
        output_path=args.output_path,
        min_hosts=args.min_hosts,
        max_hosts=args.max_hosts,
        truncate=args.truncate
    )


if __name__ == "__main__":
    cli_main()
