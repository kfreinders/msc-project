import logging
from pathlib import Path
import pyarrow.parquet as pq
import re
from typing import Iterator

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
    setup_logging()
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
    setup_logging()
    logger = get_logger()
    match = re.search(r"inftable_(\d+)_mapped\.parquet", path.name)
    if match:
        seed = int(match.group(1))
        logger.debug(f"Extracted seed {seed} from filename: {path.name}")
        return seed
    raise ValueError(f"Could not extract seed from filename: {path.name}")


def peek_host_count(parquet_path: Path) -> int:
    """
    Cheaply get the number of rows in a Parquet file.

    With the mapping applied to a transmission chain in a Paruqet file, each
    row represents an infection event. Therefore, each row corresponds to a
    host and thus the total number of rows equals the total number of hosts.
    Getting the total number of hosts this way is much cheaper than fully
    reading the file or deserialization by instantiating a NosoiSimulation
    object.

    Parameters
    ----------
    path : Path
        Path to the parquet file to read.

    Returns
    -------
    int
        The total number of hosts.
    """
    return pq.ParquetFile(parquet_path).metadata.num_rows
