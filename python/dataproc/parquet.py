import logging
from pathlib import Path
import re
from typing import Iterator


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


def remove_trivial(Path):
    pass
