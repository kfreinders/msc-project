"""
Approximate Bayesian Computation (ABC) Inference for nosoi Simulations

This module provides tools to perform Approximate Bayesian Computation (ABC)
using rejection sampling to estimate nosoi simulation parameters from summary
statistics extracted from transmission chains.

The workflow includes:
- Sampling parameter values from prior distributions
- Computing distances between observed and simulated statistics
- Performing ABC rejection to infer posterior parameter estimates
- Aggregating results and evaluating prediction errors
- Plotting error distributions

Functions in this module support parallel execution and assume summary
statistics and parameter data are stored in `NosoiSplit` format.

Dependencies
------------
- matplotlib
- numpy
- pandas
- seaborn
- sklearn
- torch

Examples
--------
Run the script directly to perform ABC on a subset of data and generate plots:

    $ python abc_inference.py

See Also
--------
NosoiSplit : Class for loading and managing simulation splits

"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from functools import partial
import json
import logging
from multiprocessing import cpu_count
import os
from pathlib import Path
import sys
import time
from typing import Callable, Dict, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import torch

from utils.logging_config import setup_logging
from dataproc.nosoi_split import NosoiSplit


# Signature for per-index inference
InferFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, int]
]


class ABCMethod(str, Enum):
    rejection = "rejection"
    regression = "regression"


def make_infer(
    method: ABCMethod,
    *,
    quantile: Optional[float] = None,
    epsilon: Optional[float] = None,
) -> InferFn:
    """
    Build a concrete per-index inference callable for ABC.

    This factory returns a function with the signature:
        infer(obs_stats, sim_stats, sim_params) -> (posterior_mean, accepted)

    Exactly one of (`quantile`, `epsilon`) must be provided, aligned with
    `method`.

    Parameters
    ----------
    method : {"rejection", "regression"}
        ABC method to use.
    quantile : float | None, optional (keyword-only)
        Distance quantile for regression-adjusted ABC.
        Must be in (0, 1] when method='regressionj.
    epsilon : float | None, optional (keyword-only)
        Absolute distance cutoff for naive rejection ABC.
        Must be >= 0 when method='rejection'.

    Returns
    -------
    InferFn
        A callable: infer(obs_stats, sim_stats, sim_params) -> (posterior_mean,
        accepted).

    Raises
    ------
    ValueError
        If an unknown `method` is given, or if the required tolerance
        (`quantile` or `epsilon`) is missing/invalid for the chosen method.
    """
    if method is ABCMethod.regression:
        if quantile is None or not (0.0 < quantile <= 1.0):
            raise ValueError(
                "quantile must be in (0, 1] for method='regression'."
            )
        return cast(InferFn, partial(abc_regression, quantile=quantile))

    if method is ABCMethod.rejection:
        if epsilon is None or epsilon < 0:
            raise ValueError(
                "epsilon must be >= 0 for method='naive'/'rejection'."
            )
        return cast(InferFn, partial(abc_rejection, epsilon=epsilon))


@dataclass
class ABCWorker:
    """
    Run ABC with regression adjustment for a single pseudo-observation index.

    Parameters
    ----------
    i : int
        Index of the pseudo-observation to use.
    obs_all : np.ndarray
        Array of all simulated summary statistics.
    params_all : np.ndarray
        Array of true simulation parameters.
    param_names : list[str]
        Parameter names (used to construct true/post keys).
    quantile : float
        Proportion of closest simulations to keep (default 0.01).
    n_samples : int
       Number of simulations to draw in each ABC run.
    seed: int
        Seed for making runs reproducable.

    Returns
    -------
    dict | None
        Dictionary with results for this observation, or None if no acceptance.
    """
    obs_all: np.ndarray
    params_all: np.ndarray
    param_names: list[str]
    infer: InferFn
    n_samples: int
    seed: int
    method: str
    quantile: float | None
    epsilon: float | None

    def __call__(self, i: int) -> Optional[dict]:
        rng = np.random.default_rng(self.seed + i)
        all_indices = np.arange(len(self.obs_all))

        # Exclude the observation itself if present
        if 0 <= i < len(all_indices):
            all_indices = np.delete(all_indices, i)

        # Subsample candidates
        if self.n_samples >= len(all_indices):
            candidate_indices = all_indices
        else:
            candidate_indices = rng.choice(
                all_indices, size=self.n_samples, replace=False
            )

        sim_stats_sampled = self.obs_all[candidate_indices]
        sim_params_sampled = self.params_all[candidate_indices]

        try:
            post_mean, k = self.infer(
                self.obs_all[i],
                sim_stats_sampled,
                sim_params_sampled,
            )
        except ValueError as e:
            raise e

        if post_mean is None or k == 0:
            # Expected when tolerance is too tight
            return None

        row = {
            "idx": i,
            "method": self.method,
            "quantile": self.quantile,
            "epsilon": self.epsilon,
            **{f"true_{n}": (v.item() if hasattr(v, "item") else float(v))
               for n, v in zip(self.param_names, self.params_all[i])},
            **{f"post_{n}": (v.item() if hasattr(v, "item") else float(v))
               for n, v in zip(self.param_names, post_mean)},
            "k": k,
        }
        return row


@dataclass
class StatusLine:
    total: int
    is_tty: bool = sys.stderr.isatty()
    last: float = 0.0

    def print(self, done: int, failed: int, *, force: bool = False) -> None:
        if not self.is_tty:
            if force:
                logging.info("FINISHED %d | FAILED %d | TOTAL %d/%d",
                             done, failed, done+failed, self.total)
            return
        now = time.monotonic()
        if not force and (now - self.last) < 0.1:
            return
        sys.stderr.write(
            f"\rFINISHED {done:_} | "
            f"FAILED {failed:_} | "
            f"TOTAL {done+failed:_}/{self.total:_}"
        )
        sys.stderr.flush()
        self.last = now


def sample_parameters(
    priors: Dict[str, Callable[[np.random.Generator], float]],
    rng: np.random.Generator
) -> dict:
    """
    Sample a single set of parameters from the provided prior distributions.

    Each prior is a callable that returns a single random sample when given
    a NumPy random generator. This function evaluates each prior to generate
    a complete parameter dictionary.

    Parameters
    ----------
    priors : dict
        A dictionary mapping parameter names to callables that generate random
        samples from their respective prior distributions.
    rng : np.random.Generator
        A NumPy random generator used to ensure reproducible sampling.

    Returns
    -------
    dict
        A dictionary of sampled parameter values keyed by parameter name.
    """
    return {k: f(rng) for k, f in priors.items()}


def abc_regression(
    obs_stats: np.ndarray,
    sim_stats: np.ndarray,
    sim_params: np.ndarray,
    quantile: float = 0.01,
) -> tuple[Optional[np.ndarray], int]:
    """
    Perform ABC with regression adjustment.

    Parameters
    ----------
    obs_stats : np.ndarray
        1D array of observed summary statistics.
    sim_stats : np.ndarray
        2D array of all simulated summary statistics.
    sim_params : np.ndarray
        2D array of all true parameters corresponding to sim_stats.
    quantile : float
        Proportion of simulations to keep (e.g. 0.01 keeps 1% closest samples).
        Default is 0.01.

    Returns
    -------
    tuple[np.ndarray, int]
        Posterior mean estimate after regression adjustment and number of
        accepted samples
    """
    # Step 1: compute distances
    diffs = sim_stats - obs_stats               # (n, f)
    distances = np.linalg.norm(diffs, axis=1)   # (n,  )

    # 2) epsilon as the empirical quantile of distances
    if not (0.0 < quantile <= 1.0):
        raise ValueError("quantile must be in (0, 1].")
    eps = np.quantile(distances, quantile)

    # Degenerate case: only exact matches within epsilon
    if eps == 0.0:
        mask = distances == 0.0
        if not np.any(mask):
            return None, 0
        y = sim_params[mask]
        accepted = int(mask.sum())
        # With Xc == 0 for all, regression brings no information → mean
        return y.mean(axis=0), accepted

    # 3) Epanechnikov kernel weights with bandwidth eps
    w = 1.0 - (distances / eps) ** 2
    w[distances > eps] = 0.0
    mask = w > 0.0
    if not np.any(mask):
        return None, 0

    Xc = diffs[mask]                 # (k, f)
    y = sim_params[mask]             # (k, p)
    weights = w[mask]                # (k,)
    accepted = int(mask.sum())

    # Step 4: local-linear regression for each parameter
    reg = LinearRegression()
    reg.fit(Xc, y, sample_weight=weights)

    # Step 5: adjustment y_adj = y - (Xc @ coef.T)
    # Predict at 0 equals reg.intercept_, so subtract slope contribution only
    y_adj = y - Xc @ reg.coef_.T
    return y_adj.mean(axis=0), accepted


def abc_rejection(
    obs_stats: np.ndarray,
    sim_stats: np.ndarray,
    sim_params: np.ndarray,
    epsilon: float,
) -> tuple[Optional[np.ndarray], int]:
    """
    Perform naive rejection ABC.

    Accept samples whose distance to the observed summary statistics is <=
    epsilon, and return the mean of the accepted parameters.

    Parameters
    ----------
    obs_stats : np.ndarray
        Observed summary statistics.
    sim_stats : np.ndarray
        Simulated summary statistics.
    sim_params : np.ndarray
        True parameters corresponding to sim_stats.
    epsilon : float
        Distance threshold (tolerance). Samples with distance > epsilon are
        rejected.

    Returns
    -------
    posterior_mean : np.ndarray
        Mean of the accepted parameter vectors.
    accepted : int
        Number of accepted samples.

    Raises
    ------
    ValueError
        If no samples are accepted (e.g., epsilon too small).
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    # Distances to observed stats
    diffs = sim_stats - obs_stats            # (n, f)
    distances = np.linalg.norm(diffs, axis=1)

    # Accept if within epsilon
    mask = distances <= epsilon
    if not np.any(mask):
        return None, 0

    y = sim_params[mask]                     # (k, p)
    posterior_mean = y.mean(axis=0)
    accepted = int(mask.sum())
    return posterior_mean, accepted


def compute_mae(
    df: pd.DataFrame,
    param_names: list[str]
) -> pd.Series:
    """
    Compute the mean absolute error (MAE) between true and posterior estimates.

    This function compares the posterior means obtained via ABC with the true
    parameter values for each pseudo-observation and returns the MAE per
    parameter.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns for both true and posterior parameter
        values. Column names must follow the pattern 'true_<param>' and
        'post_<param>'.
    param_names : list[str]
        List of parameter names (without 'true_' or 'post_' prefix) to
        evaluate.

    Returns
    -------
    pd.Series
        Series containing the MAE for each parameter, indexed by
        'post_<param>'.
    """
    post = df[[f"post_{n}" for n in param_names]]
    true = df[[f"true_{n}" for n in param_names]].values
    errors = post - true
    return errors.abs().mean()


def plot_errors(
    df: pd.DataFrame,
    param_names: list[str],
    output_path: Path
) -> None:
    """
    Plot the distribution of prediction errors for each parameter.

    This function generates and saves a histogram (with KDE) of the difference
    between posterior and true parameter values for each specified parameter.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns for both true and posterior parameter
        values. Column names must follow the pattern 'true_<param>' and
        'post_<param>'.
    param_names : list[str]
        List of parameter names (without 'true_' or 'post_' prefix) to plot.
    """
    logger = logging.getLogger(__name__)

    for param in param_names:
        true_col = f"true_{param}"
        post_col = f"post_{param}"
        plotpath = output_path / f"{post_col}.png"

        plt.figure()
        sns.histplot((df[post_col] - df[true_col]).to_list(), kde=True)
        plt.title(f"{param}")
        plt.xlabel("Prediction error")
        plt.ylabel("Count")
        plt.axvline(0, color="red", linestyle="--")
        plt.savefig(plotpath, dpi=300, format="png")

        logger.info(
            f"Saved prediction error distribution plot for {param} "
            f"to {plotpath}"
        )


def pick_num_workers(n_runs: int) -> int:
    """
    Choose worker count based SLURM, number of jobs and available cores.

    Prefers SLURM_CPUS_ON_NODE if the environental variable is defined.
    Otherwise, it used the available cpu count minus one for stability. Always
    capped by n_runs and floor at 1.

    Parameters
    ----------
    n_runs : int
        Number of ABC runs to execute.

    Returns
    -------
    int
        Number of cores to allocate for the script.
    """
    slurm_cores = os.getenv("SLURM_CPUS_ON_NODE")
    if slurm_cores:
        try:
            cores = max(1, int(slurm_cores))
        except ValueError:
            cores = max(1, cpu_count() - 1)
    else:
        cores = max(1, cpu_count() - 1)
    return max(1, min(n_runs, cores))


def run_abc(
    splits_path: Path,
    n_runs: int,
    n_samples: int,
    method: ABCMethod,
    quantile: float | None,
    epsilon: float | None,
    output_path: Path,
    seed: int,
    make_plots: bool
) -> None:
    """
    Infer posterior nosoi parameter distributions with ABC.

    This function loads a test split of simulated data, randomly selects a
    subset of pseudo-observations, and performs ABC with local linear
    regression adjustment to estimate posterior means. Results are saved to
    disk, and optional prediction error plots can be generated.

    Parameters
    ----------
    splits_path : Path
        Path to the directory containing the pickled `NosoiSplit` test split.
    n_runs : int
        Number of pseudo-observations to sample and infer from.
    n_samples: int
       Number of simulations to draw from the test set in each ABC run.
    quantile : float
        Proportion of closest simulations to retain for ABC adjustment
        (e.g. 0.01).
    output_path : Path
        Path to save the resulting parquet file of posterior estimates and
        plots.
    seed : int
        Random seed for reproducibility of sampling.
    make_plots : bool
        Whether to generate and save prediction error plots per parameter.
    """
    logger = logging.getLogger(__name__)
    setup_logging(run_name="abc")

    # Load precomputed summary stats and parameters
    logger.info(f"Loading data splits from {splits_path}")
    train_split = NosoiSplit.load("train", splits_path)
    val_split = NosoiSplit.load("val", splits_path)
    test_split = NosoiSplit.load("test", splits_path)

    X_all = torch.cat((train_split.X, val_split.X, test_split.X)).numpy()
    y_all = torch.cat((train_split.y, val_split.y, test_split.y)).numpy()

    # Randomly select a sample of pseudo-observations to condition on
    rng = np.random.default_rng(seed=seed)
    indices = rng.choice(len(X_all), size=n_runs, replace=False)

    logger.info(
        f"Randomly selecting {n_runs:_} observations..."
    )
    if seed:
        logger.info(f"seed={seed}")

    # Get parameter colum names
    param_names = (
        test_split.y_colnames or
        [f"{i}" for i in range(test_split.output_dim)]
    )
    logger.info(f"Parameters to infer: {param_names}")

    # Use partial to fix all shared arguments
    # Build the per-index infer callable once
    infer = make_infer(method=method, quantile=quantile, epsilon=epsilon)

    # Create a pickleable worker object for the process pool
    worker = ABCWorker(
        obs_all=X_all,
        params_all=y_all,
        param_names=param_names,
        infer=infer,
        n_samples=n_samples,
        seed=seed,
        method=method,
        quantile=quantile,
        epsilon=epsilon,
    )

    n_cores = pick_num_workers(n_runs)
    logger.info(
        f"Starting jobs on {n_cores} cores. This may take a while..."
    )

    status = StatusLine(total=len(indices))
    results: list[dict] = []
    done = failed = 0

    with ProcessPoolExecutor(max_workers=n_cores) as ex:
        futures = [ex.submit(worker, i) for i in indices]
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception:
                failed += 1
                status.print(done, failed)
                continue

            if r is None:
                failed += 1
            else:
                results.append(r)
                done += 1
            status.print(done, failed)

    status.print(done, failed, force=True)        # final flush
    print("", file=sys.stderr)                    # newline after CR line

    df = pd.DataFrame([r for r in results if r is not None])

    if df.empty:
        raise ValueError(
            f"No ABC results available: all samples may have failed. "
            f"Try increasing "
            f"{'--epsilon.' if method == 'rejection' else '--quantile'}"
        )

    if not output_path.exists():
        logger.info(f"Making directory: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path / "abc_data.parquet")
    logger.info(f"Saved all ABC data to {output_path / 'abc_data.parquet'}")

    logger.info("Computing MAE per parameter...")
    mae = compute_mae(df, param_names).to_dict()
    logging.info(mae)

    logging.info(f"Mean accepted samples per run: {df.loc[:, 'k'].mean()}")

    with open(output_path / "abc_mae.json", 'w') as handle:
        json.dump(mae, handle)
    logger.info(f"Saved MAE values to {output_path / 'abc_mae.json'}")

    if make_plots:
        plot_errors(df, param_names, output_path)
        logger.info(f"Exported plots to: {output_path}")


def cli_main():
    parser = argparse.ArgumentParser(
        description=(
            "Infer posterior nosoi parameter distributions with ABC."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--method", type=ABCMethod, choices=list(ABCMethod),
        default=ABCMethod.regression,
        help=(
            "ABC method to use: naive rejection or regression-adjusted "
            "(Beaumont et al., 2002) "
        )
    )
    parser.add_argument(
        "--splits-path", type=str, default="data/splits/scarce_0.00",
        help="Path to the directory containing pickled NosoiSplit objects."
    )
    parser.add_argument(
        "--n-runs", type=int, default=10_000,
        help="Number of pseudo-observations to sample."
    )
    parser.add_argument(
        "--n-samples", type=int, default=1_000,
        help="Number of simulations to draw in each ABC run."
    )
    tol_group = parser.add_mutually_exclusive_group(required=False)
    tol_group.add_argument(
        "--quantile", type=float, default=argparse.SUPPRESS,
        help=(
            "Distance quantile for regression ABC. (default: 0.01; keeps 1%% "
            "closest observations)."
        )
    )
    tol_group.add_argument(
        "--epsilon", type=float, default=argparse.SUPPRESS,
        help="Absolute distance cutoff for naive rejection ABC. (default: 0.5)"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/benchmarks",
        help="Directory to save output data."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--make_plots", action="store_true",
        help="Make and export prediction error distribution plots."
    )

    args = parser.parse_args()

    method: ABCMethod = cast(ABCMethod, args.method)
    if method is ABCMethod.regression:
        if hasattr(args, "epsilon"):
            parser.error("--epsilon is only valid with --method rejection.")
        quantile = getattr(args, "quantile", 0.01)
        if not (0.0 < quantile <= 1.0):
            parser.error("--quantile must be in (0, 1].")
        epsilon = None
    elif method is ABCMethod.rejection:
        if hasattr(args, "quantile"):
            parser.error("--quantile is only valid with --method regression.")
        epsilon = getattr(args, "epsilon", 0.5)
        if epsilon < 0:
            parser.error("--epsilon must be non-negative.")
        quantile = None

    run_abc(
        splits_path=Path(args.splits_path),
        n_runs=args.n_runs,
        n_samples=args.n_samples,
        method=args.method,
        quantile=quantile,
        epsilon=epsilon,
        output_path=Path(args.output_path),
        seed=args.seed,
        make_plots=args.make_plots,
    )


if __name__ == "__main__":
    cli_main()
