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
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import logging
from multiprocessing import cpu_count
from pathlib import Path
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import torch

from utils.logging_config import setup_logging
from dataproc.nosoi_split import NosoiSplit


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


def euclidean_distance(obs: np.ndarray, sim: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two summary statistics vectors.

    Parameters
    ----------
    obs : np.ndarray
        Observed summary statistics (1D array).
    sim : np.ndarray
        Simulated summary statistics (1D array).

    Returns
    -------
    float
        Euclidean distance between the two vectors.
    """
    return float(np.linalg.norm(obs - sim))


def epanechnikov_kernel(distances: np.ndarray, delta: float) -> np.ndarray:
    """
    Compute Epanechnikov kernel weights for a set of distances.

    Parameters
    ----------
    distances : np.ndarray
        1D array of distances between observed and simulated summary
        statistics.
    delta : float
        Bandwidth (maximum distance in accepted subset).

    Returns
    -------
    np.ndarray
        Kernel weights of the same shape as `distances`.
    """
    scaled = distances / delta
    weights = np.where(
        distances <= delta,
        (1 - scaled**2),
        0.0
    )
    return weights


def abc_regression_adjustment(
    obs_stats: np.ndarray,
    sim_stats: np.ndarray,
    sim_params: np.ndarray,
    distance_fn: Callable[[np.ndarray, np.ndarray], float],
    quantile: float = 0.01,
) -> np.ndarray:
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
    distance_fn : Callable
        Function to compute distance between summary statistics.
    quantile : float
        Proportion of simulations to keep (e.g. 0.01 keeps 1% closest samples).
        Default is 0.01.

    Returns
    -------
    np.ndarray
        Posterior mean estimate after regression adjustment.
    """
    # Step 1: compute distances
    distances = np.array([
        distance_fn(obs_stats, sim) for sim in sim_stats
    ])

    k = int(len(distances) * quantile)
    if k == 0:
        raise ValueError("No samples accepted")

    closest_indices = np.argpartition(distances, k)[:k]

    X = sim_stats[closest_indices]      # shape (k, features)
    y = sim_params[closest_indices]     # shape (k, parameters)
    dists = distances[closest_indices]  # shape (k)

    # Step 2: center summary statistics around the observation
    X_centered = X - obs_stats

    # Step 3: Compute Epanechnikov kernel weights
    delta = dists.max()
    weights = epanechnikov_kernel(dists, delta=delta)  # shape (k,)

    if np.sum(weights) == 0:
        raise ValueError(
            "All kernel weights are zero. Try increasing --quantile."
        )

    # Step 4: Local-linear regression for each parameter
    adjusted_params = []
    for j in range(y.shape[1]):
        reg = LinearRegression()
        reg.fit(X_centered, y[:, j], sample_weight=weights)
        # Predict at X = 0 (i.e., centered around obs_stats)
        y_pred = reg.predict(X_centered)
        adjusted = y[:, j] - (y_pred - reg.intercept_)
        adjusted_params.append(adjusted)

    adjusted_array = np.stack(adjusted_params, axis=1)  # shape (k, p)
    return adjusted_array.mean(axis=0)


def run_abc_for_index(
    i: int,
    obs_all: np.ndarray,
    params_all: np.ndarray,
    param_names: list[str],
    quantile: float = 0.01,
    distance_fn: Callable[[np.ndarray, np.ndarray], float] = euclidean_distance,
    n_samples: int = 1_000,
    seed: int = 42
) -> dict | None:
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
    distance_fn : Callable
        Distance function to compare summary statistics.
    n_samples : int
       Number of simulations to draw in each ABC run.
    seed: int
        Seed for making runs reproducable.

    Returns
    -------
    dict | None
        Dictionary with results for this observation, or None if no acceptance.
    """
    # Set up logger
    logger = logging.getLogger(__name__)

    rng = np.random.default_rng(seed + i)
    all_indices = np.arange(len(obs_all))
    if i in all_indices:
        all_indices = np.delete(all_indices, i)  # Exclude observation
    if n_samples >= len(all_indices):
        candidate_indices = all_indices
    else:
        candidate_indices = rng.choice(all_indices, size=n_samples, replace=False)

    sim_stats_sampled = obs_all[candidate_indices]
    sim_params_sampled = params_all[candidate_indices]

    try:
        adjusted_mean = abc_regression_adjustment(
            obs_stats=obs_all[i],
            sim_stats=sim_stats_sampled,
            sim_params=sim_params_sampled,
            distance_fn=distance_fn,
            quantile=quantile,
        )
    except Exception as e:
        logger.warning(f"ABC failed for idx={i}: {e}")
        return None

    result = {
        "idx": i,
        "quantile": quantile,
    }

    for name, value in zip([f"true_{n}" for n in param_names], params_all[i]):
        result[name] = value.item()

    for name, value in zip([f"post_{n}" for n in param_names], adjusted_mean):
        result[name] = value.item()

    return result


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


def run_abc(
    splits_path: Path,
    n_runs: int,
    n_samples: int,
    quantile: float,
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
        Path to save the resulting CSV file of posterior estimates and plots.
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
        f"Randomly selecting {n_runs:_} samples"
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
    abc_task = partial(
        run_abc_for_index,
        obs_all=X_all,
        params_all=y_all,
        param_names=param_names,
        quantile=quantile,
        n_samples=n_samples,
        seed=seed
    )

    n_cores = cpu_count() if n_runs >= cpu_count() else n_runs
    logger.info(
        f"Starting jobs on {n_cores} cores. This may take a while..."
    )

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = list(executor.map(abc_task, indices))

    df = pd.DataFrame([r for r in results if r is not None])

    if df.empty:
        raise ValueError(
            "No ABC results available: all samples may have failed. "
            "Try increasing --quantile."
        )

    if not output_path.exists():
        logger.info(f"Making directory: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path / "abc_data.csv")
    logger.info(f"Saved all ABC data to {output_path / 'abc_data.csv'}")

    logger.info("Computing MAE per parameter...")
    mae = compute_mae(df, param_names).to_dict()
    logging.info(mae)

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
    parser.add_argument(
        "--quantile", type=float, default=0.01,
        help="Proportion of simulations to retain in ABC rejection step."
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
        "--make_plots", type=bool, default=False,
        help="Make and export prediction error distribution plots."
    )

    args = parser.parse_args()
    run_abc(
        splits_path=Path(args.splits_path),
        n_runs=args.n_runs,
        n_samples=args.n_samples,
        quantile=args.quantile,
        output_path=Path(args.output_path),
        seed=args.seed,
        make_plots=args.make_plots
    )


if __name__ == "__main__":
    cli_main()
