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

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import torch

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


def euclidean_distance(obs: torch.Tensor, sim: torch.Tensor) -> float:
    """
    Compute the Euclidean distance between two summary statistics vectors.

    Parameters
    ----------
    obs : torch.Tensor
        Observed summary statistics (1D tensor).
    sim : torch.Tensor
        Simulated summary statistics (1D tensor).

    Returns
    -------
    float
        Euclidean distance between the two vectors.
    """
    return float(torch.norm(obs - sim))


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
    obs_stats: torch.Tensor,
    sim_stats: torch.Tensor,
    sim_params: torch.Tensor,
    distance_fn: Callable[[torch.Tensor, torch.Tensor], float],
    quantile: float = 0.01
) -> torch.Tensor:
    """
    Perform ABC with regression adjustment.

    Parameters
    ----------
    obs_stats : torch.Tensor
        1D tensor of observed summary statistics.
    sim_stats : torch.Tensor
        2D tensor of all simulated summary statistics.
    sim_params : torch.Tensor
        2D tensor of all true parameters corresponding to sim_stats.
    distance_fn : Callable
        Function to compute distance between summary statistics.
    quantile : float
        Proportion of simulations to keep (e.g. 0.01 keeps 1% closest samples).

    Returns
    -------
    torch.Tensor
        Posterior mean estimate after regression adjustment.
    """
    # Step 1: compute distances
    distances = torch.tensor(
        [distance_fn(obs_stats, sim) for sim in sim_stats]
    )
    k = max(1, int(len(distances) * quantile))
    closest_indices = torch.topk(distances, k=k, largest=False).indices

    X = sim_stats[closest_indices]        # (k, d)
    y = sim_params[closest_indices]       # (k, p)
    dists = distances[closest_indices]    # (k,)

    # Step 2: center summary statistics around the observation
    X_centered = X - obs_stats

    # Step 3: Compute Epanechnikov kernel weights
    delta = dists.max().item()
    weights = epanechnikov_kernel(dists.numpy(), delta=delta)  # shape (k,)

    # Step 4: Local-linear regression for each parameter
    adjusted_params = []

    for i in range(y.shape[1]):
        reg = LinearRegression()
        reg.fit(
            X_centered.numpy(),
            y[:, i].numpy(),
            sample_weight=weights
        )
        # Predict at X = 0 (i.e., s = s_obs)
        y_pred = reg.predict(X_centered.numpy())

        # Adjustment
        adjusted = y[:, i].numpy() - (y_pred - reg.intercept_)
        adjusted_params.append(adjusted)

    # Stack and return mean adjusted posterior
    adjusted_params = [i.tolist() for i in adjusted_params]
    adjusted_tensor = torch.tensor(adjusted_params).T
    return adjusted_tensor.mean(dim=0)


def run_abc_for_index(
    i: int,
    obs_all: torch.Tensor,
    params_all: torch.Tensor,
    param_names: list[str],
    quantile: float = 0.01,
    distance_fn: Callable[[torch.Tensor, torch.Tensor], float] = euclidean_distance
) -> dict | None:
    """
    Run ABC with regression adjustment for a single pseudo-observation index.

    Parameters
    ----------
    i : int
        Index of the pseudo-observation to use.
    obs_all : torch.Tensor
        Tensor of all simulated summary statistics.
    params_all : torch.Tensor
        Tensor of true simulation parameters.
    param_names : list[str]
        Parameter names (used to construct true/post keys).
    quantile : float
        Proportion of closest simulations to keep (default 0.01).
    distance_fn : Callable
        Distance function to compare summary statistics.

    Returns
    -------
    dict | None
        Dictionary with results for this observation, or None if no acceptance.
    """
    obs_stats = obs_all[i]

    try:
        adjusted_mean = abc_regression_adjustment(
            obs_stats=obs_stats,
            sim_stats=obs_all,
            sim_params=params_all,
            distance_fn=distance_fn,
            quantile=quantile
        )
    except Exception as e:
        print(f"ABC failed for idx={i}: {e}")
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


def compute_mae(df: pd.DataFrame, param_names: list[str]) -> pd.Series:
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


def plot_errors(df: pd.DataFrame, param_names: list[str]) -> None:
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
    for param in param_names:
        true_col = f"true_{param}"
        post_col = f"post_{param}"

        plt.figure()
        sns.histplot((df[post_col] - df[true_col]).to_list(), kde=True)
        plt.title(f"{param}")
        plt.xlabel("Prediction error")
        plt.ylabel("Count")
        plt.axvline(0, color="red", linestyle="--")
        plt.savefig(f"{post_col}.png", dpi=300, format="png")


# TODO: check if the StandardScaler scaling used for train_split.X is correct
# here
def main() -> None:
    # Load precomputed summary stats and parameters
    splits_path = Path("data/splits/scarce_0.00")
    train_split = NosoiSplit.load("train", splits_path)

    # Randomly select a sample of pseudo-observations to condition on
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(len(train_split.X), size=1000, replace=False)

    # Get parameter colum names
    param_names = (
        train_split.y_colnames or
        [f"{i}" for i in range(train_split.output_dim)]
    )

    # Use partial to fix all shared arguments
    abc_task = partial(
        run_abc_for_index,
        obs_all=train_split.X,
        params_all=train_split.y,
        param_names=param_names,
    )

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(abc_task, indices))

    df = pd.DataFrame([r for r in results if r is not None])
    print(df)

    mae = compute_mae(df, param_names)
    plot_errors(df, param_names)
    print(mae)


if __name__ == "__main__":
    main()
