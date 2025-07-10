import numpy as np
import pandas as pd
import torch
from typing import Callable, Dict, Tuple
from pathlib import Path
from dataproc.nosoi_split import NosoiSplit
from functools import partial
from concurrent.futures import ProcessPoolExecutor


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


def abc_from_tensor(
    obs_stats: torch.Tensor,
    sim_stats: torch.Tensor,
    sim_params: torch.Tensor,
    distance_fn: Callable[[torch.Tensor, torch.Tensor], float],
    epsilon: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform ABC rejection using precomputed summary statistics and parameters.

    Parameters
    ----------
    obs_stats : torch.Tensor
        1D tensor of observed summary statistics (e.g. shape (33,))
    sim_stats : torch.Tensor
        2D tensor of all simulated summary statistics (e.g. shape (N, 33))
    sim_params : torch.Tensor
        2D tensor of all true parameters corresponding to sim_stats (e.g. shape
        (N, 5))
    distance_fn : Callable
        A function to compute distance between summary statistics.
    epsilon : float
        Distance threshold for acceptance.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple of (accepted_parameters, distances), where accepted_parameters
        has shape (M, 5)
    """
    accepted = []
    distances = []
    for i in range(sim_stats.shape[0]):
        dist = distance_fn(obs_stats, sim_stats[i])
        if dist < epsilon:
            accepted.append(sim_params[i])
            distances.append(dist)
    return torch.stack(accepted), torch.tensor(distances)


def run_abc_for_index(
    i: int,
    obs_all: torch.Tensor,
    params_all: torch.Tensor,
    param_names: list[str],
    epsilon: float,
) -> dict | None:
    """
    Run ABC for a single pseudo-observation index.

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
    epsilon : float
        ABC acceptance threshold.

    Returns
    -------
    dict | None
        Dictionary with results for this observation, or None if no acceptance.
    """
    obs_stats = obs_all[i]

    accepted_params, _ = abc_from_tensor(
        obs_stats=obs_stats,
        sim_stats=obs_all,
        sim_params=params_all,
        distance_fn=euclidean_distance,
        epsilon=epsilon,
    )

    if len(accepted_params) == 0:
        print(f"ABC failed: no accepted samples for idx={i}")
        return None

    result = {
        "idx": i,
        "epsilon": epsilon,
        "n_accepted": len(accepted_params),
    }

    for name, value in zip([f"true_{n}" for n in param_names], params_all[i]):
        result[name] = value.item()

    for name, value in zip([f"post_{n}" for n in param_names], accepted_params.mean(dim=0)):
        result[name] = value.item()

    return result


# TODO: check if the StandardScaler scaling used for train_split.X is correct
# here
# NOTE: vary epsilon; regression
def main() -> None:
    # Load precomputed summary stats and parameters
    splits_path = Path("data/splits/scarce_0.00")
    train_split = NosoiSplit.load("train", splits_path)

    # Randomly select a sample of pseudo-observations to condition on
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(len(train_split.X), size=10, replace=False)

    # Get parameter colum names
    param_names = train_split.y_colnames or [f"{i}" for i in range(train_split.output_dim)]

    # Epsilon value
    epsilon = 1.5

    # Use partial to fix all shared arguments
    abc_task = partial(
        run_abc_for_index,
        obs_all=train_split.X,
        params_all=train_split.y,
        param_names=param_names,
        epsilon=epsilon,
    )

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(abc_task, indices))

    df = pd.DataFrame([r for r in results if r is not None])
    print(df)


if __name__ == "__main__":
    main()
