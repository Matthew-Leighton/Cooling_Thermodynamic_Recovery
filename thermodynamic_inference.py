#!/usr/bin/env python3
"""Infer thermodynamically allowed hot-equilibrium distributions.

The command-line interface accepts cooled state-probability distributions at
one or more cooling times.  With exact input probabilities it samples the
region satisfying manuscript Eqs. (2), (3), (8), (9), (11), and (12).  With
95% confidence intervals it draws cooling distributions from Dirichlet
approximations and projects probability/entropy proposals onto the feasible
region.

Run ``python thermodynamic_inference.py --help`` for usage.
"""

from __future__ import annotations

import os

# Avoid nested BLAS parallelism when sampler workers are used.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog, minimize
from scipy.stats import beta as beta_distribution
from scipy.stats import qmc


CI_Z_VALUE = 1.96
PROBABILITY_FLOOR = 1e-12
CONSTRAINT_BUFFER = 1e-8
ACCEPTANCE_TOLERANCE = 1e-9
DEFAULT_SEED = 20_260_814
DEFAULT_SAMPLE_COUNT = 100_000
DEFAULT_SIMPLEX_THRESHOLD = 6
DEFAULT_MCMC_CHAINS = 4
DEFAULT_PROPOSAL_CONCENTRATION = 200.0
DEFAULT_COLD_LOGIT_WEIGHT = 0.02
DEFAULT_PROJECTION_CHUNK_SIZE = 500


@dataclass(frozen=True)
class CoolingData:
    """Validated cooled-distribution input."""

    times: np.ndarray
    states: tuple[str, ...]
    probabilities: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    uncertain_runs: np.ndarray

    @property
    def state_count(self) -> int:
        return len(self.states)

    @property
    def has_uncertainty(self) -> bool:
        return bool(np.any(self.uncertain_runs))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cooled",
        type=Path,
        required=True,
        help=(
            "Long-form CSV with cooling_time, state, probability, and optional "
            "ci_lower_95/ci_upper_95 columns."
        ),
    )
    parser.add_argument(
        "--hot-temperature",
        type=float,
        required=True,
        help="Hot absolute temperature in any consistent unit.",
    )
    parser.add_argument(
        "--cold-temperature",
        type=float,
        required=True,
        help="Cold absolute temperature in the same unit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("thermodynamic_inference_results"),
        help="Directory for generated inference files.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help="Number of retained probability/entropy samples.",
    )
    parser.add_argument(
        "--microstate-entropies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Infer microstate entropies; disable to fix every entropy to zero.",
    )
    parser.add_argument(
        "--entropy-bound",
        type=float,
        default=None,
        help=(
            "Maximum absolute centered microstate entropy in units of k_B; "
            "defaults to ln(number of states)."
        ),
    )
    parser.add_argument(
        "--estimator",
        choices=("center", "nearest-median"),
        default="center",
        help=(
            "Point estimator. Center uses the componentwise mean and population "
            "standard deviation of all generated samples."
        ),
    )
    parser.add_argument(
        "--nearest-fraction",
        type=float,
        default=0.10,
        help="Fraction used by the nearest-median estimator.",
    )
    parser.add_argument(
        "--exact-method",
        choices=("auto", "simplex", "mcmc"),
        default="auto",
        help="Sampling method when the cooled probabilities are exact.",
    )
    parser.add_argument(
        "--simplex-threshold",
        type=int,
        default=DEFAULT_SIMPLEX_THRESHOLD,
        help="Auto mode uses simplex rejection at or below this state count.",
    )
    parser.add_argument(
        "--qmc-batch-power",
        type=int,
        default=18,
        help="Each exact simplex batch contains 2**qmc_batch_power proposals.",
    )
    parser.add_argument(
        "--max-proposals",
        type=int,
        default=None,
        help=(
            "Maximum exact simplex proposals; defaults to max(2**24, "
            "10,000 times n_samples)."
        ),
    )
    parser.add_argument(
        "--mcmc-chains",
        type=int,
        default=DEFAULT_MCMC_CHAINS,
        help="Number of chains for exact-data MCMC.",
    )
    parser.add_argument(
        "--mcmc-burn-in",
        type=int,
        default=5_000,
        help="Discarded tuning sweeps per exact-data MCMC chain.",
    )
    parser.add_argument(
        "--mcmc-thin",
        type=int,
        default=5,
        help="MCMC sweeps between retained samples.",
    )
    parser.add_argument(
        "--proposal-concentration",
        type=float,
        default=DEFAULT_PROPOSAL_CONCENTRATION,
        help="Dirichlet concentration for probability targets.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=300,
        help="Maximum iterations for each nonlinear constrained solve.",
    )
    parser.add_argument(
        "--max-attempt-factor",
        type=int,
        default=20,
        help="Maximum uncertain proposals per requested retained sample.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Independent worker processes.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--write-sample-csv",
        action="store_true",
        help="Also write the full sample matrices as CSV files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace package-generated files already present in output-dir.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive.")
    if not (
        math.isfinite(args.hot_temperature)
        and math.isfinite(args.cold_temperature)
        and args.hot_temperature > args.cold_temperature > 0.0
    ):
        raise ValueError("Require hot-temperature > cold-temperature > 0.")
    if args.entropy_bound is not None and (
        not math.isfinite(args.entropy_bound) or args.entropy_bound < 0.0
    ):
        raise ValueError("--entropy-bound must be nonnegative.")
    if not (0.0 < args.nearest_fraction <= 1.0):
        raise ValueError("--nearest-fraction must lie in (0, 1].")
    if args.simplex_threshold < 2:
        raise ValueError("--simplex-threshold must be at least 2.")
    if not (8 <= args.qmc_batch_power <= 24):
        raise ValueError("--qmc-batch-power must lie between 8 and 24.")
    if args.max_proposals is not None and args.max_proposals <= 0:
        raise ValueError("--max-proposals must be positive.")
    if args.mcmc_chains is not None and args.mcmc_chains <= 0:
        raise ValueError("--mcmc-chains must be positive.")
    if args.mcmc_burn_in < 0 or args.mcmc_thin <= 0:
        raise ValueError("MCMC burn-in must be nonnegative and thinning positive.")
    if (
        not math.isfinite(args.proposal_concentration)
        or args.proposal_concentration <= 0.0
    ):
        raise ValueError("--proposal-concentration must be positive.")
    if args.max_iterations <= 0 or args.max_attempt_factor <= 0:
        raise ValueError("Projection iteration and attempt limits must be positive.")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_for_metadata(path: Path) -> str:
    """Prefer a portable path when the file is inside the working directory."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def load_cooled_csv(path: Path) -> CoolingData:
    """Load and validate the documented long-form cooled CSV schema."""
    frame = pd.read_csv(path)
    required = {"cooling_time", "state", "probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    lower_name = "ci_lower_95"
    upper_name = "ci_upper_95"
    has_lower = lower_name in frame.columns
    has_upper = upper_name in frame.columns
    if has_lower != has_upper:
        raise ValueError(
            "Confidence intervals require both ci_lower_95 and ci_upper_95."
        )
    if not has_lower:
        frame[lower_name] = np.nan
        frame[upper_name] = np.nan

    frame = frame.copy()
    frame["cooling_time"] = pd.to_numeric(frame["cooling_time"], errors="raise")
    frame["probability"] = pd.to_numeric(frame["probability"], errors="raise")
    frame[lower_name] = pd.to_numeric(frame[lower_name], errors="raise")
    frame[upper_name] = pd.to_numeric(frame[upper_name], errors="raise")
    if frame["state"].isna().any():
        raise ValueError("State labels cannot be blank or missing.")
    frame["state"] = frame["state"].astype(str).str.strip()
    if (frame["state"] == "").any():
        raise ValueError("State labels cannot be blank or missing.")

    if frame.empty:
        raise ValueError("The cooled CSV contains no rows.")
    if not np.all(np.isfinite(frame["cooling_time"])):
        raise ValueError("Cooling times must be finite numbers.")
    if np.any(frame["cooling_time"] < 0.0):
        raise ValueError("Cooling times must be nonnegative durations.")
    if frame.duplicated(["cooling_time", "state"]).any():
        raise ValueError("Each cooling_time/state pair must appear exactly once.")

    times = np.sort(frame["cooling_time"].unique().astype(float))
    states = tuple(pd.unique(frame["state"]).tolist())
    if len(states) < 2:
        raise ValueError("At least two states are required.")

    probabilities = []
    lower_rows = []
    upper_rows = []
    uncertain_runs = []
    expected_states = set(states)
    for cooling_time in times:
        block = frame.loc[frame["cooling_time"] == cooling_time].set_index("state")
        if set(block.index) != expected_states or len(block) != len(states):
            raise ValueError(
                "Every cooling time must contain exactly the same set of states."
            )
        block = block.loc[list(states)]
        probability = block["probability"].to_numpy(float)
        if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
            raise ValueError("Probabilities must be finite and nonnegative.")
        total = float(probability.sum())
        if not np.isclose(total, 1.0, atol=1e-6, rtol=0.0):
            raise ValueError(
                f"Probabilities at cooling_time={cooling_time:g} sum to {total:.12g}, "
                "not 1."
            )
        probability = probability / total

        low = block[lower_name].to_numpy(float)
        high = block[upper_name].to_numpy(float)
        present = np.isfinite(low) | np.isfinite(high)
        uncertain = bool(np.any(present))
        if uncertain:
            if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
                raise ValueError(
                    f"Cooling time {cooling_time:g} has an incomplete confidence interval."
                )
            if np.any(low < 0.0) or np.any(high > 1.0):
                raise ValueError("Confidence bounds must lie between 0 and 1.")
            if np.any(low > probability) or np.any(probability > high):
                raise ValueError(
                    "Each probability must lie inside its stated confidence interval."
                )
            if np.any(high <= low):
                raise ValueError("Every confidence interval must have positive width.")
        else:
            low = np.full(len(states), np.nan)
            high = np.full(len(states), np.nan)

        probabilities.append(probability)
        lower_rows.append(low)
        upper_rows.append(high)
        uncertain_runs.append(uncertain)

    return CoolingData(
        times=times,
        states=states,
        probabilities=np.asarray(probabilities),
        ci_lower=np.asarray(lower_rows),
        ci_upper=np.asarray(upper_rows),
        uncertain_runs=np.asarray(uncertain_runs, dtype=bool),
    )


def safe_xlogx(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    terms = np.zeros_like(values, dtype=float)
    positive = values > 0.0
    terms[positive] = values[positive] * np.log(values[positive])
    return np.sum(terms, axis=axis)


def centered_entropy(entropy: np.ndarray) -> np.ndarray:
    """Choose the gauge whose minimum and maximum are symmetric about zero."""
    entropy = np.asarray(entropy, dtype=float)
    if entropy.ndim == 1:
        return entropy - 0.5 * (entropy.min() + entropy.max())
    return entropy - 0.5 * (
        entropy.min(axis=1, keepdims=True) + entropy.max(axis=1, keepdims=True)
    )


def implied_cold_distribution(
    probability: np.ndarray, entropy: np.ndarray, beta: float
) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    entropy = np.asarray(entropy, dtype=float)
    log_probability = np.log(np.maximum(probability, PROBABILITY_FLOOR))
    log_weight = beta * log_probability + (1.0 - beta) * entropy
    if probability.ndim == 1:
        log_weight -= np.max(log_weight)
        cold = np.exp(log_weight)
        return cold / cold.sum()
    log_weight -= np.max(log_weight, axis=1, keepdims=True)
    cold = np.exp(log_weight)
    return cold / cold.sum(axis=1, keepdims=True)


def manuscript_constraint_margins(
    probability: np.ndarray,
    entropy: np.ndarray,
    cooled: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Evaluate manuscript Eqs. (2), (3), (8), (9), (11), and (12)."""
    probability = np.asarray(probability, dtype=float)
    entropy = np.asarray(entropy, dtype=float)
    cooled = np.asarray(cooled, dtype=float)
    log_probability = np.log(np.maximum(probability, PROBABILITY_FLOOR))
    energies = -log_probability + entropy
    cold = implied_cold_distribution(probability, entropy, beta)

    hot_energy = float(probability @ energies)
    cold_energy = float(cold @ energies)
    cooled_energy = cooled @ energies

    hot_entropy = float(probability @ (entropy - log_probability))
    cold_entropy = float(cold @ (entropy - np.log(np.maximum(cold, PROBABILITY_FLOOR))))
    cooled_entropy = cooled @ entropy - safe_xlogx(cooled, axis=1)

    sigma_cooling_upper_bound = (
        cooled_entropy - hot_entropy - beta * (cooled_energy - hot_energy)
    )
    sigma_auxiliary = (
        cold_entropy - cooled_entropy - beta * (cold_energy - cooled_energy)
    )
    margins = [sigma_cooling_upper_bound, sigma_auxiliary]
    if len(cooled) > 1:
        margins.extend([np.diff(sigma_cooling_upper_bound), -np.diff(sigma_auxiliary)])
    return np.concatenate(margins)


def fixed_cooled_batch_mask(
    probabilities: np.ndarray,
    entropies: np.ndarray,
    cooled: np.ndarray,
    beta: float,
    tolerance: float = ACCEPTANCE_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized exact-data membership test using manuscript equations."""
    log_probability = np.log(np.maximum(probabilities, PROBABILITY_FLOOR))
    energies = -log_probability + entropies
    cold = implied_cold_distribution(probabilities, entropies, beta)

    hot_energy = np.sum(probabilities * energies, axis=1)
    cold_energy = np.sum(cold * energies, axis=1)
    cooled_energy = energies @ cooled.T

    hot_entropy = np.sum(probabilities * (entropies - log_probability), axis=1)
    cold_entropy = np.sum(
        cold * (entropies - np.log(np.maximum(cold, PROBABILITY_FLOOR))), axis=1
    )
    cooled_entropy = entropies @ cooled.T - safe_xlogx(cooled, axis=1)[None, :]

    sigma_cooling_upper_bound = (
        cooled_entropy
        - hot_entropy[:, None]
        - beta * (cooled_energy - hot_energy[:, None])
    )
    sigma_auxiliary = (
        cold_entropy[:, None]
        - cooled_entropy
        - beta * (cold_energy[:, None] - cooled_energy)
    )

    parts = [sigma_cooling_upper_bound, sigma_auxiliary]
    if cooled.shape[0] > 1:
        parts.extend(
            [
                np.diff(sigma_cooling_upper_bound, axis=1),
                -np.diff(sigma_auxiliary, axis=1),
            ]
        )
    margins = np.concatenate(parts, axis=1)
    minimum = margins.min(axis=1)
    return minimum >= -tolerance, minimum


def probability_basis(state_count: int) -> np.ndarray:
    dimension = state_count - 1
    return np.vstack([np.eye(dimension), -np.ones(dimension)])


def unpack_probability_coordinates(coordinates: np.ndarray) -> np.ndarray:
    return np.concatenate([coordinates, [1.0 - coordinates.sum()]])


def dirichlet_from_mean_ci(
    mean: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Approximate one Dirichlet law from marginal 95% interval widths."""
    sigma = (upper - lower) / (2.0 * CI_Z_VALUE)
    variance = np.maximum(sigma**2, np.finfo(float).tiny)
    clipped = np.clip(mean, PROBABILITY_FLOOR, 1.0)
    clipped = clipped / clipped.sum()
    estimates = clipped * (1.0 - clipped) / variance - 1.0
    valid = np.isfinite(estimates) & (estimates > 0.0)
    if not np.any(valid):
        raise ValueError("Could not infer a positive Dirichlet concentration.")
    concentration = float(np.median(estimates[valid]))
    alpha = np.maximum(clipped * concentration, PROBABILITY_FLOOR)
    return alpha


def make_projection_context(
    cooled: CoolingData,
    beta: float,
    entropy_bound: float,
    use_entropies: bool,
    proposal_concentration: float,
) -> dict[str, Any]:
    center = cooled.probabilities[0].copy()
    center = np.clip(center, PROBABILITY_FLOOR, None)
    center /= center.sum()
    alpha = np.maximum(center * proposal_concentration, PROBABILITY_FLOOR)
    scale = np.sqrt(center * (1.0 - center) / (proposal_concentration + 1.0))
    scale = np.maximum(scale, 1e-5)

    cooled_alpha: list[np.ndarray | None] = []
    for index in range(len(cooled.times)):
        if cooled.uncertain_runs[index]:
            cooled_alpha.append(
                dirichlet_from_mean_ci(
                    cooled.probabilities[index],
                    cooled.ci_lower[index],
                    cooled.ci_upper[index],
                )
            )
        else:
            cooled_alpha.append(None)
    return {
        "cooled_mean": cooled.probabilities,
        "cooled_alpha": cooled_alpha,
        "p_center": center,
        "p_alpha": alpha,
        "p_scale": scale,
        "beta": beta,
        "entropy_bound": entropy_bound,
        "entropy_range_limit": 2.0 * entropy_bound,
        "use_entropies": use_entropies,
        "state_count": cooled.state_count,
    }


def make_uncertainty_model_summary(
    cooled: CoolingData, context: dict[str, Any]
) -> pd.DataFrame:
    """Describe how each supplied interval maps to a Dirichlet marginal."""
    rows: list[dict[str, Any]] = []
    for run_index, alpha in enumerate(context["cooled_alpha"]):
        if alpha is None:
            continue
        concentration = float(alpha.sum())
        modeled_mean = alpha / concentration
        modeled_variance = (
            alpha * (concentration - alpha) / (concentration**2 * (concentration + 1.0))
        )
        modeled_lower = beta_distribution.ppf(0.025, alpha, concentration - alpha)
        modeled_upper = beta_distribution.ppf(0.975, alpha, concentration - alpha)
        for state_index, state in enumerate(cooled.states):
            input_width = (
                cooled.ci_upper[run_index, state_index]
                - cooled.ci_lower[run_index, state_index]
            )
            modeled_width = modeled_upper[state_index] - modeled_lower[state_index]
            rows.append(
                {
                    "cooling_time": cooled.times[run_index],
                    "state_index": state_index,
                    "state": state,
                    "input_probability": cooled.probabilities[run_index, state_index],
                    "input_ci_lower_95": cooled.ci_lower[run_index, state_index],
                    "input_ci_upper_95": cooled.ci_upper[run_index, state_index],
                    "dirichlet_alpha": alpha[state_index],
                    "dirichlet_total_concentration": concentration,
                    "modeled_probability": modeled_mean[state_index],
                    "modeled_standard_deviation": math.sqrt(
                        modeled_variance[state_index]
                    ),
                    "modeled_ci_lower_95": modeled_lower[state_index],
                    "modeled_ci_upper_95": modeled_upper[state_index],
                    "modeled_to_input_width_ratio": (modeled_width / input_width),
                }
            )
    return pd.DataFrame(rows)


def unpack_variable_projection(
    parameters: np.ndarray, state_count: int
) -> tuple[np.ndarray, np.ndarray]:
    dimension = state_count - 1
    probability = unpack_probability_coordinates(parameters[:dimension])
    entropy = parameters[dimension:]
    return probability, entropy


def variable_projection_margins(
    parameters: np.ndarray, cooled: np.ndarray, context: dict[str, Any]
) -> np.ndarray:
    state_count = context["state_count"]
    beta = context["beta"]
    probability, entropy = unpack_variable_projection(parameters, state_count)
    safe_probability = np.maximum(probability, PROBABILITY_FLOOR)
    log_cold = beta * np.log(safe_probability) + (1.0 - beta) * entropy
    negative_entropies = safe_xlogx(cooled, axis=1)
    kl_without_normalizer = negative_entropies - cooled @ log_cold

    margins: list[float | np.ndarray] = [probability[-1] - PROBABILITY_FLOOR]
    if len(cooled) > 1:
        margins.append(-np.diff(kl_without_normalizer))
    hot_margin = (
        np.sum(safe_probability * np.log(safe_probability))
        - probability @ log_cold
        - negative_entropies[0]
        + cooled[0] @ log_cold
    )
    margins.append(float(hot_margin))
    return np.concatenate([np.atleast_1d(part) for part in margins])


def variable_projection_jacobian(
    parameters: np.ndarray, cooled: np.ndarray, context: dict[str, Any]
) -> np.ndarray:
    state_count = context["state_count"]
    dimension = state_count - 1
    beta = context["beta"]
    probability, entropy = unpack_variable_projection(parameters, state_count)
    safe_probability = np.maximum(probability, PROBABILITY_FLOOR)
    log_cold = beta * np.log(safe_probability) + (1.0 - beta) * entropy
    basis = probability_basis(state_count)
    row_count = 1 + max(0, len(cooled) - 1) + 1
    jacobian = np.zeros((row_count, dimension + state_count))
    row = 0

    jacobian[row, :dimension] = -1.0
    row += 1
    for index in range(1, len(cooled)):
        difference = cooled[index - 1] - cooled[index]
        jacobian[row, :dimension] = basis.T @ (-beta * difference / safe_probability)
        jacobian[row, dimension:] = (beta - 1.0) * difference
        row += 1

    jacobian[row, :dimension] = basis.T @ (
        np.log(safe_probability)
        + 1.0
        - log_cold
        + beta * (cooled[0] - probability) / safe_probability
    )
    jacobian[row, dimension:] = (1.0 - beta) * (cooled[0] - probability)
    return jacobian


def variable_target_parameters(
    target_probability: np.ndarray,
    target_entropy: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = target_probability.size - 1
    target_entropy = centered_entropy(target_entropy)
    cold = implied_cold_distribution(target_probability, target_entropy, beta)
    target_log_cold = np.log(cold)
    target_log_cold -= target_log_cold.mean()
    parameters = np.concatenate([target_probability[:dimension], target_entropy])
    return target_log_cold, parameters


def zero_entropy_projection_margins(
    coordinates: np.ndarray, cooled: np.ndarray, context: dict[str, Any]
) -> np.ndarray:
    probability = unpack_probability_coordinates(coordinates)
    beta = context["beta"]
    safe_probability = np.maximum(probability, PROBABILITY_FLOOR)
    log_probability = np.log(safe_probability)
    negative_entropies = safe_xlogx(cooled, axis=1)
    margins: list[float | np.ndarray] = [probability[-1] - PROBABILITY_FLOOR]
    if len(cooled) > 1:
        cooled_differences = cooled[:-1] - cooled[1:]
        ordered = (
            negative_entropies[:-1]
            - negative_entropies[1:]
            - beta * (cooled_differences @ log_probability)
        )
        margins.append(ordered)
    hot_margin = (
        (1.0 - beta) * np.sum(probability * log_probability)
        - negative_entropies[0]
        + beta * cooled[0] @ log_probability
    )
    margins.append(float(hot_margin))
    return np.concatenate([np.atleast_1d(part) for part in margins])


def zero_entropy_projection_jacobian(
    coordinates: np.ndarray, cooled: np.ndarray, context: dict[str, Any]
) -> np.ndarray:
    probability = unpack_probability_coordinates(coordinates)
    safe_probability = np.maximum(probability, PROBABILITY_FLOOR)
    beta = context["beta"]
    basis = probability_basis(context["state_count"])
    row_count = 1 + max(0, len(cooled) - 1) + 1
    jacobian = np.zeros((row_count, context["state_count"] - 1))
    row = 0
    jacobian[row] = -1.0
    row += 1
    for index in range(1, len(cooled)):
        full_gradient = -beta * (cooled[index - 1] - cooled[index]) / safe_probability
        jacobian[row] = basis.T @ full_gradient
        row += 1
    full_gradient = (1.0 - beta) * (np.log(safe_probability) + 1.0) + beta * cooled[
        0
    ] / safe_probability
    jacobian[row] = basis.T @ full_gradient
    return jacobian


def recover_variable_projection(
    parameters: np.ndarray, context: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability, entropy = unpack_variable_projection(
        parameters, context["state_count"]
    )
    entropy = centered_entropy(entropy)
    cold = implied_cold_distribution(probability, entropy, context["beta"])
    return probability, entropy, cold


def entropy_constraint_terms(
    probability: np.ndarray,
    cooled: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return constants and entropy coefficients for the reduced margins.

    At fixed hot probability, every nonredundant thermodynamic constraint is
    affine in the microstate entropies.  Each returned margin has the form
    ``constant + coefficients @ entropy``.
    """
    log_probability = np.log(np.maximum(probability, PROBABILITY_FLOOR))
    cooled_xlogx = safe_xlogx(cooled, axis=1)
    constants: list[float] = []
    coefficients: list[np.ndarray] = []

    for index in range(1, len(cooled)):
        difference = cooled[index - 1] - cooled[index]
        constants.append(
            float(
                cooled_xlogx[index - 1]
                - cooled_xlogx[index]
                - beta * difference @ log_probability
            )
        )
        coefficients.append((beta - 1.0) * difference)

    hot_difference = cooled[0] - probability
    constants.append(
        float(
            (1.0 - beta) * np.sum(probability * log_probability)
            - cooled_xlogx[0]
            + beta * cooled[0] @ log_probability
        )
    )
    coefficients.append((1.0 - beta) * hot_difference)
    return np.asarray(constants), np.asarray(coefficients)


def project_entropy_at_probability(
    probability: np.ndarray,
    cooled: np.ndarray,
    target_entropy: np.ndarray,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    """Find the closest bounded entropy witness for one fixed probability.

    The subproblem is a linear program: thermodynamic margins are affine in
    entropy, and auxiliary variables minimize the L1 distance to the sampled
    entropy target.  This is substantially more stable than asking a nonlinear
    optimizer to discover feasibility in many sparse state dimensions.
    """
    state_count = context["state_count"]
    if (
        probability.shape != (state_count,)
        or np.any(probability <= PROBABILITY_FLOOR)
        or abs(float(probability.sum()) - 1.0) > 1e-9
    ):
        return None

    target_entropy = centered_entropy(target_entropy)
    constants, coefficients = entropy_constraint_terms(
        probability, cooled, context["beta"]
    )
    zero_block = np.zeros_like(coefficients)
    identity = np.eye(state_count)
    thermo_matrix = np.hstack([-coefficients, zero_block])
    upper_distance = np.hstack([identity, -identity])
    lower_distance = np.hstack([-identity, -identity])
    constraint_matrix = np.vstack([thermo_matrix, upper_distance, lower_distance])
    constraint_upper = np.concatenate(
        [
            constants - CONSTRAINT_BUFFER,
            target_entropy,
            -target_entropy,
        ]
    )
    objective = np.concatenate([np.zeros(state_count), np.ones(state_count)])
    bounds = [(-context["entropy_bound"], context["entropy_bound"])] * state_count + [
        (0.0, None)
    ] * state_count
    result = linprog(
        objective,
        A_ub=constraint_matrix,
        b_ub=constraint_upper,
        bounds=bounds,
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        },
    )
    if not result.success:
        return None

    entropy = centered_entropy(result.x[:state_count])
    cold = implied_cold_distribution(probability, entropy, context["beta"])
    original_margin = float(
        manuscript_constraint_margins(
            probability, entropy, cooled, context["beta"]
        ).min()
    )
    if (
        original_margin < -ACCEPTANCE_TOLERANCE
        or np.ptp(entropy) > context["entropy_range_limit"] + ACCEPTANCE_TOLERANCE
    ):
        return None
    return {
        "entropy": entropy,
        "cold": cold,
        "original_margin": original_margin,
        "linear_program_iterations": int(result.nit),
        "entropy_target_l1_objective": float(result.fun),
    }


def project_variable_target_radially(
    cooled: np.ndarray,
    target_probability: np.ndarray,
    target_entropy: np.ndarray,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    """Project a variable-entropy target toward a feasible cooled-data seed."""
    target_witness = project_entropy_at_probability(
        target_probability, cooled, target_entropy, context
    )
    radial_fraction = 1.0
    lp_calls = 1
    probability = target_probability
    witness = target_witness

    if witness is None:
        anchor_candidates = [
            context["p_center"],
            cooled[0],
        ]
        anchor_probability: np.ndarray | None = None
        anchor_witness: dict[str, Any] | None = None
        for candidate in anchor_candidates:
            candidate = np.maximum(candidate, 2.0 * PROBABILITY_FLOOR)
            candidate = candidate / candidate.sum()
            candidate_witness = project_entropy_at_probability(
                candidate, cooled, target_entropy, context
            )
            lp_calls += 1
            if candidate_witness is not None:
                anchor_probability = candidate
                anchor_witness = candidate_witness
                break
        if anchor_probability is None or anchor_witness is None:
            return None

        low = 0.0
        high = 1.0
        probability = anchor_probability
        witness = anchor_witness
        # Feasibility need not be monotone along this ray.  A coarse scan from
        # target to anchor catches broad disconnected intervals before the
        # local boundary refinement.  Narrow disconnected intervals can still
        # be missed, which is why this remains a proposal-dependent heuristic.
        for fraction in (0.8, 0.6, 0.4, 0.2):
            candidate = (
                1.0 - fraction
            ) * anchor_probability + fraction * target_probability
            candidate_witness = project_entropy_at_probability(
                candidate, cooled, target_entropy, context
            )
            lp_calls += 1
            if candidate_witness is not None:
                low = fraction
                probability = candidate
                witness = candidate_witness
                break
            high = fraction
        for _ in range(8):
            midpoint = 0.5 * (low + high)
            candidate = (
                1.0 - midpoint
            ) * anchor_probability + midpoint * target_probability
            candidate_witness = project_entropy_at_probability(
                candidate, cooled, target_entropy, context
            )
            lp_calls += 1
            if candidate_witness is None:
                high = midpoint
            else:
                low = midpoint
                probability = candidate
                witness = candidate_witness
        radial_fraction = low

    if witness is None:
        return None
    entropy = witness["entropy"]
    cold = witness["cold"]
    objective = witness["entropy_target_l1_objective"]
    return {
        "valid": True,
        "parameters": np.concatenate([probability[:-1], entropy]),
        "probability": probability,
        "entropy": entropy,
        "cold": cold,
        "reduced_margin": float(
            variable_projection_margins(
                np.concatenate([probability[:-1], entropy]), cooled, context
            ).min()
        ),
        "original_margin": witness["original_margin"],
        "iterations": witness["linear_program_iterations"],
        "objective": objective,
        "optimizer_success": True,
        "projection_strategy": "entropy_lp_radial",
        "radial_fraction": radial_fraction,
        "linear_program_calls": lp_calls,
    }


def project_target(
    cooled: np.ndarray,
    target_probability: np.ndarray,
    target_entropy: np.ndarray,
    initial_parameters: np.ndarray,
    context: dict[str, Any],
    max_iterations: int,
    cold_logit_weight: float,
) -> dict[str, Any]:
    """Project one target onto the reduced thermodynamic feasible set."""
    state_count = context["state_count"]
    dimension = state_count - 1
    basis = probability_basis(state_count)
    scale = context["p_scale"]

    if context["use_entropies"]:
        target_log_cold, target_parameters = variable_target_parameters(
            target_probability, target_entropy, context["beta"]
        )

        def objective(parameters: np.ndarray) -> float:
            probability, entropy = unpack_variable_projection(parameters, state_count)
            log_cold = (
                context["beta"] * np.log(np.maximum(probability, PROBABILITY_FLOOR))
                + (1.0 - context["beta"]) * entropy
            )
            log_cold -= log_cold.mean()
            p_term = np.sum(((probability - target_probability) / scale) ** 2)
            cold_term = np.sum((log_cold - target_log_cold) ** 2)
            return float(p_term + cold_logit_weight * cold_term)

        def objective_jacobian(parameters: np.ndarray) -> np.ndarray:
            probability, entropy = unpack_variable_projection(parameters, state_count)
            safe_probability = np.maximum(probability, PROBABILITY_FLOOR)
            log_cold = (
                context["beta"] * np.log(safe_probability)
                + (1.0 - context["beta"]) * entropy
            )
            log_cold -= log_cold.mean()
            difference = log_cold - target_log_cold
            gradient = np.zeros(dimension + state_count)
            gradient[:dimension] = basis.T @ (
                2.0 * (probability - target_probability) / scale**2
            )
            centering = (
                np.eye(state_count) - np.ones((state_count, state_count)) / state_count
            )
            cold_probability_jacobian = centering @ (
                context["beta"] * np.diag(1.0 / safe_probability) @ basis
            )
            cold_entropy_jacobian = (1.0 - context["beta"]) * centering
            gradient[:dimension] += (
                2.0 * cold_logit_weight * cold_probability_jacobian.T @ difference
            )
            gradient[dimension:] = (
                2.0 * cold_logit_weight * cold_entropy_jacobian.T @ difference
            )
            return gradient

        bounds = [(PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)] * dimension + [
            (-context["entropy_bound"], context["entropy_bound"])
        ] * state_count
        margin_function = lambda x: variable_projection_margins(x, cooled, context)
        jacobian_function = lambda x: variable_projection_jacobian(x, cooled, context)
        if initial_parameters.shape != target_parameters.shape:
            initial_parameters = target_parameters
    else:
        target_parameters = target_probability[:dimension]

        def objective(parameters: np.ndarray) -> float:
            probability = unpack_probability_coordinates(parameters)
            return float(np.sum(((probability - target_probability) / scale) ** 2))

        def objective_jacobian(parameters: np.ndarray) -> np.ndarray:
            probability = unpack_probability_coordinates(parameters)
            return basis.T @ (2.0 * (probability - target_probability) / scale**2)

        bounds = [(PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)] * dimension
        margin_function = lambda x: zero_entropy_projection_margins(x, cooled, context)
        jacobian_function = lambda x: zero_entropy_projection_jacobian(
            x, cooled, context
        )
        if initial_parameters.shape != target_parameters.shape:
            initial_parameters = target_parameters

    constraints = {
        "type": "ineq",
        "fun": lambda x: margin_function(x) - CONSTRAINT_BUFFER,
        "jac": jacobian_function,
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Values in x were outside bounds during a minimize step",
            category=RuntimeWarning,
        )
        result = minimize(
            objective,
            initial_parameters,
            jac=objective_jacobian,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iterations, "ftol": 1e-10, "disp": False},
        )

    reduced_margin = float(margin_function(result.x).min())
    if context["use_entropies"]:
        probability, entropy, cold = recover_variable_projection(result.x, context)
    else:
        probability = unpack_probability_coordinates(result.x)
        entropy = np.zeros(state_count)
        cold = implied_cold_distribution(probability, entropy, context["beta"])
    original_margin = float(
        manuscript_constraint_margins(
            probability, entropy, cooled, context["beta"]
        ).min()
    )
    valid = (
        reduced_margin >= -ACCEPTANCE_TOLERANCE
        and original_margin >= -ACCEPTANCE_TOLERANCE
        and np.all(probability >= PROBABILITY_FLOOR / 2.0)
        and abs(probability.sum() - 1.0) <= 1e-9
        and (
            not context["use_entropies"]
            or np.ptp(entropy) <= context["entropy_range_limit"] + ACCEPTANCE_TOLERANCE
        )
    )
    return {
        "valid": bool(valid),
        "parameters": result.x,
        "probability": probability,
        "entropy": entropy,
        "cold": cold,
        "reduced_margin": reduced_margin,
        "original_margin": original_margin,
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "optimizer_success": bool(result.success),
    }


def find_anchor(
    context: dict[str, Any],
    cooled: np.ndarray,
    max_iterations: int,
    cold_logit_weight: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    targets: list[tuple[np.ndarray, np.ndarray]] = [
        (context["p_center"], np.zeros(context["state_count"]))
    ]
    for _ in range(12):
        target_probability = rng.dirichlet(context["p_alpha"])
        if context["use_entropies"]:
            target_entropy = rng.uniform(
                -context["entropy_bound"],
                context["entropy_bound"],
                context["state_count"],
            )
        else:
            target_entropy = np.zeros(context["state_count"])
        targets.append((target_probability, target_entropy))

    initial: np.ndarray | None = None
    for target_probability, target_entropy in targets:
        if context["use_entropies"]:
            _, target_parameters = variable_target_parameters(
                target_probability, target_entropy, context["beta"]
            )
        else:
            target_parameters = target_probability[:-1]
        result = project_target(
            cooled,
            target_probability,
            target_entropy,
            target_parameters if initial is None else initial,
            context,
            max_iterations,
            cold_logit_weight,
        )
        if result["valid"]:
            return result
        initial = result["parameters"]
    raise RuntimeError(
        "Could not find a thermodynamically feasible anchor. Check that the "
        "input cooling distributions are mutually consistent with the selected "
        "entropy setting and bound."
    )


def simplex_points_from_uniforms(uniforms: np.ndarray, state_count: int) -> np.ndarray:
    cuts = np.sort(uniforms[:, : state_count - 1], axis=1)
    probabilities = np.empty((len(uniforms), state_count))
    probabilities[:, 0] = cuts[:, 0]
    probabilities[:, 1:-1] = np.diff(cuts, axis=1)
    probabilities[:, -1] = 1.0 - cuts[:, -1]
    return probabilities


def exact_qmc_batch(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        batch_id,
        batch_power,
        proposal_count,
        seed,
        cooled,
        beta,
        use_entropies,
        entropy_bound,
    ) = task
    state_count = cooled.shape[1]
    dimension = state_count - 1
    sobol_dimension = dimension * (2 if use_entropies else 1)
    engine = qmc.Sobol(d=sobol_dimension, scramble=True, seed=seed + batch_id)
    uniforms = engine.random_base2(batch_power)[:proposal_count]
    probabilities = simplex_points_from_uniforms(uniforms, state_count)
    positive = np.all(probabilities > PROBABILITY_FLOOR, axis=1)

    if use_entropies:
        raw_entropy = np.column_stack(
            [
                (4.0 * entropy_bound) * uniforms[:, dimension:] - 2.0 * entropy_bound,
                np.zeros(len(uniforms)),
            ]
        )
        entropy_domain = np.ptp(raw_entropy, axis=1) <= 2.0 * entropy_bound
        domain = positive & entropy_domain
        entropies = centered_entropy(raw_entropy[domain])
    else:
        domain = positive
        entropies = np.zeros((int(domain.sum()), state_count))

    candidate_probabilities = probabilities[domain]
    accepted_mask, minimum = fixed_cooled_batch_mask(
        candidate_probabilities, entropies, cooled, beta
    )
    accepted_probabilities = candidate_probabilities[accepted_mask]
    accepted_entropies = entropies[accepted_mask]
    accepted_cold = implied_cold_distribution(
        accepted_probabilities, accepted_entropies, beta
    )
    return {
        "batch_id": batch_id,
        "proposal_count": len(uniforms),
        "entropy_domain_count": int(domain.sum()),
        "probability": accepted_probabilities,
        "entropy": accepted_entropies,
        "cold": accepted_cold,
        "minimum_margin": (
            float(minimum[accepted_mask].min()) if np.any(accepted_mask) else np.nan
        ),
    }


def sample_exact_simplex(
    cooled: np.ndarray,
    beta: float,
    use_entropies: bool,
    entropy_bound: float,
    n_samples: int,
    batch_power: int,
    max_proposals: int,
    seed: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    batch_size = 1 << batch_power
    accepted_probability: list[np.ndarray] = []
    accepted_entropy: list[np.ndarray] = []
    accepted_cold: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    total_accepted = 0
    total_proposals = 0
    next_batch = 0
    start = time.perf_counter()

    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        while total_accepted < n_samples and total_proposals < max_proposals:
            remaining_proposals = max_proposals - total_proposals
            remaining_batches = math.ceil(remaining_proposals / batch_size)
            group_size = min(workers, remaining_batches)
            tasks = [
                (
                    next_batch + offset,
                    batch_power,
                    min(
                        batch_size,
                        remaining_proposals - offset * batch_size,
                    ),
                    seed,
                    cooled,
                    beta,
                    use_entropies,
                    entropy_bound,
                )
                for offset in range(group_size)
            ]
            if executor is None:
                results = [exact_qmc_batch(task) for task in tasks]
            else:
                results = list(executor.map(exact_qmc_batch, tasks))
            for result in sorted(results, key=lambda item: item["batch_id"]):
                accepted_probability.append(result["probability"])
                accepted_entropy.append(result["entropy"])
                accepted_cold.append(result["cold"])
                count = len(result["probability"])
                total_accepted += count
                total_proposals += result["proposal_count"]
                diagnostics.append(
                    {
                        "batch_id": result["batch_id"],
                        "proposal_count": result["proposal_count"],
                        "entropy_domain_count": result["entropy_domain_count"],
                        "accepted_count": count,
                        "accepted_fraction": count / result["proposal_count"],
                        "minimum_accepted_margin": result["minimum_margin"],
                    }
                )
            next_batch += group_size
            print(
                f"  exact simplex: {total_accepted:,}/{n_samples:,} accepted "
                f"from {total_proposals:,} proposals",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown()

    if total_accepted < n_samples:
        raise RuntimeError(
            f"Exact simplex sampling found {total_accepted:,} of {n_samples:,} "
            f"requested points after {total_proposals:,} proposals. Increase "
            "--max-proposals, reduce --n-samples, or use --exact-method mcmc."
        )

    probability = np.vstack(accepted_probability)[:n_samples]
    entropy = np.vstack(accepted_entropy)[:n_samples]
    cold = np.vstack(accepted_cold)[:n_samples]
    metadata = {
        "total_proposals": total_proposals,
        "raw_accepted_count": total_accepted,
        "raw_acceptance_fraction": total_accepted / total_proposals,
        "elapsed_seconds": time.perf_counter() - start,
    }
    return probability, entropy, cold, pd.DataFrame(diagnostics), metadata


def mcmc_chain(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        chain_id,
        requested_samples,
        seed,
        anchor_probability,
        anchor_entropy,
        cooled,
        beta,
        use_entropies,
        entropy_bound,
        burn_in,
        thin,
    ) = task
    rng = np.random.default_rng(seed)
    state_count = len(anchor_probability)
    probability = anchor_probability.copy()
    raw_entropy = anchor_entropy - anchor_entropy[-1]
    p_step = 0.05 / state_count
    s_step = max(0.10 * entropy_bound, 1e-3)
    target_acceptance = 0.30
    tune_interval = 100

    probabilities = np.empty((requested_samples, state_count))
    entropies = np.empty((requested_samples, state_count))
    cold_probabilities = np.empty((requested_samples, state_count))
    p_accepted_total = 0
    s_accepted_total = 0
    p_accepted_sampling = 0
    s_accepted_sampling = 0
    p_window = 0
    s_window = 0
    saved = 0
    sweep = 0
    total_sweeps = burn_in + requested_samples * thin

    while sweep < total_sweeps:
        candidate_coordinates = probability[:-1] + rng.normal(
            0.0, p_step, state_count - 1
        )
        candidate_probability = unpack_probability_coordinates(candidate_coordinates)
        p_accept = False
        if np.all(candidate_probability > PROBABILITY_FLOOR):
            candidate_entropy = centered_entropy(raw_entropy)
            p_accept = bool(
                manuscript_constraint_margins(
                    candidate_probability, candidate_entropy, cooled, beta
                ).min()
                >= -ACCEPTANCE_TOLERANCE
            )
        if p_accept:
            probability = candidate_probability
            p_accepted_total += 1
            p_window += 1
            if sweep >= burn_in:
                p_accepted_sampling += 1

        if use_entropies:
            candidate_raw_entropy = raw_entropy.copy()
            candidate_raw_entropy[:-1] += rng.normal(0.0, s_step, state_count - 1)
            candidate_raw_entropy[-1] = 0.0
            s_accept = False
            if np.ptp(candidate_raw_entropy) <= 2.0 * entropy_bound:
                candidate_entropy = centered_entropy(candidate_raw_entropy)
                s_accept = bool(
                    manuscript_constraint_margins(
                        probability, candidate_entropy, cooled, beta
                    ).min()
                    >= -ACCEPTANCE_TOLERANCE
                )
            if s_accept:
                raw_entropy = candidate_raw_entropy
                s_accepted_total += 1
                s_window += 1
                if sweep >= burn_in:
                    s_accepted_sampling += 1

        sweep += 1
        if sweep <= burn_in and sweep % tune_interval == 0:
            p_rate = p_window / tune_interval
            p_step *= float(np.clip(np.exp(p_rate - target_acceptance), 0.5, 2.0))
            p_window = 0
            if use_entropies:
                s_rate = s_window / tune_interval
                s_step *= float(np.clip(np.exp(s_rate - target_acceptance), 0.5, 2.0))
                s_window = 0

        if sweep > burn_in and (sweep - burn_in) % thin == 0:
            entropy = centered_entropy(raw_entropy)
            probabilities[saved] = probability
            entropies[saved] = entropy
            cold_probabilities[saved] = implied_cold_distribution(
                probability, entropy, beta
            )
            saved += 1

    sampling_sweeps = requested_samples * thin
    return {
        "chain_id": chain_id,
        "probability": probabilities,
        "entropy": entropies,
        "cold": cold_probabilities,
        "p_acceptance": p_accepted_sampling / sampling_sweeps,
        "s_acceptance": (
            s_accepted_sampling / sampling_sweeps if use_entropies else np.nan
        ),
        "final_p_step": p_step,
        "final_s_step": s_step if use_entropies else np.nan,
        "total_p_acceptance": p_accepted_total / total_sweeps,
        "total_s_acceptance": (
            s_accepted_total / total_sweeps if use_entropies else np.nan
        ),
    }


def split_counts(total: int, parts: int) -> list[int]:
    base, remainder = divmod(total, parts)
    return [base + int(index < remainder) for index in range(parts)]


def split_r_hat(chain_arrays: list[np.ndarray]) -> np.ndarray:
    if len(chain_arrays) < 2:
        return np.full(chain_arrays[0].shape[1], np.nan)
    common = min(len(chain) for chain in chain_arrays)
    half = common // 2
    if half < 2:
        return np.full(chain_arrays[0].shape[1], np.nan)
    split = np.stack(
        [piece for chain in chain_arrays for piece in (chain[:half], chain[-half:])]
    )
    sample_count = split.shape[1]
    within = np.mean(np.var(split, axis=1, ddof=1), axis=0)
    between = sample_count * np.var(np.mean(split, axis=1), axis=0, ddof=1)
    variance = ((sample_count - 1.0) / sample_count) * within + between / sample_count
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(variance / within)


def finite_maximum_or_none(values: np.ndarray) -> float | None:
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    return float(values.max())


def sample_exact_mcmc(
    context: dict[str, Any],
    cooled: np.ndarray,
    n_samples: int,
    chains: int,
    burn_in: int,
    thin: int,
    max_iterations: int,
    cold_logit_weight: float,
    seed: int,
    workers: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    dict[str, Any],
    np.ndarray,
    np.ndarray | None,
]:
    start = time.perf_counter()
    chain_count = min(chains, n_samples)
    counts = split_counts(n_samples, chain_count)
    base_anchor = find_anchor(context, cooled, max_iterations, cold_logit_weight, seed)
    anchors = [base_anchor]
    rng = np.random.default_rng(seed + 1)
    for chain_id in range(1, chain_count):
        target_probability = rng.dirichlet(context["p_alpha"])
        target_entropy = (
            rng.uniform(
                -context["entropy_bound"],
                context["entropy_bound"],
                context["state_count"],
            )
            if context["use_entropies"]
            else np.zeros(context["state_count"])
        )
        candidate = project_target(
            cooled,
            target_probability,
            target_entropy,
            base_anchor["parameters"],
            context,
            max_iterations,
            cold_logit_weight,
        )
        anchors.append(candidate if candidate["valid"] else base_anchor)

    tasks = [
        (
            chain_id,
            counts[chain_id],
            int(np.random.SeedSequence([seed, chain_id]).generate_state(1)[0]),
            anchors[chain_id]["probability"],
            anchors[chain_id]["entropy"],
            cooled,
            context["beta"],
            context["use_entropies"],
            context["entropy_bound"],
            burn_in,
            thin,
        )
        for chain_id in range(chain_count)
    ]
    worker_count = min(workers, chain_count)
    if worker_count == 1:
        results = [mcmc_chain(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(mcmc_chain, tasks))
    results.sort(key=lambda item: item["chain_id"])

    probability_chains = [result["probability"] for result in results]
    entropy_chains = [result["entropy"] for result in results]
    probability = np.vstack(probability_chains)
    entropy = np.vstack(entropy_chains)
    cold = np.vstack([result["cold"] for result in results])
    probability_r_hat = split_r_hat(probability_chains)
    entropy_r_hat = split_r_hat(entropy_chains) if context["use_entropies"] else None
    diagnostics = pd.DataFrame(
        [
            {
                "chain_id": result["chain_id"],
                "saved_samples": len(result["probability"]),
                "burn_in_sweeps": burn_in,
                "thin": thin,
                "probability_acceptance_fraction": result["p_acceptance"],
                "entropy_acceptance_fraction": result["s_acceptance"],
                "final_probability_step": result["final_p_step"],
                "final_entropy_step": result["final_s_step"],
            }
            for result in results
        ]
    )
    maximum_probability_r_hat = finite_maximum_or_none(probability_r_hat)
    maximum_entropy_r_hat = (
        finite_maximum_or_none(entropy_r_hat) if entropy_r_hat is not None else None
    )
    if maximum_probability_r_hat is None or (
        context["use_entropies"] and maximum_entropy_r_hat is None
    ):
        warnings.warn(
            "Split-R-hat is unavailable; use at least two chains with four "
            "saved samples each before interpreting the MCMC cloud.",
            RuntimeWarning,
        )
    elif maximum_probability_r_hat > 1.05 or (
        maximum_entropy_r_hat is not None and maximum_entropy_r_hat > 1.05
    ):
        warnings.warn(
            "One or more split-R-hat values exceed 1.05. Increase burn-in, "
            "thinning, and/or the sample count before interpreting region summaries.",
            RuntimeWarning,
        )
    metadata = {
        "chain_count": chain_count,
        "burn_in_sweeps_per_chain": burn_in,
        "thin": thin,
        "maximum_probability_split_r_hat": maximum_probability_r_hat,
        "maximum_entropy_split_r_hat": maximum_entropy_r_hat,
        "nonfinite_probability_split_r_hat_count": int(
            np.count_nonzero(~np.isfinite(probability_r_hat))
        ),
        "nonfinite_entropy_split_r_hat_count": (
            int(np.count_nonzero(~np.isfinite(entropy_r_hat)))
            if entropy_r_hat is not None
            else 0
        ),
        "elapsed_seconds": time.perf_counter() - start,
    }
    return (
        probability,
        entropy,
        cold,
        diagnostics,
        metadata,
        probability_r_hat,
        entropy_r_hat,
    )


def projection_chunk(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        chunk_id,
        requested_samples,
        seed,
        context,
        anchor_parameters,
        max_iterations,
        max_attempt_factor,
        cold_logit_weight,
    ) = task
    rng = np.random.default_rng(np.random.SeedSequence([seed, chunk_id]))
    current = None if anchor_parameters is None else anchor_parameters.copy()
    probabilities: list[np.ndarray] = []
    entropies: list[np.ndarray] = []
    cold_probabilities: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    attempts = 0
    maximum_attempts = requested_samples * max_attempt_factor

    while len(probabilities) < requested_samples and attempts < maximum_attempts:
        attempts += 1
        proposal_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        proposal_rng = np.random.default_rng(proposal_seed)
        cooled = np.stack(
            [
                (
                    proposal_rng.dirichlet(alpha)
                    if alpha is not None
                    else context["cooled_mean"][index]
                )
                for index, alpha in enumerate(context["cooled_alpha"])
            ]
        )
        target_probability = proposal_rng.dirichlet(context["p_alpha"])
        target_entropy = (
            proposal_rng.uniform(
                -context["entropy_bound"],
                context["entropy_bound"],
                context["state_count"],
            )
            if context["use_entropies"]
            else np.zeros(context["state_count"])
        )

        best: dict[str, Any] | None = None
        if context["use_entropies"]:
            best = project_variable_target_radially(
                cooled,
                target_probability,
                target_entropy,
                context,
            )

        if best is None:
            if context["use_entropies"]:
                _, target_parameters = variable_target_parameters(
                    target_probability, target_entropy, context["beta"]
                )
            else:
                target_parameters = target_probability[:-1]
            initial_candidates = [
                current,
                anchor_parameters,
                target_parameters,
                context["p_center"][:-1] if not context["use_entropies"] else None,
                cooled[0, :-1] if not context["use_entropies"] else None,
            ]
            for retry, initial in enumerate(initial_candidates):
                if initial is None:
                    continue
                result = project_target(
                    cooled,
                    target_probability,
                    target_entropy,
                    initial,
                    context,
                    max_iterations,
                    cold_logit_weight,
                )
                if best is None or result["original_margin"] > best["original_margin"]:
                    best = result
                    best["projection_strategy"] = f"slsqp_start_{retry}"
                    best["radial_fraction"] = np.nan
                    best["linear_program_calls"] = 0
                if result["valid"]:
                    break
        if best is None or not best["valid"]:
            continue

        current = best["parameters"]
        probabilities.append(best["probability"])
        entropies.append(best["entropy"])
        cold_probabilities.append(best["cold"])
        diagnostics.append(
            {
                "chunk_id": chunk_id,
                "proposal_seed": proposal_seed,
                "chunk_attempt": attempts,
                "projection_strategy": best["projection_strategy"],
                "probability_radial_fraction": best["radial_fraction"],
                "linear_program_calls": best["linear_program_calls"],
                "optimizer_iterations": best["iterations"],
                "optimizer_success": best["optimizer_success"],
                "strategy_objective": best["objective"],
                "reduced_minimum_margin": best["reduced_margin"],
                "manuscript_minimum_margin": best["original_margin"],
                "entropy_range": float(np.ptp(best["entropy"])),
            }
        )

    if len(probabilities) != requested_samples:
        raise RuntimeError(
            f"Projection chunk {chunk_id} generated {len(probabilities)} of "
            f"{requested_samples} samples after {attempts} attempts. Increase "
            "--max-attempt-factor or inspect the input constraints."
        )
    return {
        "chunk_id": chunk_id,
        "probability": np.asarray(probabilities),
        "entropy": np.asarray(entropies),
        "cold": np.asarray(cold_probabilities),
        "diagnostics": diagnostics,
        "attempts": attempts,
    }


def sample_uncertain_projection(
    context: dict[str, Any],
    n_samples: int,
    seed: int,
    workers: int,
    max_iterations: int,
    max_attempt_factor: int,
    cold_logit_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    start = time.perf_counter()
    mean_anchor_available = False
    anchor_parameters: np.ndarray | None = None
    try:
        anchor = find_anchor(
            context,
            context["cooled_mean"],
            max_iterations,
            cold_logit_weight,
            seed,
        )
        mean_anchor_available = True
        anchor_parameters = anchor["parameters"]
    except RuntimeError:
        # Individual uncertainty realizations can be feasible even if the
        # vector of marginal means is not.  Each draw is therefore allowed to
        # establish its own feasible point below.
        pass
    chunk_counts = []
    remaining = n_samples
    while remaining:
        current = min(DEFAULT_PROJECTION_CHUNK_SIZE, remaining)
        chunk_counts.append(current)
        remaining -= current
    tasks = [
        (
            chunk_id,
            count,
            seed,
            context,
            anchor_parameters,
            max_iterations,
            max_attempt_factor,
            cold_logit_weight,
        )
        for chunk_id, count in enumerate(chunk_counts)
    ]

    worker_count = min(workers, len(tasks))
    results_by_chunk: dict[int, dict[str, Any]] = {}
    completed = 0
    next_report = 0.1
    if worker_count == 1:
        iterator = map(projection_chunk, tasks)
        for result in iterator:
            results_by_chunk[result["chunk_id"]] = result
            completed += len(result["probability"])
            if completed / n_samples >= next_report or completed == n_samples:
                print(
                    f"  uncertain projection: {completed:,}/{n_samples:,}", flush=True
                )
                next_report += 0.1
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(projection_chunk, task): task[0] for task in tasks
            }
            for future in as_completed(futures):
                result = future.result()
                results_by_chunk[result["chunk_id"]] = result
                completed += len(result["probability"])
                if completed / n_samples >= next_report or completed == n_samples:
                    print(
                        f"  uncertain projection: {completed:,}/{n_samples:,}",
                        flush=True,
                    )
                    next_report += 0.1

    results = [results_by_chunk[index] for index in range(len(tasks))]
    probability = np.vstack([result["probability"] for result in results])
    entropy = np.vstack([result["entropy"] for result in results])
    cold = np.vstack([result["cold"] for result in results])
    diagnostics = pd.DataFrame(
        [row for result in results for row in result["diagnostics"]]
    )
    diagnostics.insert(0, "sample_id", np.arange(len(diagnostics)))
    total_attempts = sum(result["attempts"] for result in results)
    metadata = {
        "total_proposals": total_attempts,
        "proposal_acceptance_fraction": n_samples / total_attempts,
        "projection_chunk_size": DEFAULT_PROJECTION_CHUNK_SIZE,
        "mean_cooled_trajectory_anchor_available": mean_anchor_available,
        "elapsed_seconds": time.perf_counter() - start,
    }
    return probability, entropy, cold, diagnostics, metadata


def validate_saved_samples(
    probabilities: np.ndarray,
    entropies: np.ndarray,
    cooled: np.ndarray,
    beta: float,
    fixed_cooled: bool,
) -> dict[str, float]:
    normalization_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))
    if np.any(probabilities <= 0.0):
        raise RuntimeError("A saved probability sample is nonpositive.")
    if normalization_error > 1e-9:
        raise RuntimeError("A saved probability sample is not normalized.")

    result = {"maximum_probability_normalization_error": normalization_error}
    if fixed_cooled:
        minimum = np.inf
        for start in range(0, len(probabilities), 10_000):
            stop = min(start + 10_000, len(probabilities))
            mask, margins = fixed_cooled_batch_mask(
                probabilities[start:stop],
                entropies[start:stop],
                cooled,
                beta,
            )
            if not np.all(mask):
                raise RuntimeError("A saved exact-data sample violates a constraint.")
            minimum = min(minimum, float(margins.min()))
        result["minimum_saved_manuscript_margin"] = minimum
    return result


def compute_point_estimate(
    probabilities: np.ndarray,
    entropies: np.ndarray,
    cooled_mean: np.ndarray,
    estimator: str,
    nearest_fraction: float,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    state_count = probabilities.shape[1]
    if estimator == "center":
        estimate = probabilities.mean(axis=0)
        standard_deviation = probabilities.std(axis=0, ddof=0)
        lower = np.full(state_count, np.nan)
        upper = np.full(state_count, np.nan)
        subset_indices = np.arange(len(probabilities))
        interval = "generated-sample standard deviation"
    else:
        reference = cooled_mean[0]
        denominator = np.maximum(reference, PROBABILITY_FLOOR)
        distances = np.linalg.norm((probabilities - reference) / denominator, axis=1)
        subset_count = max(1, int(math.ceil(nearest_fraction * len(probabilities))))
        subset_indices = np.argsort(distances, kind="stable")[:subset_count]
        subset = probabilities[subset_indices]
        estimate = np.median(subset, axis=0)
        estimate /= estimate.sum()
        standard_deviation = subset.std(axis=0, ddof=0)
        lower, upper = np.percentile(subset, [16.0, 84.0], axis=0)
        interval = "marginal 16th-84th percentiles"

    frame = pd.DataFrame(
        {
            "state_index": np.arange(state_count),
            "estimate": estimate,
            "standard_deviation": standard_deviation,
            "interval_lower": lower,
            "interval_upper": upper,
        }
    )
    metadata = {
        "estimator": estimator,
        "uncertainty_definition": interval,
        "selected_sample_count": int(len(subset_indices)),
        "selected_sample_fraction": float(len(subset_indices) / len(probabilities)),
    }
    return frame, subset_indices, metadata


def make_region_summary(
    states: tuple[str, ...],
    probabilities: np.ndarray,
    entropies: np.ndarray,
    cold: np.ndarray,
    use_entropies: bool,
    probability_r_hat: np.ndarray | None,
    entropy_r_hat: np.ndarray | None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "state_index": np.arange(len(states)),
            "state": states,
            "probability_mean": probabilities.mean(axis=0),
            "probability_std": probabilities.std(axis=0, ddof=0),
            "probability_median": np.median(probabilities, axis=0),
            "probability_p16": np.percentile(probabilities, 16.0, axis=0),
            "probability_p84": np.percentile(probabilities, 84.0, axis=0),
            "probability_min": probabilities.min(axis=0),
            "probability_max": probabilities.max(axis=0),
            "implied_cold_mean": cold.mean(axis=0),
            "implied_cold_std": cold.std(axis=0, ddof=0),
        }
    )
    if use_entropies:
        frame["entropy_mean"] = entropies.mean(axis=0)
        frame["entropy_std"] = entropies.std(axis=0, ddof=0)
        frame["entropy_median"] = np.median(entropies, axis=0)
        frame["entropy_p16"] = np.percentile(entropies, 16.0, axis=0)
        frame["entropy_p84"] = np.percentile(entropies, 84.0, axis=0)
        frame["entropy_min"] = entropies.min(axis=0)
        frame["entropy_max"] = entropies.max(axis=0)
    if probability_r_hat is not None:
        frame["probability_split_r_hat"] = probability_r_hat
    if use_entropies and entropy_r_hat is not None:
        frame["entropy_split_r_hat"] = entropy_r_hat
    return frame


def ensure_output_directory(path: Path, overwrite: bool) -> None:
    generated_names = {
        "allowed_samples.npz",
        "allowed_region_summary.csv",
        "hot_equilibrium_estimate.csv",
        "sampling_diagnostics.csv",
        "uncertainty_model_summary.csv",
        "run_metadata.json",
        "allowed_probability_samples.csv",
        "allowed_entropy_samples.csv",
        "implied_cold_probability_samples.csv",
    }
    if path.exists() and not overwrite:
        conflicts = sorted(name for name in generated_names if (path / name).exists())
        if conflicts:
            raise FileExistsError(
                f"Output files already exist in {path}: {conflicts}. "
                "Choose another directory or pass --overwrite."
            )
    path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in generated_names:
            target = path / name
            if target.is_file() or target.is_symlink():
                target.unlink()


def save_outputs(
    output_dir: Path,
    cooled: CoolingData,
    probabilities: np.ndarray,
    entropies: np.ndarray,
    cold: np.ndarray,
    region_summary: pd.DataFrame,
    estimate: pd.DataFrame,
    diagnostics: pd.DataFrame,
    uncertainty_model_summary: pd.DataFrame,
    cooled_alpha: list[np.ndarray | None],
    metadata: dict[str, Any],
    selected_indices: np.ndarray,
    write_sample_csv: bool,
) -> None:
    estimate = estimate.copy()
    estimate.insert(1, "state", cooled.states)
    np.savez_compressed(
        output_dir / "allowed_samples.npz",
        probabilities=probabilities,
        entropies=entropies,
        implied_cold_probabilities=cold,
        state_labels=np.asarray(cooled.states),
        cooling_times=cooled.times,
        cooled_mean_probabilities=cooled.probabilities,
        cooled_ci_lower_95=cooled.ci_lower,
        cooled_ci_upper_95=cooled.ci_upper,
        cooled_dirichlet_alpha=np.stack(
            [
                np.full(cooled.state_count, np.nan) if alpha is None else alpha
                for alpha in cooled_alpha
            ]
        ),
        point_estimator_subset_indices=selected_indices,
    )
    region_summary.to_csv(output_dir / "allowed_region_summary.csv", index=False)
    estimate.to_csv(output_dir / "hot_equilibrium_estimate.csv", index=False)
    diagnostics.to_csv(output_dir / "sampling_diagnostics.csv", index=False)
    if not uncertainty_model_summary.empty:
        uncertainty_model_summary.to_csv(
            output_dir / "uncertainty_model_summary.csv", index=False
        )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if write_sample_csv:
        columns = list(cooled.states)
        pd.DataFrame(probabilities, columns=columns).to_csv(
            output_dir / "allowed_probability_samples.csv", index=False
        )
        pd.DataFrame(cold, columns=columns).to_csv(
            output_dir / "implied_cold_probability_samples.csv", index=False
        )
        if metadata["microstate_entropies_enabled"]:
            pd.DataFrame(entropies, columns=columns).to_csv(
                output_dir / "allowed_entropy_samples.csv", index=False
            )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(effective_argv)
    validate_args(args)
    cooled_path = args.cooled.resolve()
    output_dir = args.output_dir.resolve()
    cooled = load_cooled_csv(cooled_path)
    ensure_output_directory(output_dir, args.overwrite)

    beta = args.hot_temperature / args.cold_temperature
    entropy_bound = (
        math.log(cooled.state_count)
        if args.entropy_bound is None
        else args.entropy_bound
    )
    if args.microstate_entropies and entropy_bound == 0.0:
        warnings.warn(
            "An entropy bound of zero is equivalent to --no-microstate-entropies.",
            RuntimeWarning,
        )
        use_entropies = False
    else:
        use_entropies = args.microstate_entropies

    context = make_projection_context(
        cooled,
        beta,
        entropy_bound,
        use_entropies,
        args.proposal_concentration,
    )
    uncertainty_model_summary = make_uncertainty_model_summary(cooled, context)
    if not uncertainty_model_summary.empty:
        width_ratios = uncertainty_model_summary[
            "modeled_to_input_width_ratio"
        ].to_numpy()
        if np.any((width_ratios < 0.5) | (width_ratios > 2.0)):
            warnings.warn(
                "A single Dirichlet distribution cannot closely match every "
                "supplied marginal confidence interval. Review "
                "uncertainty_model_summary.csv after the run.",
                RuntimeWarning,
            )
    exact_data = not cooled.has_uncertainty
    probability_r_hat: np.ndarray | None = None
    entropy_r_hat: np.ndarray | None = None
    selected_max_proposals: int | None = None
    selected_chain_count: int | None = None
    print(
        f"Loaded {len(cooled.times)} cooling distributions over "
        f"{cooled.state_count} states ({'exact' if exact_data else 'uncertain'} input).",
        flush=True,
    )

    if exact_data:
        exact_method = args.exact_method
        if exact_method == "auto":
            exact_method = (
                "simplex" if cooled.state_count <= args.simplex_threshold else "mcmc"
            )
        if exact_method == "simplex":
            max_proposals = args.max_proposals or max(1 << 24, args.n_samples * 10_000)
            selected_max_proposals = max_proposals
            print("Sampling the exact-data region by scrambled-simplex rejection...")
            (
                probabilities,
                entropies,
                cold,
                diagnostics,
                sampling_metadata,
            ) = sample_exact_simplex(
                cooled.probabilities,
                beta,
                use_entropies,
                entropy_bound,
                args.n_samples,
                args.qmc_batch_power,
                max_proposals,
                args.seed,
                args.workers,
            )
            method = "exact_simplex_qmc_rejection"
        else:
            chain_count = args.mcmc_chains
            selected_chain_count = min(chain_count, args.n_samples)
            print(
                "Sampling the exact-data region with uniform-target random-walk MCMC..."
            )
            (
                probabilities,
                entropies,
                cold,
                diagnostics,
                sampling_metadata,
                probability_r_hat,
                entropy_r_hat,
            ) = sample_exact_mcmc(
                context,
                cooled.probabilities,
                args.n_samples,
                chain_count,
                args.mcmc_burn_in,
                args.mcmc_thin,
                args.max_iterations,
                DEFAULT_COLD_LOGIT_WEIGHT,
                args.seed,
                args.workers,
            )
            method = "exact_constraints_uniform_target_mcmc"
    else:
        if args.exact_method != "auto":
            warnings.warn(
                "--exact-method is ignored because confidence intervals are present.",
                RuntimeWarning,
            )
        print("Sampling the uncertainty-aware projected feasible cloud...")
        (
            probabilities,
            entropies,
            cold,
            diagnostics,
            sampling_metadata,
        ) = sample_uncertain_projection(
            context,
            args.n_samples,
            args.seed,
            args.workers,
            args.max_iterations,
            args.max_attempt_factor,
            DEFAULT_COLD_LOGIT_WEIGHT,
        )
        method = "uncertainty_aware_random_projection"

    validation_metadata = validate_saved_samples(
        probabilities,
        entropies,
        cooled.probabilities,
        beta,
        fixed_cooled=exact_data,
    )
    estimator = args.estimator
    estimate, selected_indices, estimator_metadata = compute_point_estimate(
        probabilities,
        entropies,
        cooled.probabilities,
        estimator,
        args.nearest_fraction,
    )
    region_summary = make_region_summary(
        cooled.states,
        probabilities,
        entropies,
        cold,
        use_entropies,
        probability_r_hat,
        entropy_r_hat,
    )

    metadata: dict[str, Any] = {
        "method": method,
        "scientific_interpretation": (
            "Exact-input methods sample a fixed thermodynamic feasible region. "
            "The uncertainty-aware method produces proposal-dependent random "
            "projections and is neither a uniform feasible-region sample nor a "
            "Bayesian posterior."
        ),
        "uncertainty_projection": (
            {
                "probability_target_center": "fastest cooled mean distribution",
                "variable_entropy_method": (
                    "bounded L1 entropy linear program followed, when needed, "
                    "by an anchor-dependent coarse radial scan and local refinement"
                    if use_entropies
                    else None
                ),
                "zero_entropy_method": (
                    "nonlinear constrained projection" if not use_entropies else None
                ),
                "global_nearest_projection_guaranteed": False,
            }
            if cooled.has_uncertainty
            else None
        ),
        "thermodynamic_constraints": {
            "manuscript_equations": [2, 3, 8, 9, 11, 12],
            "second_law": [
                "Sigma_cooling_upper_bound(tau) >= 0",
                "Sigma_auxiliary(tau) >= 0",
            ],
            "cooling_time_order": (
                "ascending numeric cooling_time (smallest is fastest); "
                "numeric magnitudes are otherwise unused by inference"
            ),
            "monotonicity": [
                "Sigma_cooling_upper_bound nondecreasing with cooling_time",
                "Sigma_auxiliary nonincreasing with cooling_time",
            ],
            "extra_probability_monotonicity": False,
        },
        "cooled_input_file": path_for_metadata(cooled_path),
        "cooled_input_sha256": sha256_file(cooled_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "cooling_times": cooled.times.tolist(),
        "state_labels": list(cooled.states),
        "state_count": cooled.state_count,
        "cooling_distribution_count": len(cooled.times),
        "uncertain_cooling_runs": cooled.uncertain_runs.tolist(),
        "confidence_interval_model": (
            "Dirichlet approximation from total marginal 95% interval widths; "
            "asymmetry is not otherwise modeled"
            if cooled.has_uncertainty
            else None
        ),
        "accepted_cooled_draw_replay": (
            {
                "schema_version": 1,
                "generator": "numpy.random.default_rng",
                "seed_column": "sampling_diagnostics.csv:proposal_seed",
                "draw_order": (
                    "ascending cooling_time; one Dirichlet draw for each uncertain "
                    "run; exact runs consume no random draw"
                ),
                "numpy_version": np.__version__,
            }
            if cooled.has_uncertainty
            else None
        ),
        "confidence_interval_fit": (
            {
                "minimum_modeled_to_input_width_ratio": float(
                    uncertainty_model_summary["modeled_to_input_width_ratio"].min()
                ),
                "maximum_modeled_to_input_width_ratio": float(
                    uncertainty_model_summary["modeled_to_input_width_ratio"].max()
                ),
                "details_file": "uncertainty_model_summary.csv",
            }
            if not uncertainty_model_summary.empty
            else None
        ),
        "hot_temperature": args.hot_temperature,
        "cold_temperature": args.cold_temperature,
        "temperature_ratio_beta": beta,
        "microstate_entropies_enabled": use_entropies,
        "entropy_units": "k_B",
        "entropy_bound": entropy_bound if use_entropies else 0.0,
        "entropy_gauge": (
            "minimum and maximum centered symmetrically about zero"
            if use_entropies
            else "all zero"
        ),
        "n_samples": len(probabilities),
        "seed": args.seed,
        "workers": args.workers,
        "settings": {
            "requested_exact_method": args.exact_method,
            "simplex_threshold": args.simplex_threshold,
            "qmc_batch_power": args.qmc_batch_power,
            "maximum_simplex_proposals": selected_max_proposals,
            "mcmc_chains": selected_chain_count,
            "mcmc_burn_in": args.mcmc_burn_in,
            "mcmc_thin": args.mcmc_thin,
            "proposal_concentration": args.proposal_concentration,
            "maximum_projection_iterations": args.max_iterations,
            "maximum_projection_attempt_factor": args.max_attempt_factor,
            "constraint_buffer": CONSTRAINT_BUFFER,
            "acceptance_tolerance": ACCEPTANCE_TOLERANCE,
            "cold_logit_weight": DEFAULT_COLD_LOGIT_WEIGHT,
            "nearest_fraction": args.nearest_fraction,
            "write_sample_csv": args.write_sample_csv,
        },
        "estimator": estimator_metadata,
        "sampling": sampling_metadata,
        "validation": validation_metadata,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "command": ["python", Path(__file__).name, *effective_argv],
    }
    save_outputs(
        output_dir,
        cooled,
        probabilities,
        entropies,
        cold,
        region_summary,
        estimate,
        diagnostics,
        uncertainty_model_summary,
        context["cooled_alpha"],
        metadata,
        selected_indices,
        args.write_sample_csv,
    )
    print(f"Saved {len(probabilities):,} samples to {output_dir}")
    print(f"Point estimator: {estimator}")


if __name__ == "__main__":
    main()
