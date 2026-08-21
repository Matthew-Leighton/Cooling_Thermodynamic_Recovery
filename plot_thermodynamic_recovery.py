#!/usr/bin/env python3
"""Plot thermodynamic-recovery results produced by thermodynamic_inference.py.

The command reads the original cooled-distribution CSV and an inference output
directory.  Optional hot- and cold-equilibrium CSVs are used only for plotting;
they never affect the inference itself.

Run ``python plot_thermodynamic_recovery.py --help`` for usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

# Keep plotting caches in a writable location on shared or sandboxed systems.
_PLOT_CACHE = Path(tempfile.gettempdir()) / "thermodynamic_inference_plot_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.path import Path as MarkerPath
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter, NullLocator
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.special import xlogy


CI_Z_VALUE = 1.96
PROBABILITY_FLOOR = 1e-12
DEFAULT_SEED = 20_260_820
DEFAULT_UNCERTAINTY_DRAWS = 10_000
DEFAULT_MAX_HISTOGRAM_STATES = 10
THERMODYNAMIC_CHUNK_SIZE = 2_000
EXACT_REGION_BINS = 150
LIKELY_REGION_BINS = 180
LIKELY_REGION_SMOOTHING_SIGMA = 2.5
LIKELY_REGION_MASS_FRACTION = 0.99
LIKELY_REGION_DISPLAY_QUANTILES = (0.005, 0.995)
LIKELY_REGION_CONTOUR_LEVELS = 48


def make_snowflake_marker() -> MarkerPath:
    """Return the six-arm cold-equilibrium marker used in Figure 6."""
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    for arm in range(6):
        angle = arm * np.pi / 3.0
        tip = (np.cos(angle), np.sin(angle))
        vertices.extend([(0.0, 0.0), tip])
        codes.extend([MarkerPath.MOVETO, MarkerPath.LINETO])
        branch_base = 0.55 * np.asarray(tip)
        for sign in (-1.0, 1.0):
            branch_angle = angle + sign * np.pi / 6.0
            branch_tip = branch_base + 0.35 * np.array(
                [np.cos(branch_angle), np.sin(branch_angle)]
            )
            vertices.extend([tuple(branch_base), tuple(branch_tip)])
            codes.extend([MarkerPath.MOVETO, MarkerPath.LINETO])
    return MarkerPath(np.asarray(vertices), codes)


SNOWFLAKE_MARKER = make_snowflake_marker()


@dataclass(frozen=True)
class CoolingData:
    """Cooled distributions in the inference script's long-form schema."""

    times: np.ndarray
    states: tuple[str, ...]
    probabilities: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    uncertain_runs: np.ndarray


@dataclass(frozen=True)
class EquilibriumData:
    """One optional equilibrium distribution."""

    probabilities: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    has_uncertainty: bool


@dataclass(frozen=True)
class InferenceResults:
    """Arrays and summaries saved by thermodynamic_inference.py."""

    probabilities: np.ndarray
    entropies: np.ndarray
    implied_cold: np.ndarray
    cooled_alpha: np.ndarray
    selected_indices: np.ndarray
    estimate: np.ndarray
    estimate_lower: np.ndarray
    estimate_upper: np.ndarray
    metadata: dict
    diagnostics: pd.DataFrame


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
            "Long-form cooled CSV used by thermodynamic_inference.py: "
            "cooling_time,state,probability and optional 95%% interval columns."
        ),
    )
    parser.add_argument(
        "--inference-dir",
        type=Path,
        required=True,
        help="Directory containing the outputs of thermodynamic_inference.py.",
    )
    parser.add_argument(
        "--hot-equilibrium",
        type=Path,
        default=None,
        help=(
            "Optional CSV with state,probability and optional "
            "ci_lower_95,ci_upper_95 columns."
        ),
    )
    parser.add_argument(
        "--cold-equilibrium",
        type=Path,
        default=None,
        help=(
            "Optional CSV with state,probability and optional "
            "ci_lower_95,ci_upper_95 columns."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("thermodynamic_recovery_figures"),
        help="Directory for generated figures and numeric plot data.",
    )
    parser.add_argument(
        "--format",
        choices=("pdf", "png", "svg"),
        default="pdf",
        help="Figure file format.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for raster output and saved figure rendering.",
    )
    parser.add_argument(
        "--slice-states",
        nargs=2,
        metavar=("X_STATE", "Y_STATE"),
        default=None,
        help=(
            "State labels for the Figure 1 slice. By default, use the two "
            "largest categories in the slowest cooled distribution."
        ),
    )
    parser.add_argument(
        "--max-histogram-states",
        type=int,
        default=DEFAULT_MAX_HISTOGRAM_STATES,
        help="Maximum number of states shown in Figure 2.",
    )
    parser.add_argument(
        "--uncertainty-draws",
        type=int,
        default=DEFAULT_UNCERTAINTY_DRAWS,
        help=(
            "Monte Carlo draws used for cooled-distribution uncertainty in Figure 4."
        ),
    )
    parser.add_argument(
        "--time-unit",
        type=str,
        default=None,
        help="Optional unit appended to the cooling-time axis label.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for plotting uncertainty draws.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace plot files already present in output-dir.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if args.max_histogram_states <= 0:
        raise ValueError("--max-histogram-states must be positive.")
    if args.uncertainty_draws <= 0:
        raise ValueError("--uncertainty-draws must be positive.")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative.")
    if args.slice_states is not None and args.slice_states[0] == args.slice_states[1]:
        raise ValueError("--slice-states must name two distinct states.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_probability_columns(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    lower_name = "ci_lower_95"
    upper_name = "ci_upper_95"
    has_lower = lower_name in frame.columns
    has_upper = upper_name in frame.columns
    if has_lower != has_upper:
        raise ValueError(
            "Confidence intervals require both ci_lower_95 and ci_upper_95."
        )

    probability = pd.to_numeric(frame["probability"], errors="raise").to_numpy(float)
    if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError("Probabilities must be finite and nonnegative.")
    total = float(probability.sum())
    if not np.isclose(total, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"A probability distribution sums to {total:.12g}, not 1.")
    probability = probability / total

    if not has_lower:
        lower = np.full(len(frame), np.nan)
        upper = np.full(len(frame), np.nan)
        return probability, lower, upper, False

    lower = pd.to_numeric(frame[lower_name], errors="raise").to_numpy(float)
    upper = pd.to_numeric(frame[upper_name], errors="raise").to_numpy(float)
    present = np.isfinite(lower) | np.isfinite(upper)
    uncertain = bool(np.any(present))
    if not uncertain:
        return probability, np.full(len(frame), np.nan), np.full(len(frame), np.nan), False
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise ValueError("A distribution has an incomplete confidence interval.")
    if np.any(lower < 0.0) or np.any(upper > 1.0):
        raise ValueError("Confidence bounds must lie between 0 and 1.")
    if np.any(lower > probability) or np.any(probability > upper):
        raise ValueError("Each probability must lie inside its confidence interval.")
    if np.any(upper <= lower):
        raise ValueError("Every confidence interval must have positive width.")
    return probability, lower, upper, True


def load_cooled_csv(path: Path) -> CoolingData:
    frame = pd.read_csv(path)
    required = {"cooling_time", "state", "probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing cooled CSV columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("The cooled CSV contains no rows.")

    frame = frame.copy()
    frame["cooling_time"] = pd.to_numeric(frame["cooling_time"], errors="raise")
    if not np.all(np.isfinite(frame["cooling_time"])) or np.any(
        frame["cooling_time"] < 0.0
    ):
        raise ValueError("Cooling times must be finite and nonnegative.")
    if frame["state"].isna().any():
        raise ValueError("State labels cannot be blank or missing.")
    frame["state"] = frame["state"].astype(str).str.strip()
    if (frame["state"] == "").any():
        raise ValueError("State labels cannot be blank or missing.")
    if frame.duplicated(["cooling_time", "state"]).any():
        raise ValueError("Each cooling_time/state pair must be unique.")

    times = np.sort(frame["cooling_time"].unique().astype(float))
    states = tuple(pd.unique(frame["state"]).tolist())
    if len(states) < 2:
        raise ValueError("At least two states are required.")
    expected_states = set(states)
    probabilities = []
    lower_rows = []
    upper_rows = []
    uncertain_runs = []
    for cooling_time in times:
        block = frame.loc[frame["cooling_time"] == cooling_time].set_index("state")
        if set(block.index) != expected_states or len(block) != len(states):
            raise ValueError(
                "Every cooling time must contain exactly the same set of states."
            )
        block = block.loc[list(states)]
        probability, lower, upper, uncertain = _read_probability_columns(block)
        probabilities.append(probability)
        lower_rows.append(lower)
        upper_rows.append(upper)
        uncertain_runs.append(uncertain)

    return CoolingData(
        times=times,
        states=states,
        probabilities=np.asarray(probabilities),
        ci_lower=np.asarray(lower_rows),
        ci_upper=np.asarray(upper_rows),
        uncertain_runs=np.asarray(uncertain_runs, dtype=bool),
    )


def load_equilibrium_csv(path: Path, states: tuple[str, ...]) -> EquilibriumData:
    frame = pd.read_csv(path)
    required = {"state", "probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing equilibrium CSV columns in {path}: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"The equilibrium CSV {path} contains no rows.")
    if frame["state"].isna().any():
        raise ValueError(f"State labels cannot be missing in {path}.")
    frame = frame.copy()
    frame["state"] = frame["state"].astype(str).str.strip()
    if (frame["state"] == "").any() or frame["state"].duplicated().any():
        raise ValueError(f"State labels in {path} must be nonblank and unique.")
    if set(frame["state"]) != set(states) or len(frame) != len(states):
        missing_states = sorted(set(states) - set(frame["state"]))
        extra_states = sorted(set(frame["state"]) - set(states))
        raise ValueError(
            f"Equilibrium states in {path} do not match inference states; "
            f"missing={missing_states}, extra={extra_states}."
        )
    frame = frame.set_index("state").loc[list(states)]
    probability, lower, upper, uncertain = _read_probability_columns(frame)
    return EquilibriumData(probability, lower, upper, uncertain)


def _require_npz_array(archive: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    if name not in archive.files:
        raise ValueError(f"allowed_samples.npz is missing the array {name!r}.")
    return np.asarray(archive[name])


def load_inference_results(
    inference_dir: Path, cooled: CoolingData, cooled_path: Path
) -> InferenceResults:
    npz_path = inference_dir / "allowed_samples.npz"
    estimate_path = inference_dir / "hot_equilibrium_estimate.csv"
    metadata_path = inference_dir / "run_metadata.json"
    diagnostics_path = inference_dir / "sampling_diagnostics.csv"
    for path in (npz_path, estimate_path, metadata_path, diagnostics_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required inference output not found: {path}")

    with np.load(npz_path, allow_pickle=False) as archive:
        probabilities = _require_npz_array(archive, "probabilities").astype(float)
        entropies = _require_npz_array(archive, "entropies").astype(float)
        implied_cold = _require_npz_array(
            archive, "implied_cold_probabilities"
        ).astype(float)
        state_labels = tuple(
            str(value) for value in _require_npz_array(archive, "state_labels").tolist()
        )
        cooling_times = _require_npz_array(archive, "cooling_times").astype(float)
        cooled_mean = _require_npz_array(
            archive, "cooled_mean_probabilities"
        ).astype(float)
        cooled_lower = _require_npz_array(archive, "cooled_ci_lower_95").astype(float)
        cooled_upper = _require_npz_array(archive, "cooled_ci_upper_95").astype(float)
        cooled_alpha = _require_npz_array(
            archive, "cooled_dirichlet_alpha"
        ).astype(float)
        selected_indices = _require_npz_array(
            archive, "point_estimator_subset_indices"
        ).astype(np.int64)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    diagnostics = pd.read_csv(diagnostics_path)

    sample_count = probabilities.shape[0]
    state_count = len(cooled.states)
    expected_sample_shape = (sample_count, state_count)
    if probabilities.ndim != 2 or probabilities.shape[1] != state_count:
        raise ValueError("Inference probability samples have an unexpected shape.")
    if entropies.shape != expected_sample_shape or implied_cold.shape != expected_sample_shape:
        raise ValueError("Inference entropy or implied-cold samples have an unexpected shape.")
    if state_labels != cooled.states:
        raise ValueError(
            "State labels or their order differ between --cooled and the inference output."
        )
    if cooling_times.shape != cooled.times.shape or not np.allclose(
        cooling_times, cooled.times, atol=1e-12, rtol=0.0
    ):
        raise ValueError("Cooling times differ between --cooled and the inference output.")
    if cooled_mean.shape != cooled.probabilities.shape or not np.allclose(
        cooled_mean, cooled.probabilities, atol=1e-10, rtol=0.0
    ):
        raise ValueError("Cooled means differ between --cooled and the inference output.")
    if not np.allclose(cooled_lower, cooled.ci_lower, atol=1e-10, rtol=0.0, equal_nan=True):
        raise ValueError("Cooled lower intervals differ between input and inference output.")
    if not np.allclose(cooled_upper, cooled.ci_upper, atol=1e-10, rtol=0.0, equal_nan=True):
        raise ValueError("Cooled upper intervals differ between input and inference output.")
    if cooled_alpha.shape != cooled.probabilities.shape:
        raise ValueError("Saved cooled Dirichlet parameters have an unexpected shape.")
    if selected_indices.ndim != 1 or len(selected_indices) == 0:
        raise ValueError("The saved point-estimator subset is empty or malformed.")
    if np.any(selected_indices < 0) or np.any(selected_indices >= sample_count):
        raise ValueError("The saved point-estimator subset contains an invalid sample index.")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
        raise ValueError("Inference probability samples must be finite and positive.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("Inference probability samples are not normalized.")
    if not np.all(np.isfinite(entropies)):
        raise ValueError("Inference entropy samples contain nonfinite values.")
    if not np.all(np.isfinite(implied_cold)) or np.any(implied_cold <= 0.0):
        raise ValueError("Inference implied-cold samples must be finite and positive.")
    if not np.allclose(implied_cold.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("Inference implied-cold samples are not normalized.")
    for run_index, uncertain in enumerate(cooled.uncertain_runs):
        alpha = cooled_alpha[run_index]
        if uncertain and (not np.all(np.isfinite(alpha)) or np.any(alpha <= 0.0)):
            raise ValueError(
                "An uncertain cooling run is missing valid saved Dirichlet parameters."
            )

    recorded_hash = metadata.get("cooled_input_sha256")
    if recorded_hash and sha256_file(cooled_path) != recorded_hash:
        warnings.warn(
            "The supplied cooled CSV has the same numerical content as the saved "
            "arrays, but its SHA-256 hash differs from the file used for inference.",
            RuntimeWarning,
        )

    estimate_frame = pd.read_csv(estimate_path)
    required_estimate = {"state", "estimate", "standard_deviation", "interval_lower", "interval_upper"}
    missing_estimate = required_estimate - set(estimate_frame.columns)
    if missing_estimate:
        raise ValueError(
            f"hot_equilibrium_estimate.csv is missing columns: {sorted(missing_estimate)}"
        )
    estimate_frame["state"] = estimate_frame["state"].astype(str)
    if set(estimate_frame["state"]) != set(cooled.states):
        raise ValueError("The point-estimate state labels do not match the cooled input.")
    estimate_frame = estimate_frame.set_index("state").loc[list(cooled.states)]
    estimate = estimate_frame["estimate"].to_numpy(float)
    standard_deviation = estimate_frame["standard_deviation"].to_numpy(float)
    interval_lower = estimate_frame["interval_lower"].to_numpy(float)
    interval_upper = estimate_frame["interval_upper"].to_numpy(float)
    if not np.all(np.isfinite(estimate)) or np.any(estimate < 0.0):
        raise ValueError("The inferred point estimate is invalid.")
    if not np.isclose(estimate.sum(), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("The inferred point estimate is not normalized.")

    use_percentiles = np.isfinite(interval_lower) & np.isfinite(interval_upper)
    estimate_lower = np.where(use_percentiles, interval_lower, estimate - standard_deviation)
    estimate_upper = np.where(use_percentiles, interval_upper, estimate + standard_deviation)
    estimate_lower = np.clip(estimate_lower, 0.0, 1.0)
    estimate_upper = np.clip(estimate_upper, 0.0, 1.0)
    # The nearest-median estimator normalizes its componentwise median after
    # computing marginal percentiles. That normalized point need not lie inside
    # every raw marginal interval, so expand display endpoints to include it.
    estimate_lower = np.minimum(estimate_lower, estimate)
    estimate_upper = np.maximum(estimate_upper, estimate)

    return InferenceResults(
        probabilities=probabilities,
        entropies=entropies,
        implied_cold=implied_cold,
        cooled_alpha=cooled_alpha,
        selected_indices=selected_indices,
        estimate=estimate,
        estimate_lower=estimate_lower,
        estimate_upper=estimate_upper,
        metadata=metadata,
        diagnostics=diagnostics,
    )


def configure_style(figure_format: str) -> None:
    required_commands = ["latex"]
    if figure_format == "png":
        required_commands.append("dvipng")
    missing_commands = [
        command for command in required_commands if shutil.which(command) is None
    ]
    if missing_commands:
        formatted = ", ".join(missing_commands)
        raise RuntimeError(
            "LaTeX rendering is required for the recovery figures, but these "
            f"commands were not found: {formatted}. Install a TeX distribution "
            "and ensure its binaries are on PATH."
        )
    matplotlib.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "savefig.transparent": False,
        }
    )


def latex_escape_text(value: str) -> str:
    """Escape a user-supplied label for LaTeX text mode."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def format_axes(ax: plt.Axes, categorical_x: bool = False) -> None:
    if categorical_x:
        ax.xaxis.set_minor_locator(NullLocator())
    elif ax.get_xscale() == "log":
        ax.xaxis.set_minor_locator(LogLocator(subs="auto"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    if ax.get_yscale() == "log":
        ax.yaxis.set_minor_locator(LogLocator(subs="auto"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(
        which="major",
        width=0.8,
        length=4.5,
        direction="in",
        top=not categorical_x,
        right=True,
    )
    ax.tick_params(
        which="minor",
        width=0.6,
        length=2.25,
        direction="in",
        top=not categorical_x,
        right=True,
    )
    if categorical_x:
        ax.tick_params(axis="x", which="both", bottom=False, top=False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)


def finite_linear_limits(
    values: Iterable[np.ndarray | float],
    *,
    pad_fraction: float = 0.05,
    lower_clip: float | None = None,
    upper_clip: float | None = None,
) -> tuple[float, float]:
    pieces = []
    for value in values:
        array = np.asarray(value, dtype=float).ravel()
        pieces.append(array[np.isfinite(array)])
    finite = np.concatenate(pieces) if pieces else np.empty(0)
    if finite.size == 0:
        raise ValueError("Cannot determine axis bounds from nonfinite data.")
    lower = float(finite.min())
    upper = float(finite.max())
    span = upper - lower
    if span <= np.finfo(float).eps * max(1.0, abs(lower), abs(upper)):
        span = max(0.02, 0.10 * max(abs(lower), abs(upper), 0.1))
    lower -= pad_fraction * span
    upper += pad_fraction * span
    if lower_clip is not None:
        lower = max(lower_clip, lower)
    if upper_clip is not None:
        upper = min(upper_clip, upper)
    if not lower < upper:
        center = float(finite.mean())
        half_width = max(0.01, 0.05 * max(abs(center), 1.0))
        lower = center - half_width
        upper = center + half_width
        if lower_clip is not None:
            lower = max(lower_clip, lower)
        if upper_clip is not None:
            upper = min(upper_clip, upper)
    return lower, upper


def finite_log_limits(values: Iterable[np.ndarray | float]) -> tuple[float, float]:
    pieces = []
    for value in values:
        array = np.asarray(value, dtype=float).ravel()
        pieces.append(array[np.isfinite(array) & (array > 0.0)])
    finite = np.concatenate(pieces) if pieces else np.empty(0)
    if finite.size == 0:
        raise ValueError("Cannot determine logarithmic bounds without positive data.")
    log_lower = float(np.log10(finite.min()))
    log_upper = float(np.log10(finite.max()))
    span = log_upper - log_lower
    if span < 1e-12:
        span = 0.4
    return 10.0 ** (log_lower - 0.05 * span), 10.0 ** (
        log_upper + 0.05 * span
    )


def set_adaptive_time_axis(ax: plt.Axes, times: np.ndarray, unit: str | None) -> None:
    times = np.asarray(times, dtype=float)
    use_log = bool(
        np.all(times > 0.0)
        and len(times) > 1
        and float(times.max() / times.min()) >= 20.0
    )
    if use_log:
        ax.set_xscale("log")
        ax.set_xlim(*finite_log_limits([times]))
    else:
        ax.set_xlim(*finite_linear_limits([times], pad_fraction=0.05))
    label = (
        "Cooling time"
        if unit is None
        else f"Cooling time ({latex_escape_text(unit)})"
    )
    ax.set_xlabel(label)


def error_endpoints(
    mean: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    low_endpoint = np.where(np.isfinite(lower), lower, mean)
    high_endpoint = np.where(np.isfinite(upper), upper, mean)
    return low_endpoint, high_endpoint


def errorbar_amounts(
    mean: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    low_endpoint, high_endpoint = error_endpoints(mean, lower, upper)
    return np.vstack(
        [
            np.clip(mean - low_endpoint, 0.0, None),
            np.clip(high_endpoint - mean, 0.0, None),
        ]
    )


def place_adaptive_legend(ax: plt.Axes, *, force_outside: bool = False) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if force_outside or len(handles) >= 5:
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            ncol=1,
            frameon=False,
            borderaxespad=0.0,
        )
    else:
        ax.legend(handles, labels, loc="best", frameon=False)


def format_value(value: float) -> str:
    return f"{value:g}"


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def choose_slice_indices(
    cooled: CoolingData, slice_states: list[str] | None
) -> tuple[int, int]:
    if slice_states is None:
        order = np.argsort(-cooled.probabilities[-1], kind="stable")
        return int(order[0]), int(order[1])
    missing = [state for state in slice_states if state not in cooled.states]
    if missing:
        raise ValueError(f"--slice-states contains unknown labels: {missing}")
    return cooled.states.index(slice_states[0]), cooled.states.index(slice_states[1])


def projection_display_range(values: np.ndarray, exact_region: bool) -> np.ndarray:
    """Return the full exact range or robust uncertain-projection endpoints."""
    values = np.asarray(values, dtype=float)
    if exact_region:
        return values
    return np.quantile(values, LIKELY_REGION_DISPLAY_QUANTILES)


def highest_density_threshold(density: np.ndarray, mass_fraction: float) -> float:
    """Return the density cutoff containing the requested gridded mass."""
    positive = np.asarray(density, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        return np.nan
    ordered = np.sort(positive)[::-1]
    cumulative = np.cumsum(ordered)
    cutoff_index = int(
        np.searchsorted(cumulative, mass_fraction * cumulative[-1], side="left")
    )
    return float(ordered[min(cutoff_index, len(ordered) - 1)])


def plot_state_space(
    cooled: CoolingData,
    results: InferenceResults,
    hot: EquilibriumData | None,
    cold: EquilibriumData | None,
    slice_indices: tuple[int, int],
    output_path: Path,
    dpi: int,
) -> None:
    x_index, y_index = slice_indices
    allowed_x = results.probabilities[:, x_index]
    allowed_y = results.probabilities[:, y_index]
    cooled_x = cooled.probabilities[:, x_index]
    cooled_y = cooled.probabilities[:, y_index]
    cooled_x_low, cooled_x_high = error_endpoints(
        cooled_x, cooled.ci_lower[:, x_index], cooled.ci_upper[:, x_index]
    )
    cooled_y_low, cooled_y_high = error_endpoints(
        cooled_y, cooled.ci_lower[:, y_index], cooled.ci_upper[:, y_index]
    )
    exact_region = str(results.metadata.get("method", "")).startswith("exact_")

    x_bound_values: list[np.ndarray | float] = [
        projection_display_range(allowed_x, exact_region),
        cooled_x_low,
        cooled_x_high,
        results.estimate_lower[x_index],
        results.estimate_upper[x_index],
    ]
    y_bound_values: list[np.ndarray | float] = [
        projection_display_range(allowed_y, exact_region),
        cooled_y_low,
        cooled_y_high,
        results.estimate_lower[y_index],
        results.estimate_upper[y_index],
    ]
    for equilibrium in (hot, cold):
        if equilibrium is not None:
            x_low, x_high = error_endpoints(
                equilibrium.probabilities[[x_index]],
                equilibrium.ci_lower[[x_index]],
                equilibrium.ci_upper[[x_index]],
            )
            y_low, y_high = error_endpoints(
                equilibrium.probabilities[[y_index]],
                equilibrium.ci_lower[[y_index]],
                equilibrium.ci_upper[[y_index]],
            )
            x_bound_values.extend([x_low, x_high])
            y_bound_values.extend([y_low, y_high])
    padding = 0.04 if exact_region else 0.08
    x_limits = finite_linear_limits(
        x_bound_values, pad_fraction=padding, lower_clip=0.0, upper_clip=1.0
    )
    y_limits = finite_linear_limits(
        y_bound_values, pad_fraction=padding, lower_clip=0.0, upper_clip=1.0
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    ax.set_facecolor("0.58" if exact_region else "0.66")

    histogram, x_edges, y_edges = np.histogram2d(
        allowed_x,
        allowed_y,
        bins=EXACT_REGION_BINS if exact_region else LIKELY_REGION_BINS,
        range=[x_limits, y_limits],
    )
    density = (
        histogram.T
        if exact_region
        else gaussian_filter(
            histogram.T,
            sigma=LIKELY_REGION_SMOOTHING_SIGMA,
            mode="constant",
        )
    )
    positive = density[density > 0.0]
    if positive.size:
        if exact_region:
            masked_density = np.ma.masked_less_equal(density, 0.0)
            ax.pcolormesh(
                x_edges,
                y_edges,
                masked_density,
                shading="auto",
                cmap=LinearSegmentedColormap.from_list(
                    "allowed_white", ["white", "white"]
                ),
                vmin=0.0,
                vmax=max(1.0, float(positive.max())),
                rasterized=True,
                zorder=1,
            )
        else:
            minimum = highest_density_threshold(
                density, LIKELY_REGION_MASS_FRACTION
            )
            maximum = float(positive.max())
            if maximum <= minimum:
                maximum = minimum * (1.0 + 1e-12)
            grey_to_white = LinearSegmentedColormap.from_list(
                "likely_grey_to_white", ["0.76", "1.0"]
            )
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            ax.contourf(
                x_centers,
                y_centers,
                density,
                levels=np.linspace(
                    minimum, maximum, LIKELY_REGION_CONTOUR_LEVELS
                ),
                cmap=grey_to_white,
                norm=PowerNorm(gamma=0.75, vmin=minimum, vmax=maximum, clip=True),
                antialiased=False,
                zorder=1,
            )

    if hot is not None:
        ax.plot(
            [hot.probabilities[x_index], cooled_x[0]],
            [hot.probabilities[y_index], cooled_y[0]],
            color="black",
            linestyle=":",
            linewidth=1.0,
            alpha=0.65,
            zorder=4,
        )
    if cold is not None:
        ax.plot(
            [cooled_x[-1], cold.probabilities[x_index]],
            [cooled_y[-1], cold.probabilities[y_index]],
            color="black",
            linestyle=":",
            linewidth=1.0,
            alpha=0.65,
            zorder=4,
        )

    ax.plot(
        cooled_x,
        cooled_y,
        color="black",
        marker="o",
        markersize=4,
        linestyle="-",
        linewidth=1.1,
        alpha=0.78,
        label="Cooled distributions",
        zorder=7,
    )
    if np.any(cooled.uncertain_runs):
        uncertain = cooled.uncertain_runs
        ax.errorbar(
            cooled_x[uncertain],
            cooled_y[uncertain],
            xerr=errorbar_amounts(
                cooled_x[uncertain],
                cooled.ci_lower[uncertain, x_index],
                cooled.ci_upper[uncertain, x_index],
            ),
            yerr=errorbar_amounts(
                cooled_y[uncertain],
                cooled.ci_lower[uncertain, y_index],
                cooled.ci_upper[uncertain, y_index],
            ),
            fmt="none",
            color="black",
            elinewidth=0.8,
            capsize=1.5,
            alpha=0.38,
            zorder=5,
        )

    hot_temperature = float(results.metadata["hot_temperature"])
    cold_temperature = float(results.metadata["cold_temperature"])
    if hot is not None:
        hot_arguments = {
            "color": "tab:red",
            "marker": "*",
            "markersize": 8,
            "linestyle": "none",
            "label": f"Hot equilibrium ({format_value(hot_temperature)} K)",
            "zorder": 9,
        }
        if hot.has_uncertainty:
            ax.errorbar(
                hot.probabilities[x_index],
                hot.probabilities[y_index],
                xerr=errorbar_amounts(
                    hot.probabilities[[x_index]],
                    hot.ci_lower[[x_index]],
                    hot.ci_upper[[x_index]],
                ),
                yerr=errorbar_amounts(
                    hot.probabilities[[y_index]],
                    hot.ci_lower[[y_index]],
                    hot.ci_upper[[y_index]],
                ),
                elinewidth=1.2,
                capsize=2,
                **hot_arguments,
            )
        else:
            ax.plot(
                hot.probabilities[x_index],
                hot.probabilities[y_index],
                **hot_arguments,
            )
    if cold is not None:
        cold_arguments = {
            "color": "tab:blue",
            "marker": SNOWFLAKE_MARKER,
            "markersize": 9,
            "linestyle": "none",
            "label": f"Cold equilibrium ({format_value(cold_temperature)} K)",
            "zorder": 9,
        }
        if cold.has_uncertainty:
            ax.errorbar(
                cold.probabilities[x_index],
                cold.probabilities[y_index],
                xerr=errorbar_amounts(
                    cold.probabilities[[x_index]],
                    cold.ci_lower[[x_index]],
                    cold.ci_upper[[x_index]],
                ),
                yerr=errorbar_amounts(
                    cold.probabilities[[y_index]],
                    cold.ci_lower[[y_index]],
                    cold.ci_upper[[y_index]],
                ),
                elinewidth=1.2,
                capsize=2,
                **cold_arguments,
            )
        else:
            ax.plot(
                cold.probabilities[x_index],
                cold.probabilities[y_index],
                **cold_arguments,
            )
    ax.errorbar(
        results.estimate[x_index],
        results.estimate[y_index],
        xerr=np.asarray(
            [
                [results.estimate[x_index] - results.estimate_lower[x_index]],
                [results.estimate_upper[x_index] - results.estimate[x_index]],
            ]
        ),
        yerr=np.asarray(
            [
                [results.estimate[y_index] - results.estimate_lower[y_index]],
                [results.estimate_upper[y_index] - results.estimate[y_index]],
            ]
        ),
        color="tab:purple",
        marker="s",
        markersize=5,
        linestyle="none",
        elinewidth=1.3,
        capsize=2,
        label="Inferred hot equilibrium",
        zorder=10,
    )

    region_label = (
        "Thermodynamically allowed region"
        if str(results.metadata.get("method", "")).startswith("exact_")
        else "Thermodynamically likely region"
    )
    region_patch = Patch(facecolor="white", edgecolor="0.55", label=region_label)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [region_patch, *handles],
        [region_label, *labels],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
    )
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xlabel(f"Probability of {latex_escape_text(cooled.states[x_index])}")
    ax.set_ylabel(f"Probability of {latex_escape_text(cooled.states[y_index])}")
    ax.set_title("Recovery in state-probability space")
    format_axes(ax)
    save_figure(fig, output_path, dpi)


def plot_state_histogram(
    cooled: CoolingData,
    results: InferenceResults,
    hot: EquilibriumData | None,
    maximum_states: int,
    output_path: Path,
    dpi: int,
) -> None:
    state_count = len(cooled.states)
    shown_count = min(maximum_states, state_count)
    order = np.argsort(-cooled.probabilities[-1], kind="stable")[:shown_count]
    positions = np.arange(shown_count, dtype=float)

    series: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, str]] = []
    if hot is not None:
        series.append(
            (
                f"Hot equilibrium ({format_value(float(results.metadata['hot_temperature']))} K)",
                hot.probabilities[order],
                hot.ci_lower[order],
                hot.ci_upper[order],
                "tab:red",
            )
        )
    fastest_label = (
        "Cooled distribution"
        if len(cooled.times) == 1
        else f"Fastest cooling (time {format_value(cooled.times[0])})"
    )
    series.append(
        (
            fastest_label,
            cooled.probabilities[0, order],
            cooled.ci_lower[0, order],
            cooled.ci_upper[0, order],
            "0.65",
        )
    )
    if len(cooled.times) > 1:
        series.append(
            (
                f"Slowest cooling (time {format_value(cooled.times[-1])})",
                cooled.probabilities[-1, order],
                cooled.ci_lower[-1, order],
                cooled.ci_upper[-1, order],
                "0.25",
            )
        )
    series.append(
        (
            "Inferred hot equilibrium",
            results.estimate[order],
            results.estimate_lower[order],
            results.estimate_upper[order],
            "tab:purple",
        )
    )

    series_count = len(series)
    width = min(0.22, 0.82 / series_count)
    offsets = (np.arange(series_count) - 0.5 * (series_count - 1)) * width
    figure_width = max(6.8, 0.62 * shown_count + 2.2)
    fig, ax = plt.subplots(figsize=(figure_width, 4.6), constrained_layout=True)
    bound_endpoints = [np.asarray([0.0])]
    for offset, (label, mean, lower, upper, color) in zip(offsets, series):
        low_endpoint, high_endpoint = error_endpoints(mean, lower, upper)
        bound_endpoints.extend([low_endpoint, high_endpoint])
        ax.bar(
            positions + offset,
            mean,
            width,
            color=color,
            alpha=0.78,
            label=label,
            zorder=2,
        )
        lower_error = mean - low_endpoint
        upper_error = high_endpoint - mean
        if np.any(lower_error > 0.0) or np.any(upper_error > 0.0):
            ax.errorbar(
                positions + offset,
                mean,
                yerr=np.vstack([lower_error, upper_error]),
                fmt="none",
                color="black",
                linewidth=0.9,
                capsize=2,
                zorder=3,
            )

    ax.set_ylim(
        *finite_linear_limits(
            bound_endpoints,
            pad_fraction=0.06,
            lower_clip=0.0,
        )
    )
    ax.set_xlim(-0.55, shown_count - 0.45)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [latex_escape_text(cooled.states[index]) for index in order],
        rotation=35,
        ha="right",
    )
    ax.set_xlabel("State")
    ax.set_ylabel("Probability")
    title = "State-probability recovery"
    if shown_count < state_count:
        title += f" (top {shown_count} by slowest cooled probability)"
    ax.set_title(title)
    place_adaptive_legend(ax, force_outside=True)
    format_axes(ax, categorical_x=True)
    save_figure(fig, output_path, dpi)


def dirichlet_from_mean_standard_deviation(
    mean: np.ndarray, standard_deviation: np.ndarray
) -> np.ndarray:
    """Fit one Dirichlet concentration to marginal standard deviations."""
    standard_deviation = np.asarray(standard_deviation, dtype=float)
    clipped = np.clip(mean, PROBABILITY_FLOOR, 1.0)
    clipped = clipped / clipped.sum()
    estimates = np.full_like(clipped, np.nan)
    usable = np.isfinite(standard_deviation) & (standard_deviation > 0.0)
    estimates[usable] = (
        clipped[usable]
        * (1.0 - clipped[usable])
        / standard_deviation[usable] ** 2
        - 1.0
    )
    valid = np.isfinite(estimates) & (estimates > 0.0)
    if not np.any(valid):
        raise ValueError("Could not infer a positive Dirichlet concentration.")
    concentration = float(np.median(estimates[valid]))
    return np.maximum(clipped * concentration, PROBABILITY_FLOOR)


def dirichlet_from_mean_ci(
    mean: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Match the 95% uncertainty approximation used by the inference script."""
    standard_deviation = (upper - lower) / (2.0 * CI_Z_VALUE)
    return dirichlet_from_mean_standard_deviation(mean, standard_deviation)


def _selected_proposal_seeds(
    results: InferenceResults, selected_indices: np.ndarray
) -> np.ndarray:
    required = {"sample_id", "proposal_seed"}
    missing = required - set(results.diagnostics.columns)
    if missing:
        raise ValueError(
            "Uncertain-run entropy production requires sampling_diagnostics.csv "
            f"columns {sorted(required)}; missing {sorted(missing)}."
        )
    diagnostics = results.diagnostics.copy()
    if diagnostics["sample_id"].duplicated().any():
        raise ValueError("sampling_diagnostics.csv contains duplicate sample_id values.")
    diagnostics = diagnostics.set_index("sample_id")
    missing_ids = np.setdiff1d(selected_indices, diagnostics.index.to_numpy())
    if missing_ids.size:
        raise ValueError(
            "sampling_diagnostics.csv is missing selected sample IDs, including "
            f"{int(missing_ids[0])}."
        )
    seed_values = diagnostics.loc[selected_indices, "proposal_seed"]
    if seed_values.isna().any():
        raise ValueError("A selected uncertain sample has no proposal_seed.")
    return seed_values.to_numpy(dtype=np.int64)


def iter_selected_thermodynamic_inputs(
    cooled: CoolingData,
    results: InferenceResults,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    selected = results.selected_indices
    uncertain = bool(np.any(cooled.uncertain_runs))
    seeds = _selected_proposal_seeds(results, selected) if uncertain else None
    if uncertain:
        replay = results.metadata.get("accepted_cooled_draw_replay")
        if not isinstance(replay, dict) or replay.get("schema_version") != 1:
            raise ValueError(
                "The inference output does not record a supported accepted-cooled-"
                "draw replay schema; rerun inference with this package version."
            )
        recorded_numpy = str(replay.get("numpy_version", ""))
        if recorded_numpy != np.__version__:
            raise ValueError(
                "Exact replay of accepted cooled-distribution draws requires NumPy "
                f"{recorded_numpy}, but plotting is running under NumPy {np.__version__}."
            )

    for start in range(0, len(selected), THERMODYNAMIC_CHUNK_SIZE):
        stop = min(start + THERMODYNAMIC_CHUNK_SIZE, len(selected))
        indices = selected[start:stop]
        probability = results.probabilities[indices]
        entropy = results.entropies[indices]
        implied_cold = results.implied_cold[indices]
        if uncertain:
            cooled_draws = np.empty(
                (len(indices), len(cooled.times), len(cooled.states)), dtype=float
            )
            assert seeds is not None
            for local_index, proposal_seed in enumerate(seeds[start:stop]):
                rng = np.random.default_rng(int(proposal_seed))
                for run_index in range(len(cooled.times)):
                    alpha = results.cooled_alpha[run_index]
                    if np.all(np.isfinite(alpha)):
                        cooled_draws[local_index, run_index] = rng.dirichlet(alpha)
                    else:
                        cooled_draws[local_index, run_index] = cooled.probabilities[
                            run_index
                        ]
        else:
            cooled_draws = np.broadcast_to(
                cooled.probabilities,
                (len(indices), *cooled.probabilities.shape),
            )
        yield probability, entropy, implied_cold, cooled_draws


def compute_entropy_production_summary(
    cooled: CoolingData, results: InferenceResults
) -> pd.DataFrame:
    beta = float(results.metadata["temperature_ratio_beta"])
    sample_count = len(results.selected_indices)
    cooling_values = np.empty((sample_count, len(cooled.times)), dtype=float)
    auxiliary_values = np.empty_like(cooling_values)
    cursor = 0
    for probability, entropy, implied_cold, cooled_draws in iter_selected_thermodynamic_inputs(
        cooled, results
    ):
        count = len(probability)
        energy_levels = -np.log(probability) + entropy
        hot_energy = np.sum(probability * energy_levels, axis=1)
        hot_entropy = np.sum(probability * entropy - xlogy(probability, probability), axis=1)
        cold_energy = np.sum(implied_cold * energy_levels, axis=1)
        cold_entropy = np.sum(
            implied_cold * entropy - xlogy(implied_cold, implied_cold), axis=1
        )
        cooled_energy = np.einsum("smk,sk->sm", cooled_draws, energy_levels)
        cooled_entropy = np.sum(
            cooled_draws * entropy[:, None, :] - xlogy(cooled_draws, cooled_draws),
            axis=2,
        )
        cooling_values[cursor : cursor + count] = (
            cooled_entropy
            - hot_entropy[:, None]
            - beta * (cooled_energy - hot_energy[:, None])
        )
        auxiliary_values[cursor : cursor + count] = (
            cold_entropy[:, None]
            - cooled_entropy
            - beta * (cold_energy[:, None] - cooled_energy)
        )
        cursor += count

    cooling_p16, cooling_median, cooling_p84 = np.percentile(
        cooling_values, [16.0, 50.0, 84.0], axis=0
    )
    auxiliary_p16, auxiliary_median, auxiliary_p84 = np.percentile(
        auxiliary_values, [16.0, 50.0, 84.0], axis=0
    )
    return pd.DataFrame(
        {
            "cooling_time": cooled.times,
            "cooling_upper_bound_median": cooling_median,
            "cooling_upper_bound_p16": cooling_p16,
            "cooling_upper_bound_p84": cooling_p84,
            "auxiliary_median": auxiliary_median,
            "auxiliary_p16": auxiliary_p16,
            "auxiliary_p84": auxiliary_p84,
            "inference_sample_count": sample_count,
        }
    )


def plot_entropy_production(
    summary: pd.DataFrame,
    time_unit: str | None,
    output_path: Path,
    dpi: int,
) -> None:
    times = summary["cooling_time"].to_numpy(float)
    cooling_median = summary["cooling_upper_bound_median"].to_numpy(float)
    cooling_p16 = summary["cooling_upper_bound_p16"].to_numpy(float)
    cooling_p84 = summary["cooling_upper_bound_p84"].to_numpy(float)
    auxiliary_median = summary["auxiliary_median"].to_numpy(float)
    auxiliary_p16 = summary["auxiliary_p16"].to_numpy(float)
    auxiliary_p84 = summary["auxiliary_p84"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    ax.errorbar(
        times,
        cooling_median,
        yerr=np.vstack([cooling_median - cooling_p16, cooling_p84 - cooling_median]),
        color="tab:red",
        marker="o",
        linestyle="-",
        linewidth=1.4,
        capsize=3,
        label="Initial cooling (upper bound)",
    )
    ax.errorbar(
        times,
        auxiliary_median,
        yerr=np.vstack(
            [auxiliary_median - auxiliary_p16, auxiliary_p84 - auxiliary_median]
        ),
        color="tab:blue",
        marker="o",
        linestyle="-",
        linewidth=1.4,
        capsize=3,
        label="Auxiliary relaxation",
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, zorder=0)
    y_limits = finite_linear_limits(
        [cooling_p16, cooling_p84, auxiliary_p16, auxiliary_p84, 0.0],
        pad_fraction=0.08,
    )
    ax.set_ylim(*y_limits)
    set_adaptive_time_axis(ax, times, time_unit)
    ax.set_ylabel(r"Entropy production ($k_{\mathrm{B}}$)")
    ax.set_title("Inferred entropy production")
    place_adaptive_legend(ax, force_outside=True)
    format_axes(ax)
    save_figure(fig, output_path, dpi)


def draw_distribution_samples(
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    has_uncertainty: bool,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if has_uncertainty:
        alpha = dirichlet_from_mean_ci(mean, lower, upper)
        return rng.dirichlet(alpha, size=count)
    return np.broadcast_to(mean, (count, len(mean)))


def kl_divergence_rows(distribution: np.ndarray, reference: np.ndarray) -> np.ndarray:
    distribution = np.asarray(distribution, dtype=float)
    reference = np.asarray(reference, dtype=float)
    values = np.sum(
        xlogy(distribution, distribution) - xlogy(distribution, reference), axis=-1
    )
    return np.maximum(values, 0.0)


def compute_kl_summary(
    cooled: CoolingData,
    results: InferenceResults,
    hot: EquilibriumData,
    draw_count: int,
    seed: int,
) -> pd.DataFrame:
    if np.any(hot.probabilities <= 0.0):
        raise ValueError(
            "KL divergence to the supplied hot equilibrium is infinite or undefined "
            "because it contains a zero-probability state."
        )
    rng = np.random.default_rng(seed)
    hot_draws = draw_distribution_samples(
        hot.probabilities,
        hot.ci_lower,
        hot.ci_upper,
        hot.has_uncertainty,
        draw_count,
        rng,
    )

    rows: list[dict[str, float | int | str]] = []
    for run_index, cooling_time in enumerate(cooled.times):
        alpha = results.cooled_alpha[run_index]
        if np.all(np.isfinite(alpha)):
            cooled_draws = rng.dirichlet(alpha, size=draw_count)
        else:
            cooled_draws = np.broadcast_to(
                cooled.probabilities[run_index], (draw_count, len(cooled.states))
            )
        values = kl_divergence_rows(cooled_draws, hot_draws)
        p16, median, p84 = np.percentile(values, [16.0, 50.0, 84.0])
        rows.append(
            {
                "series": "cooled",
                "cooling_time": float(cooling_time),
                "central_value": float(median),
                "median": float(median),
                "p16": float(p16),
                "p84": float(p84),
                "sample_count": draw_count,
            }
        )

    point_value = float(
        kl_divergence_rows(results.estimate[None, :], hot.probabilities[None, :])[0]
    )
    rows.append(
        {
            "series": "inferred",
            "cooling_time": np.nan,
            "central_value": point_value,
            "median": point_value,
            "p16": np.nan,
            "p84": np.nan,
            "sample_count": 1,
        }
    )
    return pd.DataFrame(rows)


def plot_kl_divergence(
    summary: pd.DataFrame,
    time_unit: str | None,
    output_path: Path,
    dpi: int,
) -> None:
    cooled_rows = summary.loc[summary["series"] == "cooled"]
    inferred_row = summary.loc[summary["series"] == "inferred"].iloc[0]
    times = cooled_rows["cooling_time"].to_numpy(float)
    median = cooled_rows["median"].to_numpy(float)
    p16 = cooled_rows["p16"].to_numpy(float)
    p84 = cooled_rows["p84"].to_numpy(float)
    inferred_central = float(inferred_row["central_value"])

    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    lower_error = median - p16
    upper_error = p84 - median
    cooled_arguments = {
        "color": "black",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.4,
        "alpha": 0.78,
        "label": "Cooled distributions",
    }
    if np.any(lower_error > 0.0) or np.any(upper_error > 0.0):
        cooled_handle = ax.errorbar(
            times,
            median,
            yerr=np.vstack([lower_error, upper_error]),
            capsize=3,
            **cooled_arguments,
        )
    else:
        cooled_handle = ax.plot(times, median, **cooled_arguments)[0]
    x_limits = (
        finite_log_limits([times])
        if np.all(times > 0.0) and len(times) > 1 and times.max() / times.min() >= 20.0
        else finite_linear_limits([times], pad_fraction=0.05)
    )
    point_handle = ax.hlines(
        inferred_central,
        x_limits[0],
        x_limits[1],
        color="tab:purple",
        linewidth=1.5,
        label=r"$D_{\mathrm{KL}}$ of point estimate",
    )

    y_values = [p16, p84, inferred_central]
    finite_positive = np.concatenate(
        [
            np.asarray(value, dtype=float).ravel()
            for value in y_values
            if np.asarray(value).size
        ]
    )
    finite_positive = finite_positive[
        np.isfinite(finite_positive) & (finite_positive > 0.0)
    ]
    all_strictly_positive = bool(
        np.all(p16 > 0.0)
        and inferred_central > 0.0
        and finite_positive.size
    )
    dynamic_range = (
        float(finite_positive.max() / finite_positive.min())
        if finite_positive.size
        else 1.0
    )
    if all_strictly_positive and dynamic_range >= 20.0:
        ax.set_yscale("log")
        ax.set_ylim(*finite_log_limits(y_values))
    else:
        ax.set_ylim(
            *finite_linear_limits(y_values + [0.0], pad_fraction=0.08)
        )
    set_adaptive_time_axis(ax, times, time_unit)
    ax.set_xlim(*x_limits)
    ax.set_ylabel(r"$D_{\mathrm{KL}}(p\,\Vert\,\pi_{\mathrm{hot}})$")
    ax.set_title("Divergence from the true hot equilibrium")
    legend_handles = [cooled_handle, point_handle]
    legend_labels = [
        "Cooled distributions",
        r"$D_{\mathrm{KL}}$ of point estimate",
    ]
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        borderaxespad=0.0,
    )
    format_axes(ax)
    save_figure(fig, output_path, dpi)


def prepare_output_paths(
    output_dir: Path, figure_format: str, include_kl: bool, overwrite: bool
) -> dict[str, Path]:
    paths = {
        "figure1": output_dir / f"figure_1_state_space.{figure_format}",
        "figure2": output_dir / f"figure_2_state_probabilities.{figure_format}",
        "figure3": output_dir / f"figure_3_entropy_production.{figure_format}",
        "figure3_data": output_dir / "figure_3_entropy_production_data.csv",
    }
    if include_kl:
        paths.update(
            {
                "figure4": output_dir / f"figure_4_kl_divergence.{figure_format}",
                "figure4_data": output_dir / "figure_4_kl_divergence_data.csv",
            }
        )
    stale_kl_paths = {
        output_dir / f"figure_4_kl_divergence.{extension}"
        for extension in ("pdf", "png", "svg")
    }
    stale_kl_paths.add(output_dir / "figure_4_kl_divergence_data.csv")
    conflicts = {path for path in paths.values() if path.exists()}
    if not include_kl:
        conflicts.update(path for path in stale_kl_paths if path.exists())
    conflicts = sorted(conflicts)
    if conflicts and not overwrite:
        formatted = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(
            f"Plot outputs already exist: {formatted}. Choose another directory "
            "or pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in conflicts:
            if path.is_file() or path.is_symlink():
                path.unlink()
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    cooled_path = args.cooled.resolve()
    inference_dir = args.inference_dir.resolve()
    output_dir = args.output_dir.resolve()

    cooled = load_cooled_csv(cooled_path)
    results = load_inference_results(inference_dir, cooled, cooled_path)
    hot = (
        load_equilibrium_csv(args.hot_equilibrium.resolve(), cooled.states)
        if args.hot_equilibrium is not None
        else None
    )
    cold = (
        load_equilibrium_csv(args.cold_equilibrium.resolve(), cooled.states)
        if args.cold_equilibrium is not None
        else None
    )
    paths = prepare_output_paths(output_dir, args.format, hot is not None, args.overwrite)
    configure_style(args.format)

    slice_indices = choose_slice_indices(cooled, args.slice_states)
    print(
        "Figure 1 slice states: "
        f"{cooled.states[slice_indices[0]]!r} and {cooled.states[slice_indices[1]]!r}",
        flush=True,
    )
    plot_state_space(
        cooled, results, hot, cold, slice_indices, paths["figure1"], args.dpi
    )
    print(f"Saved {paths['figure1']}", flush=True)

    plot_state_histogram(
        cooled,
        results,
        hot,
        args.max_histogram_states,
        paths["figure2"],
        args.dpi,
    )
    print(f"Saved {paths['figure2']}", flush=True)

    entropy_summary = compute_entropy_production_summary(cooled, results)
    entropy_summary.to_csv(paths["figure3_data"], index=False)
    plot_entropy_production(
        entropy_summary, args.time_unit, paths["figure3"], args.dpi
    )
    print(f"Saved {paths['figure3']} and {paths['figure3_data']}", flush=True)

    if hot is None:
        print(
            "Skipping Figure 4 because --hot-equilibrium was not supplied.",
            flush=True,
        )
    else:
        kl_summary = compute_kl_summary(
            cooled, results, hot, args.uncertainty_draws, args.seed
        )
        kl_summary.to_csv(paths["figure4_data"], index=False)
        plot_kl_divergence(
            kl_summary,
            args.time_unit,
            paths["figure4"],
            args.dpi,
        )
        print(f"Saved {paths['figure4']} and {paths['figure4_data']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
