# Thermodynamic Recovery from Cooled Distributions

Python package for implementing the thermodynamic recovery method developed in the following paper: https://www.biorxiv.org/content/10.64898/2026.04.21.720011v1.abstract. This software is designed to take a set of probability distributions from the endpoints of nonequilibrium cooling experiments at different cooling rates, together with the initial and final temperatures, and estimate the initial hot equilibrium distribution. The outputs are a set of thermodynamically allowed (or likely, if uncertainty analysis is enabled) points in distribution-space, together with a point estimate for the distribution which is by default the center of mass of the set of allowed points.

The recovery algorithm infers hot-equilibrium state probabilities from probability
distributions observed after cooling. It implements the thermodynamic bounds
in Eqs. (2), (3), (8), (9), (11), and (12) of the accompanying manuscript. The key computational aspect is the generation of thermodynamically allowed sample points. This package contains substantial algorithmic improvements over the algorithms used in the original manuscript, but the underlying logic of the thermodynamic constraints is unchanged.


Two command-line programs are provided:

- `thermodynamic_inference.py` generates thermodynamically feasible samples
  and a point estimate of the hot-equilibrium distribution.
- `plot_thermodynamic_recovery.py` turns an inference output directory into up
  to four recovery figures.

Hot- and cold-equilibrium reference files are never read by the inference
program. They are optional plotting inputs used only to evaluate a recovery.


AI Disclosure: We acknowledge the use of AI tools (in this case ChatGPT 5.6 Sol Ultra) in the creation of this package, mainly for software architecture and best practices. The sampling algorithm for generating thermodynamically allowed distributions was also substantially improved (2+ orders of magnitude faster) from the initial algorithm used in our paper through an AI-generated suggestion. The author (Matthew Leighton) takes responsibility for the method and package.

We have tested the package extensively on both toy datasets with few states, and the trp-cage dataset from MSM coarse-graining of extensive MD simulations used in the manuscript. We have not tested the package on other cryo-EM datasets, though we see no reason it will not work similarly well. If you test the package on a new dataset and find it performs poorly, we would be interested to know and understand why!

Any questions should be directed to matthew.leighton@yale.edu.


# Technical documentation for the package begins below:

## Quick start

Python 3.10 or newer is required. From a source checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run a small six-state inference:

```bash
python thermodynamic_inference.py \
  --cooled example_datasets/six_state_toy/cooled_distributions.csv \
  --hot-temperature 500 \
  --cold-temperature 100 \
  --no-microstate-entropies \
  --n-samples 100 \
  --workers 1 \
  --output-dir quickstart_results
```

Then generate its recovery figures:

```bash
python plot_thermodynamic_recovery.py \
  --cooled example_datasets/six_state_toy/cooled_distributions.csv \
  --inference-dir quickstart_results \
  --hot-equilibrium example_datasets/six_state_toy/hot_equilibrium.csv \
  --cold-equilibrium example_datasets/six_state_toy/cold_equilibrium.csv \
  --slice-states state_2 state_5 \
  --time-unit "simulation steps" \
  --output-dir quickstart_figures
```

The plotting program uses LaTeX and Computer Modern fonts to match the REVTeX
manuscript. PDF and SVG output require `latex`; PNG output additionally
requires `dvipng`.

Installing the repository also creates the commands `thermodynamic-inference`
and `thermodynamic-recovery-plot`:

```bash
python -m pip install .
```

The example data and precomputed outputs are part of the source repository;
they are not installed by the wheel.

## Method at a glance

| Input | Default method | Meaning of generated probabilities | Main diagnostics |
|---|---|---|---|
| Exact, at most 6 states | Scrambled Sobol rejection | Selected uniform measure on a fixed feasible region | Proposal acceptance |
| Exact, more than 6 states | Random-walk MCMC | Correlated sample from a fixed feasible region | Acceptance and split-R-hat |
| One or more 95% intervals | Random projection | Proposal-dependent projected feasible cloud | CI fit, radial contraction, margins, entropy saturation |

Every saved sample is checked against the manuscript equations. Cooling-time
values are used only to order the runs: the smallest value is the fastest
cooling condition. Their numerical spacing does not enter the inference.

## Input data

### Cooled distributions

Use a long-form CSV with one row per cooling condition and state:

```text
cooling_time,state,probability,ci_lower_95,ci_upper_95
1,state_A,0.18,0.16,0.20
1,state_B,0.28,0.25,0.31
1,state_C,0.54,0.51,0.57
10,state_A,0.15,0.13,0.17
...
```

Required columns are:

- `cooling_time`: a nonnegative number used to order the cooling conditions.
- `state`: a label. Every condition must contain the same labels.
- `probability`: the mean state probability.

The optional `ci_lower_95` and `ci_upper_95` columns contain marginal 95%
confidence limits. Within a cooling condition, both limits must be supplied
for every state or omitted for every state. A condition with blank bounds is
treated as exact, even if another condition has uncertainty.

Each distribution must sum to one, and each `(cooling_time, state)` pair must
be unique. Aggregate experimental or simulation replicates before constructing
the file. The program retains every supplied state and does not collapse states
automatically.

### Equilibrium references for plotting

Optional hot- and cold-equilibrium files use:

```text
state,probability,ci_lower_95,ci_upper_95
```

The confidence columns may be blank. Labels must match the cooled input.
Supplying a hot reference enables Figure 4; its probabilities must be strictly
positive for a finite KL divergence. A cold reference is shown only in Figure
1. Neither reference affects inference.

## Inference settings

### Microstate entropies

Microstate entropies are inferred by default. Disable them with:

```bash
--no-microstate-entropies
```

Only entropy differences are identifiable. Saved entropy vectors are centered
so that their minimum and maximum are symmetric about zero. The default bound
is `±ln(number of states)` in units of `k_B`. For example,
`--entropy-bound 2` permits a full entropy range of four.

### Exact cooled probabilities

At or below the default six-state threshold, scrambled Sobol proposals are
uniform in simplex volume and in a linear, gauge-fixed entropy-difference
domain. Only proposals satisfying every inequality are retained. Above the
threshold, four symmetric random-walk MCMC chains target uniform volume on the
same joint feasible domain.

Select a method with `--exact-method simplex` or `--exact-method mcmc`, or
change the automatic cutoff with `--simplex-threshold`. Simplex rejection can
be expensive for narrow regions; the six-state example accepts about 0.02% of
uniform probability proposals. MCMC output is correlated, so its acceptance
rates and split-R-hat diagnostics should always be inspected.

When entropies vary, uniform joint sampling weights a probability vector by
the volume of compatible entropy differences. The resulting probability
center is therefore a marginal center of the selected joint measure, not the
center of the probability projection obtained by treating entropy only as an
existential nuisance parameter.

### Cooled probabilities with uncertainty

For each uncertain condition, the program approximates a Dirichlet
distribution from the supplied mean and total 95% interval width. Each
proposal then:

1. draws one realization of every uncertain cooled distribution;
2. draws a probability target centered on the fastest cooled mean;
3. draws an entropy target when entropies are enabled;
4. projects the target onto the feasible set; and
5. verifies the saved sample using the manuscript equations.

With variable entropies, feasibility at a target probability is solved as a
linear program. If necessary, the probability is moved toward the fastest
cooled distribution using a coarse radial scan and local boundary refinement.
This construction is stable for sparse systems but is anchor-dependent and is
not a global nearest projection. Narrow disconnected feasible intervals along
a ray can be missed.

Asymmetric input limits are accepted, but only their total width is used to fit
the Dirichlet approximation. A single concentration parameter generally
cannot reproduce every marginal interval. The resulting fit is recorded in
`uncertainty_model_summary.csv` and should be checked rather than interpreted
as an exact uncertainty model.

### Point estimate

The default `--estimator center` reports the componentwise mean and population
standard deviation of all generated samples. For uncertain input, these
quantities summarize the proposal-dependent projected cloud; the standard
deviation is not a calibrated confidence or credible interval.

The alternative `--estimator nearest-median` selects the requested fraction of
samples closest to the fastest cooled mean, reports their normalized
componentwise median, and supplies marginal 16th-84th percentiles. The default
fraction is 10%; change it with `--nearest-fraction`.

## Output files

The inference output directory contains:

| File | Contents |
|---|---|
| `allowed_samples.npz` | Probabilities, entropies, implied-cold probabilities, input arrays, state labels, and selected-estimator indices |
| `allowed_region_summary.csv` | Per-state summaries of the full generated cloud; the historical filename is retained for compatibility |
| `hot_equilibrium_estimate.csv` | Point estimate and population SD, or median and 16th-84th percentiles |
| `sampling_diagnostics.csv` | Rejection, MCMC, or projection diagnostics for each retained sample or chain |
| `uncertainty_model_summary.csv` | Input-versus-modeled interval widths for uncertain data |
| `run_metadata.json` | Settings, method interpretation, hashes, dependency versions, and validation results |

With the center estimator, the percentile columns in
`hot_equilibrium_estimate.csv` are blank. With nearest-median, they contain the
16th and 84th percentiles.

Pass `--write-sample-csv` to add full probability and implied-cold CSV files;
an entropy CSV is also written when microstate entropies are enabled. NPZ is
the default because 100,000-sample CSV files are substantially larger and
slower to read. Existing generated files are not replaced unless
`--overwrite` is supplied.

## Interpreting diagnostics

For uncertainty-aware runs, inspect:

- `modeled_to_input_width_ratio` in `uncertainty_model_summary.csv`;
- `probability_radial_fraction`, where values below one indicate contraction
  toward the fastest cooled distribution;
- `entropy_range`, especially repeated contact with
  `2 * entropy_bound`; and
- `manuscript_minimum_margin`, where repeated values near the numerical
  constraint buffer indicate boundary-dominated witnesses.

For exact MCMC, inspect chain acceptance and every split-R-hat value. The
program warns when split-R-hat exceeds 1.05.

## Recovery figures

The plotting command reads the original cooled CSV and its inference output:

```bash
python plot_thermodynamic_recovery.py \
  --cooled example_datasets/trp_cage_high_probability/cooled_distributions.csv \
  --inference-dir example_datasets/trp_cage_high_probability/recovery_outputs \
  --hot-equilibrium example_datasets/trp_cage_high_probability/hot_equilibrium.csv \
  --cold-equilibrium example_datasets/trp_cage_high_probability/cold_equilibrium.csv \
  --slice-states state_23 state_10 \
  --time-unit ns \
  --output-dir recovery_figures
```

It writes:

1. `figure_1_state_space`: the cooled trajectory, references, point estimate,
   and full generated probability cloud in a two-state slice;
2. `figure_2_state_probabilities`: the point estimate, references, and fastest
   and slowest cooled distributions for at most ten states;
3. `figure_3_entropy_production`: cooling upper-bound and auxiliary-relaxation
   summaries for the estimator-selected samples; and
4. `figure_4_kl_divergence`: cooled and inferred KL divergence from the
   supplied hot reference, written only when that reference is provided.

Figure 3 uses one solver-selected entropy witness paired with each selected
probability and cooled-distribution realization. Its 16th-84th percentile bars
describe that generated witness cloud, not a posterior interval or the full
range over all compatible entropies. It does not evaluate entropy production
at the componentwise mean probability. The red curve is an upper bound; a
staircase near `1e-8`, `2e-8`, and so on can indicate saturation of the
numerical constraint buffer rather than a resolved physical scale.

Figures 2 and 4 use the reported point estimate. Figure 1 uses the full cloud.
Formats are PDF by default; PNG and SVG are available with `--format`. Axis
limits include all plotted means and error bars. Without `--slice-states`,
Figure 1 uses the two most populated categories in the slowest cooled
distribution.

The numeric data behind Figures 3 and 4 are saved as CSV. Existing plot files
require `--overwrite` to be replaced.

## Assumptions and limitations

- Temperatures are positive absolute temperatures with `T_hot > T_cold`.
- Energies and microstate entropies are temperature-independent.
- Cooling is monotonic in temperature.
- Mean energy decreases monotonically during cooling, as required for the
  cooling upper bound in manuscript Eq. (12).
- State dynamics obey local detailed balance.
- The cooling upper bound is nondecreasing with cooling time, and auxiliary
  relaxation entropy production is nonincreasing with cooling time.

The inequalities identify distributions that are not ruled out by the stated
thermodynamic bounds. Because Eq. (12) is an upper bound, feasible membership
alone does not establish nonnegative entropy production along a particular
physical cooling path.

## Reproducibility and testing

The default inference seed is fixed at `20260814`; change it with `--seed`.
Uncertainty-aware work is split into deterministic chunks, so changing
`--workers` does not change the result. Sobol batches and MCMC chain seeds are
also independent of worker assignment.

Bundled uncertain outputs were generated with NumPy 1.26.4. Figure 3 replays
the paired cooled draws and therefore requires that exact NumPy version. For
the recorded Python dependency versions, install:

```bash
python -m pip install -r requirements-reproducibility.txt
```

`run_metadata.json` records input and script hashes, dependency versions, the
random seed, numerical tolerances, and the complete inference settings.
Accepted proposal seeds and fitted Dirichlet parameters permit replay of each
paired cooled realization under the recorded NumPy version.

Run the tests from the repository root with:

```bash
python -m unittest discover -s tests -v
```

Full commands for all bundled examples are documented in
[`example_datasets/README.md`](example_datasets/README.md).

## Troubleshooting

- **Output files already exist:** choose a new directory or pass
  `--overwrite` deliberately.
- **LaTeX or `dvipng` is missing:** install a TeX distribution; `dvipng` is
  needed only for PNG output.
- **Dirichlet interval-fit warning:** inspect
  `uncertainty_model_summary.csv`; the warning does not mean the run failed.
- **MCMC convergence warning:** increase burn-in or retained sampling and
  inspect acceptance and split-R-hat diagnostics.
- **NumPy replay mismatch while plotting Figure 3:** install the pinned
  reproducibility requirements or rerun inference with the current NumPy
  version before plotting.
