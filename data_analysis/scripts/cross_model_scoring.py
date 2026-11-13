#!/usr/bin/env python3
"""
Cross-model scoring of the experimentally tested designs (adabmDCA energies).

Produces: Fig. 3 (A: relative energy distributions, B: energy-penalty matrix),
Fig. S2 (all divergence bins) and the supplementary CSV of the functional
enzymes missed by the low-entropy models.

# Cross-model scoring analysis with adabmDCA

Companion to `verify_analysis.ipynb`: same experimental designs, energy convention,
functional subsets, divergence bins, median penalties, strict empirical percentiles,
`beyond_max` flags, supplementary CSV, LaTeX tables and pairwise rank statistics.

Run all cells in order with **Python (DCA_env)**. DCA operations come directly from
the installed package: `load_params`, `get_tokens`, `encode_sequence`, `one_hot`,
`compute_energy` and `compute_density`. There are **no locally defined functions**.
The analysis-specific aggregation uses NumPy/Pandas; it is not a built-in adabmDCA analysis.

Inputs are the released files in `models/tested_models/` and `data/experiment/`.
Outputs go to `figures/` and `results/` of the package.
CPU and float64 are used to match the numerical precision of the original notebook.

Plot styling follows `entropy.ipynb`, `mutability.ipynb` and
`overfitting_analysis/evaluate_dE_take1out.ipynb`: Seaborn deep model colors,
LaTeX typography when available, inward ticks, dark borders and translucent histograms
with outlines. Histogram colors identify the **generating model** (eaDCA red, meDCA blue).

Converted from the notebook data_analysis/verify_analysis_adabmDCA.ipynb; the analysis code is unchanged.
Cell boundaries are kept as `# %%`, so the script can also be run cell by cell.
"""

def display(obj):
    """Script-mode replacement for IPython's display(): print the object."""
    print(getattr(obj, "data", obj))

# %%
from pathlib import Path
from tempfile import TemporaryDirectory
from importlib.metadata import version
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

import adabmDCA
from adabmDCA.fasta import get_tokens, encode_sequence
from adabmDCA.functional import one_hot
from adabmDCA.graph import compute_density
from adabmDCA.io import load_params
from adabmDCA.statmech import compute_energy
from adabmDCA.utils import get_device, get_dtype

print('Python:', sys.executable)
print('adabmDCA:', version('adabmDCA'), '|', adabmDCA.__file__)

# %% [markdown]
# ## Configuration
#
# Find the repository by searching the current working directory and its parents.
# The paths below select the released models explicitly, avoiding similarly named old models.
# ZIP archives are preferred, as in the original notebook; an uncompressed `<model>_params.dat`
# is accepted if its ZIP is absent. ZIP members are copied into a temporary directory solely
# to provide `load_params` with a plain file; parameter parsing is entirely done by adabmDCA.

# %%
from paths import (EXPERIMENT_DIR, FIGURES_INTERMEDIATE, FIGURES_MAIN,
                   FIGURES_SUPPLEMENTARY, RELEASED_MODELS_DIR, REPO_ROOT, RESULTS_DIR)
MODELS = ['eaDCA', 'bmDCA', 'edDCA', 'meDCA']
ENTROPY = {'eaDCA': 113.35, 'bmDCA': 121.82, 'edDCA': 136.05, 'meDCA': 150.13}
BINS = ['20-25%', '40-45%', '60-65%']
PLOT_BINS = BINS[:2]
PARAM_FILES = {}
for model in MODELS:
    candidates = [RELEASED_MODELS_DIR / f'{model}_params{suffix}'
                  for suffix in ('.zip', '.dat')]
    PARAM_FILES[model] = next((p for p in candidates if p.is_file()), None)
    if PARAM_FILES[model] is None:
        raise FileNotFoundError(f'Missing released parameters: {candidates}')

experiment_dir = EXPERIMENT_DIR
THREE_MODEL_CSV = next((experiment_dir / name for name in
                        ['Sep2024_bmDCA_edDCA_eaDCA_CM_REdata.csv',
                         'bmDCA_edDCA_eaDCA_CM_REdata.csv']
                        if (experiment_dir / name).is_file()), None)
MEDCA_CSV = experiment_dir / 'meDCA_CM_REdata.csv'
if THREE_MODEL_CSV is None or not MEDCA_CSV.is_file():
    raise FileNotFoundError(f'Missing experimental CSVs in {experiment_dir}')

device = get_device('cpu')
dtype = get_dtype('float64')
tokens = get_tokens('protein')
BATCH_SIZE = 256
# Limit small CPU matrix operations to a reasonable number of threads.
torch.set_num_threads(min(4, torch.get_num_threads()))
print('Figures:', FIGURES_MAIN, FIGURES_SUPPLEMENTARY, FIGURES_INTERMEDIATE)
print('Device / dtype / alphabet:', device, dtype, tokens)
for model, path in PARAM_FILES.items():
    print(f'{model:6s} <- {path.relative_to(REPO_ROOT)}')
print('Three-model CSV:', THREE_MODEL_CSV.relative_to(REPO_ROOT))
print('meDCA CSV:', MEDCA_CSV.relative_to(REPO_ROOT))

# %% [markdown]
# ## Plot style
#
# The same model-to-color mapping as the entropy and mutability notebooks is used.
# Set `USE_TEX = False` if a partial LaTeX installation causes rendering errors.
# Percent signs are escaped only when LaTeX is active.

# %%
USE_TEX = all(shutil.which(program) is not None for program in ('latex', 'dvipng', 'kpsewhich'))
palette = sns.color_palette('deep', 4)
colors = dict(zip(['meDCA', 'edDCA', 'bmDCA', 'eaDCA'], palette))
plt.rcParams.update({
    'text.usetex': USE_TEX,
    'font.family': 'sans-serif',
    'font.weight': 'normal',
    'font.size': 24,
    'axes.labelweight': 'normal',
    'axes.titleweight': 'normal',
    'axes.labelsize': 32,
    'axes.titlesize': 34,
    'axes.linewidth': 1.5,
    'axes.edgecolor': '#2F2F2F',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'axes.titlecolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'legend.labelcolor': 'black',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.labelsize': 27,
    'ytick.labelsize': 27,
    'legend.frameon': False,
    'axes.grid': False,
    'axes.axisbelow': True,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 300,
})
bin_labels = {b: b.replace('%', r'\%') if USE_TEX else b for b in BINS}
print('LaTeX rendering:', USE_TEX)

# %% [markdown]
# ## Load the four models and check density
#
# `load_params` returns `bias[L,q]` and symmetric `coupling_matrix[L,q,L,q]`.
# `compute_density(coupling_matrix != 0)` counts nonzero coefficients, not merely
# position pairs. Expected densities: bmDCA approximately 100%, eaDCA 2.8%,
# edDCA 2.9%, meDCA 12.5%.
#
# The same token order is passed to parameter loading and sequence encoding.
# The package computes $E(A)=-\sum_i h_i(a_i)-\sum_{i<j}J_{ij}(a_i,a_j)$.
# No gauge transformation or energy normalization is applied.

# %%
PARAMS, DENSITY = {}, {}
with TemporaryDirectory(prefix='verify_adabmdca_') as temporary_dir:
    for model, source in PARAM_FILES.items():
        parameter_path = source
        if source.suffix == '.zip':
            if not zipfile.is_zipfile(source):
                raise ValueError(f'{source} is not a ZIP archive; check the Git LFS download.')
            with zipfile.ZipFile(source) as archive:
                members = [name for name in archive.namelist()
                           if name.endswith('.dat') and not name.startswith('__MACOSX')]
                if len(members) != 1:
                    raise ValueError(f'Expected one parameter .dat in {source}: {members}')
                parameter_path = Path(temporary_dir) / f'{model}.dat'
                with archive.open(members[0]) as src, parameter_path.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
        PARAMS[model] = load_params(str(parameter_path), tokens=tokens, device=device, dtype=dtype)
        DENSITY[model] = compute_density(PARAMS[model]['coupling_matrix'] != 0)
        L, q = PARAMS[model]['bias'].shape
        print(f'{model}: L={L} q={q} density={100 * DENSITY[model]:6.3f}%')

model_shapes = {tuple(params['bias'].shape) for params in PARAMS.values()}
if len(model_shapes) != 1 or q != len(tokens):
    raise ValueError(f'Incompatible model shapes/alphabet: {model_shapes}, tokens={tokens}')

# %% [markdown]
# ## Load the experimental designs
#
# Keep the same model-specific rows and row order as the original. Functional labels
# come from `Null Mode 3Sigma Classification`; relative enrichment comes from
# `M9c Norm R.E.`. Neither functional labels nor divergence bins are recomputed.
# Unknown sequence symbols and inconsistent lengths stop execution instead of being
# silently mapped to gaps.

# %%
# LOAD AND PREPARE EXPERIMENTAL DESIGNS
# Collect one filtered table for each of the two experiments.
frames = []
# The first CSV supplies eaDCA, bmDCA and edDCA (MODELS[:3]); the second supplies meDCA.
for path, accepted_models in [(THREE_MODEL_CSV, MODELS[:3]), (MEDCA_CSV, ['meDCA'])]:
    # Read all values as text and preserve empty cells as empty strings.
    # Numeric fields are converted explicitly below.
    raw = pd.read_csv(path, keep_default_na=False, dtype=str)
    # Example: "eaDCA 20-25%" becomes ["eaDCA", "20-25%"].
    groups = raw['Group'].str.split()
    frame = pd.DataFrame({
        # Preserve the aligned sequence, full design identifier and original group.
        'seq': raw['Sequence'], 'name': raw['Name'], 'group': raw['Group'],
        # First word: generating model. Second word: divergence interval.
        # If the second word is missing, use the full group label as a fallback.
        'model': groups.str[0], 'bin': groups.str[1].fillna(raw['Group']),
        # Extract the number following "Model_Energy" or "DCAenergy" in the name.
        # Missing matches become NaN: energies are not recalculated here.
        'stored_E': raw['Name'].str.extract(
            r'(?:Model_Energy|DCAenergy)\s+(-?[\d.]+)', expand=False).astype(float),
        # Convert normalized relative enrichment to float; empty entries become NaN.
        're': raw['M9c Norm R.E.'].replace('', np.nan).astype(float),
        # Use the existing experimental classification: True only for "Functional".
        # Remove surrounding whitespace without recalculating functionality thresholds.
        'functional': raw['Null Mode 3Sigma Classification'].str.strip().eq('Functional'),
    })
    # Keep only the models assigned to this CSV, excluding other groups
    # (such as natural controls) while preserving row order.
    frames.append(frame.loc[frame['model'].isin(accepted_models)])
# Combine both tables with consecutive indices. Duplicate sequences are retained.
design_df = pd.concat(frames, ignore_index=True)
# Require at least one design and sequence lengths matching the model length L.
# This check does not perform sequence alignment.
if design_df.empty or not design_df['seq'].str.len().eq(L).all():
    raise ValueError(f'Expected nonempty aligned designs of length {L}.')
# Every design must belong to a divergence interval configured in BINS.
if not design_df['bin'].isin(BINS).all():
    raise ValueError('Unexpected divergence bin in experimental designs.')
# adabmDCA converts letters to integer indices using the same token order
# used to load the parameters. Unknown symbols raise an error.
# The result has shape (N, L): N designs, L positions per sequence.
encoded = encode_sequence(design_df['seq'].tolist(), tokens=tokens)
# Convert to a PyTorch integer tensor on the configured device.
# One-hot encoding and energy calculation take place in the next cell.
seqs_int = torch.as_tensor(encoded, device=device, dtype=torch.int64)
# Report the design count, sequence length and alphabet size.
print(f'Designs: {len(design_df)} | L={L} | q={q}')

# Summarize all designs before displaying individual examples.
# Summing the Boolean functional labels counts the functional designs in each group.
# Reindex to preserve the model/bin order and include any missing groups with zero counts.
design_summary_index = pd.MultiIndex.from_product(
    [MODELS, BINS], names=['Model', 'Divergence interval']
)
design_summary = (
    design_df.groupby(['model', 'bin'])['functional']
    .agg(**{'Number of designs': 'size', 'Functional designs': 'sum'})
    .reindex(design_summary_index, fill_value=0)
)
# An empty group's percentage is undefined (NaN), rather than zero percent.
design_summary['Functional (%)'] = (
    100 * design_summary['Functional designs']
    / design_summary['Number of designs'].replace(0, np.nan)
)
print('\nDesign counts and functional rates by model and divergence interval')
# Format percentages to one decimal place for display without rounding stored values.
display(design_summary.style.format({'Functional (%)': '{:.1f}%'}, na_rep='N/A'))

# Visual sanity check: show the first three designs in every model/divergence group.
# Use the configured group order and preserve original row indices for easy lookup.
# This preview does not filter or modify design_df or the encoded sequences.
preview_columns = ['name', 'seq', 'stored_E', 're', 'functional']
with pd.option_context('display.max_colwidth', None, 'display.max_columns', None):
    for model in MODELS:
        for divergence_bin in BINS:
            group_rows = design_df.loc[
                design_df['model'].eq(model) & design_df['bin'].eq(divergence_bin)
            ]
            preview_rows = group_rows.head(3)
            print(f'\n{model} | {divergence_bin} | '
                  f'showing {len(preview_rows)} of {len(group_rows)} designs')
            display(preview_rows[preview_columns])

# %% [markdown]
# ## Score every design with every model using adabmDCA
#
# Use package one-hot encoding and `compute_energy` in batches, without gradients.
# Each column `E_<model>` holds energies assigned by that scoring model to every design.
# These same energies are reused for validation and all downstream results.

# %%
# Store one NumPy energy array per scoring model, in the same row order as design_df.
E_ALL = {}
# Disable gradient tracking: we only evaluate fixed models, without training them.
with torch.no_grad():
    # Score every design with this model, regardless of which model generated it.
    for scorer in MODELS:
        # Collect the per-sequence energies from successive batches.
        batches = []
        # Split along the sequence dimension to limit intermediate memory usage.
        # Each batch has shape (batch_size, L); the final batch may be smaller.
        for batch in seqs_int.split(BATCH_SIZE):
            # Convert integer indices to one-hot vectors using adabmDCA.
            # Shape: (batch_size, L, q), with one active symbol per position.
            # Match the floating-point precision of the loaded model parameters.
            batch_oh = one_hot(batch, num_classes=q, dtype=dtype)
            # adabmDCA returns one energy per sequence using this scorer's parameters.
            # Lower energy means higher probability within this fixed model.
            # detach() explicitly removes gradient tracking (already disabled here);
            # cpu() makes the result available for conversion to NumPy.
            batches.append(compute_energy(batch_oh, PARAMS[scorer]).detach().cpu())
        # Concatenate batches in their original order into an array of shape (N,).
        E_ALL[scorer] = torch.cat(batches).numpy()
        # Stop if any calculated energy is NaN or infinite.
        if not np.isfinite(E_ALL[scorer]).all():
            raise ValueError(f'Non-finite energies from {scorer}')
        # Add this scorer's energies to the experimental table. Array assignment is
        # positional: each energy corresponds to the sequence in the same row.
        design_df[f'E_{scorer}'] = E_ALL[scorer]
        # Report completion for this scoring model.
        print(f'{scorer}: scored {len(design_df)} designs')

# %% [markdown]
# ## Validate stored energies and published functional rates
#
# As in the original notebook, report correlation and maximum absolute discrepancy
# against energies embedded in design names (expected discrepancies around $10^{-4}$,
# not machine precision). Print all functional rates next to Table 1 values.
# The additional `matches_table_1` column compares percentages rounded to one decimal.
# These are diagnostic reports: inspect discrepancies before interpreting the figures.

# %%
# VALIDATION 1: compare recalculated energies with energies recorded in design names.
# Collect one diagnostic row per generating model.
validation_rows = []
for model in MODELS:
    # Keep this model's own designs with an available recorded energy.
    # Other generators and missing recorded energies are excluded from this check.
    own = design_df.loc[design_df['model'].eq(model) & design_df['stored_E'].notna()]
    # Select the energies already calculated by the same model that generated these designs.
    calculated = own[f'E_{model}'].to_numpy()
    # Extract the recorded energies in the identical row order for pairwise comparison.
    stored = own['stored_E'].to_numpy()
    validation_rows.append({
        # n is the number of designs included in this model's energy comparison.
        'model': model, 'n': len(own),
        # Pearson correlation measures linear agreement; [0, 1] selects the cross-correlation.
        # A value near 1 alone does not rule out an energy offset or scale difference.
        # Fewer than two designs yield NaN; constant arrays can also make it undefined.
        'correlation': np.corrcoef(calculated, stored)[0, 1] if len(own) > 1 else np.nan,
        # Largest absolute difference across paired energies: zero means exact agreement.
        # Return NaN when there are no eligible designs.
        'max_abs_delta': np.max(np.abs(calculated - stored)) if len(own) else np.nan,
    })
# Display the diagnostics with one row per model. Discrepancies do not raise an error here.
energy_validation = pd.DataFrame(validation_rows).set_index('model')
display(energy_validation)

# VALIDATION 2: compare experimental functional rates with the published Table 1.
# Build all model/bin combinations in the configured order (MODELS first, then BINS).
group_index = pd.MultiIndex.from_product([MODELS, BINS], names=['model', 'bin'])
# Summing Boolean labels counts functional designs; size counts all tested designs.
# Reindexing includes every expected combination, exposing absent groups as NaN.
functional_rates = (design_df.groupby(['model', 'bin'])['functional']
                    .agg(functional_count='sum', total='size').reindex(group_index))
# Stop if an expected model/divergence group is absent from the loaded experiments.
if functional_rates['total'].isna().any():
    raise ValueError('Missing model/divergence group in the experimental CSVs.')
# Percentage functional = 100 times functional count divided by total design count.
functional_rates['percent'] = 100 * functional_rates['functional_count'] / functional_rates['total']
# Published reference percentages, entered explicitly in MODELS/BINS order:
# eaDCA (three bins), bmDCA (three bins), edDCA (three bins), meDCA (three bins).
# Update this ordering if the configured model or bin order changes.
functional_rates['table_1_percent'] = [31.8, 12.6, 2.0, 28.6, 11.9, 1.3,
                                      23.9, 5.2, 0.0, 30.9, 5.2, 0.4]
# Compare at the one-decimal precision reported in Table 1.
# False flags a mismatch for inspection; it does not automatically stop execution.
functional_rates['matches_table_1'] = (functional_rates['percent'].round(1)
                                      == functional_rates['table_1_percent'])
# Show counts, calculated percentages, reference percentages and agreement flags.
display(functional_rates)

# %% [markdown]
# ## Result 1 — energy-penalty matrices
#
# At fixed divergence, rows are generating models $G$ and columns scoring models $S$:
# $$M_{G,S}=\operatorname{median}_{A\in F_{G,b}}E_S(A)
# -\operatorname{median}_{A\in F_{S,b}}E_S(A).$$
# Only **functional designs** enter both medians. A positive penalty means higher
# median energy relative to the scorer's own functional designs. The comparisons
# remain conditional on the curated, experimentally tested libraries.

# %%
# Organize energies already calculated in the scoring cell; no energies are recomputed.
# generator = model that produced the design; scorer = model assigning its energy.
# functional_energies stores functional-only subsets; reference_energies stores
# all own-model tested designs for the later percentile/beyond_max analysis.
functional_energies, reference_energies = {}, {}
# Visit every generating model.
for generator in MODELS:
    # Visit every configured divergence interval.
    for b in BINS:
        # Boolean row selector for designs from this generator in this interval.
        group_mask = design_df['model'].eq(generator) & design_df['bin'].eq(b)
        # Retrieve this group's energies under each of the four scoring models.
        for scorer in MODELS:
            # Select only functional designs and extract the scorer's energy column.
            # The dictionary key is the tuple (scorer, generator, divergence bin).
            # to_numpy() returns a 1D array; it can be empty if no designs are functional.
            functional_energies[scorer, generator, b] = design_df.loc[
                group_mask & design_df['functional'], f'E_{scorer}'].to_numpy()
        # Store ALL designs in the group, including nonfunctional designs, scored
        # by their own generator. This reference is NOT used for the penalty medians.
        reference_energies[generator, b] = design_df.loc[group_mask, f'E_{generator}'].to_numpy()

# Store one median energy-penalty matrix per plotted divergence interval.
PENALTY = {}
# The plotted intervals are the first two configured bins (20-25% and 40-45%).
for b in PLOT_BINS:
    # Start a 4-by-4 matrix with undefined entries (NaN).
    # Rows represent generators; columns represent scorers, both in MODELS order.
    matrix = np.full((len(MODELS), len(MODELS)), np.nan)
    # enumerate supplies both the row index gi and the generating-model name.
    for gi, generator in enumerate(MODELS):
        # si is the column index for this scoring model.
        for si, scorer in enumerate(MODELS):
            # Functional designs from the row generator, evaluated by the column scorer.
            other = functional_energies[scorer, generator, b]
            # Functional designs generated AND evaluated by that same scorer.
            # Both arrays therefore use the same energy function and divergence bin.
            own = functional_energies[scorer, scorer, b]
            # Compute the difference only if both functional subsets are nonempty.
            # Otherwise leave NaN, since the comparison is undefined.
            if len(other) and len(own):
                # Positive: the generator's functional designs have higher median energy.
                # Negative: they have lower median energy. The diagonal is zero
                # whenever defined, because the two subsets are identical there.
                matrix[gi, si] = np.median(other) - np.median(own)
    # Keep full-precision values for subsequent tables and plots.
    PENALTY[b] = matrix
    # Label the interval and matrix orientation explicitly.
    print(f'Energy penalty, {b}: rows = generator, columns = scorer')
    # Add model names to both axes; round only the displayed table to one decimal.
    display(pd.DataFrame(matrix, index=MODELS, columns=MODELS).round(1))
    # Summarize each scorer's penalties on the other three generators.
    for si, scorer in enumerate(MODELS):
        # Select this scorer's column, excluding its diagonal (own-generator) entry.
        others = matrix[np.arange(len(MODELS)) != si, si]
        # Average the available off-diagonal penalties, ignoring NaN entries.
        # Each generator receives equal weight, regardless of its number of designs.
        # +.1f prints an explicit sign and one decimal; all-NaN input remains undefined.
        print(f'  Mean penalty imposed by {scorer}: {np.nanmean(others):+.1f}')

# %% [markdown]
# ### LaTeX for Table 1 of the letter

# %%
for b in ["20-25%", "40-45%"]:
    M = PENALTY[b]
    print(f"% {b} divergence")
    for gi, G in enumerate(MODELS):
        print(f"{G} & " + " & ".join(f"${M[gi, si]:+.1f}$" for si in range(len(MODELS))) + r" \\")
    print()

# %% [markdown]
# ## Figure — energy-penalty heatmap
#
# Same matrices and fixed color scale ($-35$ to $+35$) as the original.
# Model labels use the repository palette; the diverging cell colors encode signed energy gaps.
# The entropy labels are supplied constants, not estimates from this notebook.

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), layout='constrained')
for panel, ax, b in zip(['A', 'B'], axes, PLOT_BINS):
    matrix = PENALTY[b]
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-35, vmax=35, aspect='equal')
    ax.set_xticks(range(4), [f'{m}\n$S={ENTROPY[m]:.0f}$' for m in MODELS])
    ax.set_yticks(range(4), MODELS)
    ax.set_xlabel('Scoring model', labelpad=12)
    ax.set_ylabel('Generating model', labelpad=12)
    ax.set_title(f'{bin_labels[b]} divergence', pad=16)
    ax.text(-0.20, 1.10, panel, transform=ax.transAxes, fontsize=26, fontweight='bold')
    ax.tick_params(length=0, pad=8)
    for label, model in zip(ax.get_xticklabels(), MODELS):
        label.set_color(colors[model])
    for label, model in zip(ax.get_yticklabels(), MODELS):
        label.set_color(colors[model])
    for gi in range(4):
        for si in range(4):
            value = matrix[gi, si]
            ax.text(si, gi, f'{value:+.0f}', ha='center', va='center', fontsize=28,
                    color='white' if abs(value) > 18 else 'black')
colorbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.03, fraction=0.035)
colorbar.set_label('Median energy penalty', labelpad=14)
fig.savefig(FIGURES_SUPPLEMENTARY / 'figS2_energy_penalty_matrix.pdf', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Figure — eaDCA/meDCA scoring asymmetry
#
# For each scorer, subtract the median energy of its own functional designs.
# Red: eaDCA functional designs scored by meDCA. Blue: meDCA functional designs
# scored by eaDCA. Each distribution uses the **generator's** usual color.
# The bin edges and normalized densities are identical to the original analysis.

# %%
# Histogram colors identify the generating model: eaDCA red, meDCA blue.
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), layout='constrained')
ASYMMETRY = {}
for panel, ax, b in zip(['A', 'B'], axes, PLOT_BINS):
    low_scorer = functional_energies['eaDCA', 'meDCA', b] - np.median(
        functional_energies['eaDCA', 'eaDCA', b])
    high_scorer = functional_energies['meDCA', 'eaDCA', b] - np.median(
        functional_energies['meDCA', 'meDCA', b])
    edges = np.linspace(min(low_scorer.min(), high_scorer.min()) - 3,
                        max(low_scorer.max(), high_scorer.max()) + 3, 32)
    ASYMMETRY[b] = {'eaDCA': low_scorer, 'meDCA': high_scorer, 'edges': edges}
    for values, scorer, generator in [(high_scorer, 'meDCA', 'eaDCA'),
                                       (low_scorer, 'eaDCA', 'meDCA')]:
        ax.hist(values, bins=edges, density=True, histtype='stepfilled',
                color=colors[generator], alpha=0.5, edgecolor='none',
                label=f'{generator} designs\nscored by {scorer}')
        ax.hist(values, bins=edges, density=True, histtype='step',
                color=colors[generator], alpha=0.9, linewidth=2.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
    ax.set_title(f'{bin_labels[b]} divergence', pad=16)
    ax.text(-0.15, 1.08, panel, transform=ax.transAxes, fontsize=26, fontweight='bold')
    ax.set_xlabel("Energy relative to scorer's\nown functional designs", labelpad=12)
    ax.set_ylabel('Density', labelpad=10)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=1)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)  # Leave room for the legend above the data.
    ax.legend(fontsize=16, loc='upper right')
fig.savefig(FIGURES_INTERMEDIATE / 'cross_model_energy_asymmetry.pdf', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Result 2 — functional meDCA designs beyond the tested energy range
#
# For each functional meDCA design, compare its energy under each scorer with that
# scorer's **all tested designs** at the same divergence, including nonfunctional ones.
# The percentile is $100\,\mathrm{mean}(E_{\rm ref}<E)$ (strict inequality; ties excluded).
# `beyond_max` uses $E>\max(E_{\rm ref})$; `beyond_all_three` combines eaDCA, bmDCA and edDCA.
# These are the same statistics as the original; an out-of-range flag does not prove
# zero generation probability. Print flagged 20–25% designs in decreasing enrichment order.

# %%
# Build per-design records for functional meDCA sequences across all divergence bins.
record_frames = []
# Analyze each divergence interval separately to use matched reference libraries.
for b in BINS:
    # Retain only experimentally functional designs generated by meDCA in this bin.
    selected = design_df.loc[design_df['model'].eq('meDCA') & design_df['bin'].eq(b)
                             & design_df['functional']]
    # Copy divergence, enrichment and sequence into a separate results table.
    # Adding result columns below does not modify design_df.
    records_b = selected[['bin', 're', 'seq']].copy()
    # Insert a short sequence identifier as the first column. Split the full name
    # at the literal pipe before DCAenergy, allowing surrounding whitespace;
    # keep the preceding text and remove leading/trailing whitespace.
    records_b.insert(0, 'sid', selected['name'].str.split(
        r'\s*\|\s*DCAenergy', regex=True).str[0].str.strip())
    # Evaluate relative energy ranks under each of the four scoring models.
    for scorer in MODELS:
        # Reference: ALL tested designs generated by this scorer in the same bin,
        # evaluated by that scorer, including functional and nonfunctional designs.
        reference = reference_energies[scorer, b]
        # Energies already assigned by this scorer to the selected functional meDCA designs.
        values = selected[f'E_{scorer}'].to_numpy()
        # Broadcasting compares every selected energy with every reference energy:
        # reference[None, :] has shape (1, R); values[:, None] has shape (N, 1).
        # The comparison has shape (N, R); its row mean is the fraction of reference
        # energies STRICTLY below each selected energy. Multiply by 100 for percent.
        # Equal energies are excluded; this is an empirical rank, not generation probability.
        percentiles = 100 * (reference[None, :] < values[:, None]).mean(axis=1)
        # Store each empirical percentile rounded to one decimal place.
        records_b[f'{scorer}_pct'] = [round(float(value), 1) for value in percentiles]
        # Flag energies strictly above the largest energy in this tested reference.
        # Use unrounded energies; equality with the maximum is not flagged.
        records_b[f'{scorer}_beyond_max'] = values > reference.max()
    # Require all three non-meDCA flags to be True for the same design.
    # meDCA_beyond_max is calculated above but is not part of this combined flag.
    records_b['beyond_all_three'] = records_b[
        [f'{scorer}_beyond_max' for scorer in ['eaDCA', 'bmDCA', 'edDCA']]].all(axis=1)
    # Collect this bin's full results, including designs without the combined flag.
    record_frames.append(records_b)
# Combine all bins, assigning consecutive row indices. Every functional meDCA design is retained.
records_df = pd.concat(record_frames, ignore_index=True)
# For the printed shortlist only, keep flagged designs in the 20-25% bin.
# Sort by decreasing enrichment; stable sorting preserves row order when enrichment ties.
star = records_df.loc[records_df['bin'].eq('20-25%') & records_df['beyond_all_three']
                     ].sort_values('re', ascending=False, kind='stable')
# Report the number of shortlisted designs. Being outside a tested energy range
# does not imply that the model has zero probability of generating the sequence.
print(f'{len(star)} functional meDCA designs at 20-25% are beyond all three tested ranges.')
# Print aligned column headings: identifier, enrichment and scorer-specific percentiles.
print(f'{"enzyme":24s} {"RE":>6} {"meDCA%":>7} {"eaDCA%":>7} {"bmDCA%":>7} {"edDCA%":>7}')
# Convert the shortlist to row dictionaries for formatted text output.
for r in star.to_dict('records'):
    # Display enrichment with a sign and two decimals; meDCA percentile with no
    # decimals and other percentiles with one. The stored percentiles remain unchanged.
    print(f'{r["sid"]:24s} {r["re"]:+6.2f} {r["meDCA_pct"]:7.0f} '
          f'{r["eaDCA_pct"]:7.1f} {r["bmDCA_pct"]:7.1f} {r["edDCA_pct"]:7.1f}')
    # Print the complete aligned sequence beneath its summary.
    print('   ' + r['seq'])

# %% [markdown]
# ### Export the full supplementary CSV (same columns and ordering as the original)

# %%
cols = ['sid', 'bin', 're', 'beyond_all_three',
        'eaDCA_beyond_max', 'bmDCA_beyond_max', 'edDCA_beyond_max',
        'meDCA_pct', 'eaDCA_pct', 'bmDCA_pct', 'edDCA_pct', 'seq']
supplement = records_df.sort_values(['bin', 're'], ascending=[True, False], kind='stable')[cols]
out = RESULTS_DIR / 'functional_meDCA_enzymes_full.csv'
supplement.to_csv(out, index=False)
print(f'Wrote {out}: {len(supplement)} functional meDCA designs; '
      f'{int(supplement["beyond_all_three"].sum())} flagged beyond_all_three')

# %% [markdown]
# ### LaTeX for Table 2 of the letter
#
# Use the same four explicitly named designs as the original notebook.
# This is a fixed selection for reproducing the letter, not an automatic top-four ranking.

# %%
letter_four = ["meDCA batch 1 | seq 147", "meDCA batch 1 | seq 72",
               "meDCA batch 1 | seq 250", "meDCA batch 1 | seq 233"]
by_sid = records_df.set_index("sid").to_dict("index")
for sid in letter_four:
    r = by_sid[sid]
    tag = "meDCA-" + sid.split()[-1]
    print(f'{tag} & ${r["re"]:+.2f}$ & {r["meDCA_pct"]:.0f} '+r'\\')
    print(r'\multicolumn{3}{@{}l@{}}{\ttfamily\scriptsize ' + r["seq"] + r'}\\')

# %% [markdown]
# ## Optional — pairwise rank statistic
#
# Fraction of all functional-design pairs for which the other generator receives
# strictly higher energy than the scorer's own design. Ties contribute zero, as in
# the original. This measures ordering rather than energy-gap magnitude; it is not a p-value.

# %%
# Store the fraction of pairwise comparisons in which the other generator's
# functional design has higher energy than the scorer's own functional design.
# Keys are (scorer, generator, divergence bin). Energies are not recomputed here.
RANK = {}
# Compare the two models separately within each plotted divergence interval.
for b in PLOT_BINS:
    # Evaluate both directions: eaDCA scores meDCA designs, then meDCA scores eaDCA designs.
    # These use different energy functions and need not yield complementary fractions.
    for scorer, generator in [('eaDCA', 'meDCA'), ('meDCA', 'eaDCA')]:
        # Functional designs from the other generator, evaluated by the selected scorer.
        other = functional_energies[scorer, generator, b]
        # The scorer's own functional designs, evaluated by that same scorer.
        # Both groups are functional-only and belong to the same divergence bin.
        own = functional_energies[scorer, scorer, b]
        # Broadcasting compares every other-group energy against every own-group energy.
        # Shapes (N_other, 1) and (1, N_own) produce an (N_other, N_own) Boolean array.
        # np.mean averages ALL entries: True counts as 1, False as 0.
        # Strict > means equal energies count as False. If either group is empty,
        # return NaN because the comparison is undefined.
        # This empirical pairwise fraction measures ordering, not energy-gap magnitude,
        # a statistical p-value, or the probability of generating a sequence.
        RANK[scorer, generator, b] = (np.mean(other[:, None] > own[None, :])
                                     if len(other) and len(own) else np.nan)
    # Print both directions to two decimal places; full precision remains in RANK.
    # "Worse" means higher energy (lower probability within this fixed scoring model).
    print(f"{b}: eaDCA on meDCA-functional P(worse)={RANK['eaDCA', 'meDCA', b]:.2f} | "
          f"meDCA on eaDCA-functional P(worse)={RANK['meDCA', 'eaDCA', b]:.2f}")

# %% [markdown]
# ## Scope and interpretation
#
# - This script preserves the original numerical analysis. It does not train models,
#   generate sequences, estimate entropy, or recalculate experimental classifications.
# - Energies compare sequence probabilities within a fixed model. Penalties subtract
#   energies computed by the same scorer; raw energies across models are not calibrated probabilities.
# - meDCA was tested in a separate experimental batch. This does not affect a fixed
#   sequence's computed energy, but it can affect which designs enter the functional subsets.
# - The reference libraries are curated tested designs. `beyond_max` means outside their
#   observed energy range, not impossible to generate. Divergence matching and median-based
#   comparisons do not remove effects of library selection.
# - Re-running overwrites only this notebook's output files in its dedicated folder.

# %% [markdown]
# ## Combined figure — scoring asymmetry and energy penalties
#
# Top row (**A**): the two energy-asymmetry histograms. Bottom row (**B**):
# the corresponding penalty heatmaps, with the same divergence interval in each column.
# Rectangular plotting boxes share identical widths and heights; the colorbar is placed outside
# the grid. Data, normalization, model colors and typography match the individual figures.
# Run the preceding setup, scoring and penalty cells first.
# Saved as `fig03_cross_model_energy_penalty.pdf` in `figures/main/`.

# %%
# Histogram colors identify the generating model: eaDCA red, meDCA blue.
# Combine the existing analyses: histograms on top, penalty heatmaps below.
# Identical rectangular axes and fixed grid margins keep all four plotting boxes aligned.
# The colorbar occupies an inset outside the lower-right axes, without shrinking it.
combined_fig, combined_axes = plt.subplots(
    2, 2, figsize=(17, 12), layout=None,
    subplot_kw={'box_aspect': 0.72},
    gridspec_kw={'left': 0.075, 'right': 0.90, 'bottom': 0.07, 'top': 0.94,
                 'wspace': 0.16, 'hspace': 0.58},
)

for column, b in enumerate(PLOT_BINS):
    # Top row: use the same centered energies, bin edges and histogram styling.
    histogram_ax = combined_axes[0, column]
    low_scorer = functional_energies['eaDCA', 'meDCA', b] - np.median(
        functional_energies['eaDCA', 'eaDCA', b])
    high_scorer = functional_energies['meDCA', 'eaDCA', b] - np.median(
        functional_energies['meDCA', 'meDCA', b])
    edges = np.linspace(min(low_scorer.min(), high_scorer.min()) - 3,
                        max(low_scorer.max(), high_scorer.max()) + 3, 32)
    for values, scorer, generator in [(high_scorer, 'meDCA', 'eaDCA'),
                                      (low_scorer, 'eaDCA', 'meDCA')]:
        histogram_ax.hist(
            values, bins=edges, density=True, histtype='stepfilled',
            color=colors[generator], alpha=0.5, edgecolor='none',
            label=f'{generator} designs\nscored by {scorer}',
        )
        histogram_ax.hist(
            values, bins=edges, density=True, histtype='step',
            color='black', linewidth=1.5,
        )
    histogram_ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
    histogram_ax.set_title(f'{bin_labels[b]} divergence', pad=16)
    histogram_ax.set_xlabel("Energy relative to scorer's\nown functional designs", labelpad=12)
    if column == 0:
        histogram_ax.set_ylabel('Density', labelpad=10)
    histogram_ax.grid(True, alpha=0.25, linestyle='--', linewidth=1)
    histogram_ax.set_ylim(0, histogram_ax.get_ylim()[1] * 1.55)
    histogram_ax.legend(fontsize=22, loc='upper right')

    # Bottom row: preserve matrix values, color scale, annotations and model colors.
    heatmap_ax = combined_axes[1, column]
    matrix = PENALTY[b]
    combined_image = heatmap_ax.imshow(
        matrix, cmap='RdBu_r', vmin=-35, vmax=35, aspect='auto',
    )
    heatmap_ax.set_xticks(range(4), [f'{m}\n$S={ENTROPY[m]:.0f}$' for m in MODELS])
    heatmap_ax.set_yticks(range(4), MODELS)
    heatmap_ax.set_xlabel('Scoring model', labelpad=12)
    if column == 0:
        heatmap_ax.set_ylabel('Generating model', labelpad=12)
    heatmap_ax.tick_params(length=0, pad=8)
    for label, model in zip(heatmap_ax.get_xticklabels(), MODELS):
        label.set_color(colors[model])
    for label, model in zip(heatmap_ax.get_yticklabels(), MODELS):
        label.set_color(colors[model])
    for gi in range(4):
        for si in range(4):
            value = matrix[gi, si]
            heatmap_ax.text(
                si, gi, f'{value:+.0f}', ha='center', va='center', fontsize=28,
                color='white' if abs(value) > 18 else 'black',
            )

# A dedicated colorbar preserves the common widths and left/right edges of the panels.
combined_cax = combined_axes[1, 1].inset_axes([1.06, 0.09, 0.045, 0.82])
combined_colorbar = combined_fig.colorbar(combined_image, cax=combined_cax)
combined_colorbar.set_label('Median energy penalty', labelpad=14)

# One letter per row, in the left margin and above the corresponding plotting boxes.
combined_fig.canvas.draw()
for row, letter in enumerate(['A', 'B']):
    row_box = combined_axes[row, 0].get_position()
    combined_fig.text(
        row_box.x0 - 0.062, row_box.y1 + 0.022, letter,
        fontsize=38, fontweight='bold', ha='left', va='bottom',
    )

combined_output = FIGURES_MAIN / 'fig03_cross_model_energy_penalty.pdf'
combined_fig.savefig(combined_output, bbox_inches='tight')
combined_fig.savefig(combined_output.with_suffix('.png'), bbox_inches='tight', dpi=300)
plt.show()
print(f'Saved combined figure: {combined_output}')

# %% [markdown]
# ## Mean-based energy penalty and combined figure
#
# Repeat the cross-model comparison using the **arithmetic mean** instead of the median.
# The experimental designs, functional labels, divergence intervals and DCA energies are unchanged.
# For generator $G$, scorer $S$, and divergence bin $b$, define
#
# $$M^{\mathrm{mean}}_{G,S}(b)
# =\frac{1}{|F_{G,b}|}\sum_{A\in F_{G,b}} E_S(A)
# -\frac{1}{|F_{S,b}|}\sum_{A\in F_{S,b}} E_S(A),$$
#
# where $F_{G,b}$ contains only the **functional tested designs** generated by $G$ in bin $b$.
# Rows are generators and columns are scorers. Positive values indicate higher average energy
# relative to the scorer's own functional designs. Unlike the median, the arithmetic mean
# is sensitive to the magnitude of extreme energies. Empty functional subsets yield undefined entries.
#
# In row **A**, each histogram is now centered on the **mean energy of the scorer's own functional
# designs**. Thus the mean of each displayed distribution equals the corresponding entry in
# the new penalty matrix. Row **B** displays those mean-based matrices. The compact layout,
# model palette, density normalization and fixed heatmap scale ($-35$ to $+35$) are preserved;
# histogram bin edges are recalculated from the mean-centered values using the same rule as before.
# The color scale saturates outside its limits, while cell annotations show the calculated values.
#
# The original `PENALTY` matrices and median-based PDF remain available. Percentiles,
# `beyond_max` flags and pairwise rank statistics do not use these centers and remain unchanged.
# Run the preceding setup, energy-scoring and functional-subset cells before this section.
# The new figure is saved separately as `cross_model_energy_penalty_mean.pdf` in `figures/intermediate/`.

# %%
# Repeat the penalty calculation on the same functional subsets, using arithmetic means.
# Keep separate results so the median-based analysis remains available for comparison.
PENALTY_MEAN = {}
for b in PLOT_BINS:
    mean_matrix = np.full((len(MODELS), len(MODELS)), np.nan)
    for gi, generator in enumerate(MODELS):
        for si, scorer in enumerate(MODELS):
            other = functional_energies[scorer, generator, b]
            own = functional_energies[scorer, scorer, b]
            # Both energies are evaluated by the same scorer within the same divergence bin.
            if len(other) and len(own):
                mean_matrix[gi, si] = np.mean(other) - np.mean(own)
    PENALTY_MEAN[b] = mean_matrix
    print(f'Mean energy penalty, {b}: rows = generator, columns = scorer')
    display(pd.DataFrame(mean_matrix, index=MODELS, columns=MODELS).round(1))
    for si, scorer in enumerate(MODELS):
        # Give each of the other generators equal weight in this column summary.
        other_penalties = mean_matrix[np.arange(len(MODELS)) != si, si]
        print(f'  Mean penalty imposed by {scorer}: {np.nanmean(other_penalties):+.1f}')

# %%
# Histogram colors identify the generating model: eaDCA red, meDCA blue.
# Mean-based variant: mean-centered histograms above mean-penalty heatmaps.
# Identical rectangular axes and fixed grid margins keep all four plotting boxes aligned.
# The colorbar occupies an inset outside the lower-right axes, without shrinking it.
mean_combined_fig, mean_combined_axes = plt.subplots(
    2, 2, figsize=(17, 12), layout=None,
    subplot_kw={'box_aspect': 0.72},
    gridspec_kw={'left': 0.075, 'right': 0.90, 'bottom': 0.07, 'top': 0.94,
                 'wspace': 0.16, 'hspace': 0.58},
)

ASYMMETRY_MEAN = {}
for column, b in enumerate(PLOT_BINS):
    # Top row: subtract the scorer's OWN mean functional energy, then rebuild bin edges.
    histogram_ax = mean_combined_axes[0, column]
    low_scorer = functional_energies['eaDCA', 'meDCA', b] - np.mean(
        functional_energies['eaDCA', 'eaDCA', b])
    high_scorer = functional_energies['meDCA', 'eaDCA', b] - np.mean(
        functional_energies['meDCA', 'meDCA', b])
    edges = np.linspace(min(low_scorer.min(), high_scorer.min()) - 3,
                        max(low_scorer.max(), high_scorer.max()) + 3, 32)
    ASYMMETRY_MEAN[b] = {'eaDCA': low_scorer, 'meDCA': high_scorer, 'edges': edges}
    for values, scorer, generator in [(high_scorer, 'meDCA', 'eaDCA'),
                                      (low_scorer, 'eaDCA', 'meDCA')]:
        histogram_ax.hist(
            values, bins=edges, density=True, histtype='stepfilled',
            color=colors[generator], alpha=0.5, edgecolor='none',
            label=f'{generator} designs\nscored by {scorer}',
        )
        histogram_ax.hist(
            values, bins=edges, density=True, histtype='step',
            color='black', linewidth=1.5,
        )
    histogram_ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
    histogram_ax.set_title(f'{bin_labels[b]} divergence', pad=16)
    histogram_ax.set_xlabel("Energy relative to mean of\nscorer's own functional designs", labelpad=12)
    if column == 0:
        histogram_ax.set_ylabel('Density', labelpad=10)
    histogram_ax.grid(True, alpha=0.25, linestyle='--', linewidth=1)
    histogram_ax.set_ylim(0, histogram_ax.get_ylim()[1] * 1.55)
    histogram_ax.legend(fontsize=22, loc='upper right')

    # Bottom row: preserve matrix values, color scale, annotations and model colors.
    heatmap_ax = mean_combined_axes[1, column]
    matrix = PENALTY_MEAN[b]
    mean_combined_image = heatmap_ax.imshow(
        matrix, cmap='RdBu_r', vmin=-35, vmax=35, aspect='auto',
    )
    heatmap_ax.set_xticks(range(4), [f'{m}\n$S={ENTROPY[m]:.0f}$' for m in MODELS])
    heatmap_ax.set_yticks(range(4), MODELS)
    heatmap_ax.set_xlabel('Scoring model', labelpad=12)
    if column == 0:
        heatmap_ax.set_ylabel('Generating model', labelpad=12)
    heatmap_ax.tick_params(length=0, pad=8)
    for label, model in zip(heatmap_ax.get_xticklabels(), MODELS):
        label.set_color(colors[model])
    for label, model in zip(heatmap_ax.get_yticklabels(), MODELS):
        label.set_color(colors[model])
    for gi in range(4):
        for si in range(4):
            value = matrix[gi, si]
            heatmap_ax.text(
                si, gi, f'{value:+.0f}', ha='center', va='center', fontsize=28,
                color='white' if abs(value) > 18 else 'black',
            )

# A dedicated colorbar preserves the common widths and left/right edges of the panels.
mean_combined_cax = mean_combined_axes[1, 1].inset_axes([1.06, 0.09, 0.045, 0.82])
mean_combined_colorbar = mean_combined_fig.colorbar(mean_combined_image, cax=mean_combined_cax)
mean_combined_colorbar.set_label('Mean energy penalty', labelpad=14)

# One letter per row, in the left margin and above the corresponding plotting boxes.
mean_combined_fig.canvas.draw()
for row, letter in enumerate(['A', 'B']):
    row_box = mean_combined_axes[row, 0].get_position()
    mean_combined_fig.text(
        row_box.x0 - 0.062, row_box.y1 + 0.022, letter,
        fontsize=38, fontweight='bold', ha='left', va='bottom',
    )

mean_combined_output = FIGURES_INTERMEDIATE / 'cross_model_energy_penalty_mean.pdf'
mean_combined_fig.savefig(mean_combined_output, bbox_inches='tight')
plt.show()
print(f'Saved mean-based combined figure: {mean_combined_output}')
