#!/usr/bin/env python3
"""
Shared paths of the reproducibility package.

Every script imports its input and output locations from here, so no path is
hard-coded twice and the package can be moved or renamed without editing any
analysis code. All paths are derived from the location of this file, so the
scripts can be run from any working directory.

Layout
------
    data/natural_msa/        natural Chorismate Mutase MSA and sequence weights
    data/entropy_arrays/     pre-computed entropies, energies and divergences
    data/leave_one_out/      leave-one-group-out MSAs and retrained models
    data/generated_samples/  large equilibrium samples (2M sequences per model)
    data/transformer/        FASTA files used by the GPT scripts
    data/cache/              cached energies/divergences of the large samples
    data/archives/           zip archives of all of the above
    models/                  DCA parameters under their development names
    figures/main/            figures of the main text
    figures/supplementary/   figures of the Supplemental Material
    figures/intermediate/    exploratory panels, not used in the paper
    results/                 numerical results (CSV, JSON, logs)
"""

from pathlib import Path

# Root of the package (the parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent

# Root of the git repository, which holds the released models and the
# experimental data shared with the rest of the project
REPO_ROOT = ROOT.parent

# ---------------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
NATURAL_MSA_DIR = DATA_DIR / "natural_msa"
NATURAL_MSA = NATURAL_MSA_DIR / "CM_130530_MC.fasta"
NATURAL_WEIGHTS = NATURAL_MSA_DIR / "weights.dat"

ENTROPY_ARRAYS_DIR = DATA_DIR / "entropy_arrays"
LEAVE_ONE_OUT_DIR = DATA_DIR / "leave_one_out"
FILTERED_MSA_DIR = LEAVE_ONE_OUT_DIR / "filtered_msa"
TAKE_1OUT_MODELS_DIR = LEAVE_ONE_OUT_DIR / "take_1out_models"
GENERATED_SAMPLES_DIR = DATA_DIR / "generated_samples"
TRANSFORMER_DATA_DIR = DATA_DIR / "transformer"
CACHE_DIR = DATA_DIR / "cache"
ARCHIVES_DIR = DATA_DIR / "archives"

# DCA parameters under their development names:
#   meDCA = Parameters_edDCA_density=0.125_high_entropy, edDCA = Parameters_conv_dec_new,
#   eaDCA = Parameters_EAALG0.8_new,                     bmDCA = Parameters_conv_dense_new
MODELS_DIR = ROOT / "models"

# Released parameters and in vivo measurements, shared with the rest of the repository
RELEASED_MODELS_DIR = REPO_ROOT / "models" / "tested_models"
EXPERIMENT_DIR = REPO_ROOT / "data" / "experiment"

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
FIGURES_DIR = ROOT / "figures"
FIGURES_MAIN = FIGURES_DIR / "main"
FIGURES_SUPPLEMENTARY = FIGURES_DIR / "supplementary"
FIGURES_INTERMEDIATE = FIGURES_DIR / "intermediate"
RESULTS_DIR = ROOT / "results"

for _directory in (FIGURES_MAIN, FIGURES_SUPPLEMENTARY, FIGURES_INTERMEDIATE,
                   RESULTS_DIR, CACHE_DIR):
    _directory.mkdir(parents=True, exist_ok=True)
