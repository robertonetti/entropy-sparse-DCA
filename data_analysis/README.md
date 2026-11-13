# Figure and Results Reproducibility Package

Everything, and only what is needed, to regenerate the figures and the numerical
results of the paper. Each analysis is a plain Python script; the code is the code of
the original notebooks, in the same order, so the outputs are identical. Cells are kept
as `# %%` delimiters, so every script can still be run cell by cell in VSCode or
Jupytext.

## Layout

```
scripts/        analysis scripts (see the table below) and paths.py
data/           archives/ holds every input as a zip; the other folders are what
                they unpack to, and are not tracked by git
figures/        main/ · supplementary/ · intermediate/
results/        numerical results (CSV, JSON, logs)
```

`scripts/paths.py` holds every input and output location, so no path is hard-coded
twice and a script can be run from any working directory.

## Figure map

| Paper figure | Script | Output |
| --- | --- | --- |
| Fig. 1 | — | schematic, drawn by hand |
| Fig. 2 | `entropy_and_energy_landscape.py` | `figures/main/fig02_energy_distributions_and_effective_sequences.pdf` |
| Fig. 2 (left panel alone) | `entropy_and_energy_landscape.py` | `figures/main/fig02a_energy_distributions_grid.pdf` |
| Fig. 3 | `cross_model_scoring.py` | `figures/main/fig03_cross_model_energy_penalty.pdf` |
| Fig. 4 | `generalization_mutability_figure.py` | `figures/main/fig04_generalization_and_mutability.pdf` |
| Fig. 5 | `entropy_and_energy_landscape.py` | `figures/main/fig05_entropy_vs_parameter_density.pdf` |
| Fig. S1 | `experimental_functional_rates.py` | `figures/supplementary/figS1_experimental_functional_rates.pdf` |
| Fig. S2 | `cross_model_scoring.py` | `figures/supplementary/figS2_energy_penalty_matrix.pdf` |
| Fig. S3 | `leave_one_out_generalization.py` | `figures/supplementary/figS3_leave_one_out_pca.pdf` |
| Fig. S4 | `leave_one_out_generalization.py` | `figures/supplementary/figS4_excluded_sequence_divergence.pdf` |
| Fig. S5 | `leave_one_out_generalization.py` | `figures/supplementary/figS5_leave_one_out_energy_shift.png` |
| Fig. S6 | `mutability.py` | `figures/supplementary/figS6_mutability_all_models.pdf` |
| Fig. S7 | `mixing_time.py` | `figures/supplementary/figS7_mixing_time.png` |
| Fig. S8 | `leave_one_out_generalization.py` | `figures/supplementary/figS8_entropy_vs_density_all_families.pdf` |
| Fig. S9 | `sequence_diversity.py` | `figures/supplementary/figS9_sequence_diversity.pdf` |
| Tab. S1 | `transformer/gpt_inference.py` | printed to stdout |

The two panels of Fig. 4 are also produced on their own, by
`leave_one_out_generalization.py` (left) and `mutability.py` (right), in
`figures/intermediate/`. That folder holds the exploratory panels that are not in the
paper; nothing there is a published figure.

Support scripts:

- `entropy_decile_correction.py` — the decile correction of the entropies (below).
- `leave_one_out_msa_preparation.py` — builds the leave-one-group-out training sets.
- `transformer/{transformer,mc,transformer_train,Transformer_Reint}.py` — the GPT model
  and training code imported by `gpt_inference.py`.

## The entropy correction (Fig. 2, right panel)

The experimentally tested designs were selected from the **lowest-energy decile** of
each model, so the entropy of each divergence bin is restricted to that decile before it
is converted into an effective number of sequences:

```
S_{B,10} = S_B - (<E>_B - <E>_{B,10}) + log(0.1)
```

`entropy_and_energy_landscape.py` applies this correction directly: it imports
`entropy_decile_correction.py` and plots `S_bin_10` in the right panel of Fig. 2. The
published version of the figure was produced **before** the correction (its y axis
reaches 10^76 instead of 10^68), so the figure regenerated here is the one that must
replace it.

`entropy_decile_correction.py` can also be run on its own: it prints the corrected
numbers, writes `results/entropy_correction_results.json` and saves two control figures
in `figures/intermediate/` — the `S_tot` vs `S_10` barplot, and a variant of Fig. 2 in
which the bin entropies are recomputed from scratch instead of being read from the
stored arrays (the two agree within ~1 nat).

## Data

Everything the scripts read is in `data/archives/`, which is what the repository
tracks (through git-lfs). Unpack it once, from the `data/` folder:

```bash
cd data && for f in archives/*.zip; do unzip -o "$f"; done
mkdir -p cache && unzip -o archives/caches.zip -d cache
```

`caches.zip` holds the cached energies and divergences of the large samples and the
cached panel data of Fig. 4. With it, Fig. 2 and Fig. 4 can be regenerated **without**
unpacking the 427 MB of `*_large_sample.zip`.

| Path | Used by |
| --- | --- |
| `data/natural_msa/` | almost every script (natural MSA and sequence weights) |
| `data/entropy_arrays/` | `entropy_and_energy_landscape.py`, `entropy_decile_correction.py` |
| `data/leave_one_out/` | `leave_one_out_generalization.py`, `generalization_mutability_figure.py` |
| `data/generated_samples/` (4 × 2M sequences) | `sequence_diversity.py`, and `entropy_decile_correction.py` on a cold cache |
| `data/transformer/` | `transformer/gpt_inference.py` |
| `data/cache/` | energies and divergences of the large samples; with this cache, the entropy scripts run without reading `data/generated_samples/` |
| `../models/tested_models/` | `mutability.py`, `generalization_mutability_figure.py`, `mixing_time.py`, `cross_model_scoring.py`, `entropy_decile_correction.py` |

Two inputs come from the repository root instead of being duplicated here, since they
are the released data of the project:

- `../models/tested_models/{bmDCA,eaDCA,edDCA,meDCA}_params.{zip,dat}` — the four final
  models, used by every script that needs model parameters;
- `../data/experiment/*.csv` — in vivo relative-enrichment data, used by
  `cross_model_scoring.py`.

This package used to carry a second copy of those four models under their development
names (`Parameters_edDCA_density=0.125_high_entropy.dat` = meDCA,
`Parameters_conv_dec_new.dat` = edDCA, `Parameters_EAALG0.8_new.dat` = eaDCA,
`Parameters_conv_dense_new.dat` = bmDCA). The copies were byte-identical to the released
files, so they were removed and `paths.MODEL_FILES` points at the released ones.

## Running

```bash
conda activate DCA_env
cd scripts
python entropy_and_energy_landscape.py     # Fig. 2, Fig. 5
python cross_model_scoring.py              # Fig. 3, Fig. S2
python generalization_mutability_figure.py # Fig. 4
```

Requirements: `adabmDCA`, PyTorch, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn,
SciPy, tqdm. The plots use `text.usetex=True`, so a working LaTeX installation
(`latex`, `dvipng`) is required. A CUDA GPU is used when available; everything also runs
on CPU, more slowly.

## Differences from the notebooks

The code is unchanged except for what a plain interpreter and the current library
versions require:

1. markdown cells became comment blocks, code cells are verbatim and separated by `# %%`;
2. a bare expression at the end of a cell (implicit Jupyter display) became `print(...)`;
3. `display()` in `cross_model_scoring.py` falls back to `print()`;
4. input and output paths come from `paths.py` instead of being hard-coded;
5. `entropy_and_energy_landscape.py` plots the decile-corrected bin entropies in Fig. 2;
6. `generalization_mutability_figure.py` caches the arrays of its two panels in
   `data/cache/fig04_panel_data.npz`, so a change of style redraws Fig. 4 in seconds
   instead of recomputing the leave-one-out energies and the mutability;
7. in `leave_one_out_generalization.py` and `generalization_mutability_figure.py` the
   excluded sequence is passed to `compute_energy` as a batch of one, `(1, L, q)`.
   The notebooks flattened it to `(1, L*q)`, which the installed version of `adabmDCA`
   rejects — as written, those two notebooks no longer run in the current environment.
