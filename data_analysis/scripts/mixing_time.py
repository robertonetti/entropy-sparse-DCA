#!/usr/bin/env python3
"""
Mixing time of the Monte Carlo sampling for the four DCA models.

Figures
-------
Fig. S7   figures/supplementary/figS7_mixing_time.png

Inputs
------
data/natural_msa/   natural MSA and sequence weights
models/             the four DCA models under their development names

Outputs
-------
results/mixing_time_analysis.log   stdout and stderr of the run, which are redirected
                                   to the log file instead of the terminal

Runtime
-------
Long: the sampling is run until the chains decorrelate, for four models. GPU recommended.
"""

import sys
import os
from pathlib import Path

# Redirect stdout and stderr to log file
from paths import FIGURES_SUPPLEMENTARY, MODELS_DIR, NATURAL_MSA, NATURAL_WEIGHTS, RESULTS_DIR

log_file = open(RESULTS_DIR / 'mixing_time_analysis.log', 'w')
sys.stdout = log_file
sys.stderr = log_file

try:
    # ==============================================================================
    # IMPORTS AND SETUP
    # ==============================================================================
    print("=" * 80)
    print("Starting mixing time analysis")
    print("=" * 80)
    
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving figures
    import matplotlib.pyplot as plt
    
    from adabmDCA.io import load_params, load_chains
    from adabmDCA.resampling import compute_mixing_time
    from adabmDCA.utils import resample_sequences
    from adabmDCA.fasta import get_tokens, import_from_fasta, compute_weights
    from adabmDCA.stats import get_freq_single_point, get_freq_two_points
    from adabmDCA.functional import one_hot
    from adabmDCA.utils import init_parameters, init_chains, get_device, get_dtype, get_mask_save
    from adabmDCA.sampling import get_sampler
    from adabmDCA.training import train_graph, train_edDCA, train_eaDCA
    from adabmDCA.io import save_params, save_chains
    from adabmDCA.graph import compute_density
    from adabmDCA.statmech import compute_energy
    
    print("All libraries imported successfully!")
    
    # ==============================================================================
    # DEVICE, PATHS, AND TOKENS CONFIGURATION
    # ==============================================================================
    
    # Configure computational devices
    device = get_device("cpu")          # CPU device for general operations
    device2 = get_device("cuda")        # GPU device for intensive computation
    dtype = get_dtype("float32")         # Data type for tensor operations
    
    # Data paths
    data_path = str(NATURAL_MSA)
    weights_path = NATURAL_WEIGHTS
    
    # Model paths - different DCA model variants trained on the same MSA
    HAmodel_path = str(MODELS_DIR / "Parameters_edDCA_density=0.125_high_entropy.dat")
    decmodel_path = str(MODELS_DIR / "Parameters_conv_dec_new.dat")
    eamodel_path = str(MODELS_DIR / "Parameters_EAALG0.8_new.dat")
    densemodel_path = str(MODELS_DIR / "Parameters_conv_dense_new.dat")
    
    # Define amino acid tokens (20 amino acids + gap character)
    tokens = get_tokens("ACDEFGHIKLMNPQRSTVWY-")
    
    print("Configuration loaded successfully!")
    print(f"Using devices: {device} and {device2}")
    print(f"Number of tokens: {len(tokens)}")
    
    # Create output directory

    
    # ==============================================================================
    # LOAD DATA
    # ==============================================================================
    print("\n" + "=" * 80)
    print("Loading data")
    print("=" * 80)
    
    headers, sequences = import_from_fasta(data_path, tokens=tokens, filter_sequences=True, remove_duplicates=True)
    M, L, q = len(sequences), len(sequences[0]), len(tokens)
    print(f"Number of sequences in the MSA: M={M}")
    print(f"Length of the MSA: L={L}")
    print(f"Number of Potts states: q={q}")
    
    msa_enc = torch.tensor(sequences, device=device, dtype=torch.int32)
    msa_oh = one_hot(msa_enc, num_classes=q).to(dtype)
    
    weights = torch.tensor(np.loadtxt(weights_path), device=device, dtype=dtype)
    
    # ==============================================================================
    # LOAD MODEL PARAMETERS
    # ==============================================================================
    print("\n" + "=" * 80)
    print("Loading model parameters")
    print("=" * 80)
    
    params_HA = load_params(HAmodel_path, tokens=tokens, device=device, dtype=dtype)
    print(f"Loaded: {HAmodel_path}")
    
    params_dec = load_params(decmodel_path, tokens=tokens, device=device, dtype=dtype)
    print(f"Loaded: {decmodel_path}")
    
    params_ea = load_params(eamodel_path, tokens=tokens, device=device, dtype=dtype)
    print(f"Loaded: {eamodel_path}")
    
    params_dense = load_params(densemodel_path, tokens=tokens, device=device, dtype=dtype)
    print(f"Loaded: {densemodel_path}")
    
    # ==============================================================================
    # SAMPLING CONFIGURATION
    # ==============================================================================
    get_mixing_time = True
    max_sweeps = 10_000_000
    num_gen = 10_000
    sampler_alg = "gibbs"
    sampler = get_sampler(sampler_alg)
    num_mixing_times = 2
    num_sweeps = None
    beta = 1.0
    
    print(f"\nSampling configuration:")
    print(f"  Sampler: {sampler_alg}")
    print(f"  Max sweeps: {max_sweeps}")
    print(f"  Number of generations: {num_gen}")
    print(f"  Beta: {beta}")
    
    # ==============================================================================
    # RESAMPLE SEQUENCES
    # ==============================================================================
    print("\n" + "=" * 80)
    print("Resampling sequences")
    print("=" * 80)
    
    msa_mix = resample_sequences(msa_oh, weights=weights, nextract=M)
    print(f"Resampled {M} sequences")
    
    # ==============================================================================
    # COMPUTE MIXING TIME FOR ALL MODELS
    # ==============================================================================
    print("\n" + "=" * 80)
    print("Computing mixing time for all models")
    print("=" * 80)
    
    # Define models to analyze
    models = {
        'meDCA': params_HA,
        'edDCA': params_dec,
        'eaDCA': params_ea,
        'bmDCA': params_dense
    }
    
    # Store results for all models
    all_results = {}
    mixing_times = {}
    
    # Compute mixing time for each model
    for model_name, params in models.items():
        print(f"\nComputing mixing time for {model_name}...")
        results = compute_mixing_time(
            sampler=sampler,
            params=params,
            data=msa_mix,
            n_max_sweeps=max_sweeps,
            beta=1.0,
        )
        all_results[model_name] = results
        mixing_times[model_name] = results['t_half'][-1]
        print(f"  Mixing time: {mixing_times[model_name]}")
    
    # ==============================================================================
    # CREATE COMPARISON PLOT
    # ==============================================================================
    print("\n" + "=" * 80)
    print("Creating comparison plot")
    print("=" * 80)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=96)
    axes = axes.flatten()
    
    # Plot each model in a subplot
    for idx, (model_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        
        # Plot SeqID(t)
        ax.plot(results["t_half"], results["seqid_t"], label=r"SeqID$(t)$", color="#002CFF")
        ax.fill_between(results["t_half"],
                        np.array(results["seqid_t"]) - np.array(results["std_seqid_t"]),
                        np.array(results["seqid_t"]) + np.array(results["std_seqid_t"]),
                        color="#002CFF", alpha=0.2)
        
        # Plot SeqID(t, t/2)
        ax.plot(results["t_half"], results["seqid_t_t_half"], label=r"SeqID$(t, t/2)$", color="#67BAA6")
        ax.fill_between(results["t_half"],
                        np.array(results["seqid_t_t_half"]) - np.array(results["std_seqid_t_t_half"]),
                        np.array(results["seqid_t_t_half"]) + np.array(results["std_seqid_t_t_half"]),
                        color="#67BAA6", alpha=0.2)
        
        # Labels and legend
        ax.set_xlabel(r"$t/2$ (Sweeps)")
        ax.set_ylabel("Sequence Identity")
        ax.legend(loc='upper right')
        ax.set_title(f"{model_name}")
        
        # Add annotation for mixing time
        ax.annotate(r"$t^{\mathrm{mix}}=$" + f"{results['t_half'][-1]}", 
                    xy=(0.96, 0.7), xycoords='axes fraction', fontsize=12,
                    verticalalignment='top', horizontalalignment='right', 
                    bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    output_file = FIGURES_SUPPLEMENTARY / "figS7_mixing_time.png"
    plt.savefig(output_file)
    print(f"Figure saved to: {output_file}")
    
    # ==============================================================================
    # SUMMARY
    # ==============================================================================
    print("\n" + "=" * 80)
    print("MIXING TIME SUMMARY")
    print("=" * 80)
    for model_name, mixing_time in mixing_times.items():
        print(f"  {model_name}: {mixing_time}")
    
    print("\n" + "=" * 80)
    print("Analysis completed successfully!")
    print("=" * 80)

except Exception as e:
    print(f"\nERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    # Close log file
    log_file.close()
