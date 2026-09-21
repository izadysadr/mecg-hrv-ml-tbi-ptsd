#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Title: HRV Poincaré and Nonlinear Metrics Extraction from M-ECG RR Intervals
===============================================================================

Author:      AQIL IZADYSADR

Description:
------------
This production-ready script computes time-domain, geometric, and nonlinear 
Heart Rate Variability (HRV) metrics from preprocessed RR interval data. 

1. **Poincaré Analysis**:
   - SD1: short-term variability
   - SD2: long-term variability
   - SD1/SD2 ratio

2. **Nonlinear HRV Analysis**:
   - Approximate Entropy (ApEn)
   - Sample Entropy (SampEn)
   - Detrended Fluctuation Analysis alpha1 (DFA α1)

Features:
---------
- Automatically detects RR intervals in seconds or milliseconds and converts.
- Memory-optimized for large cohort scaling.
- Unified logging for progress tracking and error handling.
- Saves all computed metrics into a single structured CSV file.

Inputs:
-------
- Directory containing RR interval pickle files named as:
  `*_rr_intervals_corrected.np.pkl`

Outputs:
--------
- CSV file containing all computed geometric/nonlinear metrics per subject.

Usage:
------
Run via command line specifying the input directory and output CSV path:

    python 03_extract_hrv_nonlinear.py -i /path/to/data -o /path/to/save/non_linear_hrv_metrics.csv

===============================================================================
"""

import argparse
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import antropy as ant
import matplotlib.pyplot as plt
import nolds
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def compute_and_plot_poincare(
    rr_array: np.ndarray,
    plot_standard: bool = False,
    save_path_standard: Optional[str] = None
) -> Dict[str, float]:
    """
    Computes Poincaré plot metrics to quantify short-term and long-term HRV.

    The Poincaré plot represents a scatter of the current RR interval against the 
    subsequent one. Fitting an ellipse to this scatter yields SD1 (minor axis, 
    correlating with rapid, parasympathetic beat-to-beat changes) and SD2 (major axis, 
    reflecting continuous, long-term autonomic regulation).

    Args:
        rr_array (np.ndarray): A 2D array where the second column contains RR intervals.
        plot_standard (bool, optional): If True, generates a diagnostic standard 
            Poincaré scatter plot with the fitted ellipse. Defaults to False.
        save_path_standard (Optional[str], optional): File path to save the generated 
            plot. Required if plot_standard is True and saving is desired. Defaults to None.

    Returns:
        Dict[str, float]: A dictionary containing the computed metrics:
            - SD1: Standard deviation perpendicular to the line of identity.
            - SD2: Standard deviation along the line of identity.
            - SD1/SD2: Ratio of short- to long-term variability.
    """
    # Extract RR intervals from the second column
    rr = rr_array[:, 1]

    # Detect if RR intervals are in seconds (mean < 5s is a safe physiological heuristic)
    # and convert to milliseconds to ensure standard HRV unit scaling.
    if np.mean(rr) < 5:
        rr = rr * 1000
        logger.debug("Detected RR intervals in seconds. Converted to ms for computation.")

    # Create lagged RR pairs for phase space reconstruction
    rr_n = rr[:-1]
    rr_n1 = rr[1:]

    # Compute SD1 and SD2 analytically without needing to rotate the coordinate system.
    # diff captures the beat-to-beat variance (high frequency).
    diff = rr_n1 - rr_n
    SD1 = np.sqrt(np.var(diff) / 2)
    SD2 = np.sqrt(2 * np.var(rr) - (np.var(diff) / 2))
    
    # Calculate the SD1/SD2 ratio used in the manuscript
    ratio = SD1 / SD2 if SD2 != 0 else np.nan

    # ----- Standard Plot -----
    if plot_standard:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter plot of RR(n) vs RR(n+1) showing system trajectory
        axs[0].scatter(rr_n, rr_n1, color='blue', alpha=0.5, s=10)
        axs[0].set_title("Poincaré Plot")
        axs[0].set_xlabel("RR(n) [ms]")
        axs[0].set_ylabel("RR(n+1) [ms]")
        axs[0].axis('equal')

        # Ellipse overlay representing orthogonal SD1 and SD2 deviations
        rr_mean = np.mean(rr)
        ellipse = Ellipse(
            (rr_mean, rr_mean),
            width=2 * SD2 * np.sqrt(2),
            height=2 * SD1 * np.sqrt(2),
            angle=45,
            edgecolor='red',
            facecolor='none',
            linewidth=2
        )
        axs[0].add_patch(ellipse)

        # Line of identity representing theoretical zero variance
        min_rr = min(min(rr_n), min(rr_n1))
        max_rr = max(max(rr_n), max(rr_n1))
        axs[0].plot([min_rr, max_rr], [min_rr, max_rr], 'k--', alpha=0.5)

        # Bar plot summarizing the geometric metrics used in the manuscript
        axs[1].bar(['SD1', 'SD2', 'SD1/SD2'], [SD1, SD2, ratio], color=['C0', 'C1', 'C2'])
        axs[1].set_title("Computed Metrics")
        axs[1].set_ylabel("Value")

        plt.tight_layout()
        if save_path_standard:
            plt.savefig(save_path_standard, dpi=300, bbox_inches='tight')
            logger.info(f"Standard Poincaré plot saved to {save_path_standard}")

        plt.close(fig)

    return {
        'SD1': float(SD1),
        'SD2': float(SD2),
        'SD1/SD2': float(ratio)
    }


def compute_nonlinear_hrv_metrics(
    rr_series: np.ndarray,
    m: int = 2,
    r: Optional[float] = None,
    dfa_scales: Tuple[int, int] = (4, 16),
    plot: bool = False
) -> Dict[str, float]:
    """
    Computes nonlinear complexity and fractal scaling HRV metrics from RR intervals.

    Evaluates the signal's unpredictability via Approximate and Sample Entropy, and 
    its self-similarity via Detrended Fluctuation Analysis (DFA). These metrics are 
    particularly robust for characterizing complex cardiac dynamics derived from 
    unconventional sources like M-ECG.

    Args:
        rr_series (np.ndarray): A 2D array where col 0 is time and col 1 is RR intervals.
        m (int, optional): Embedding dimension for entropy calculations. Defaults to 2.
        r (Optional[float], optional): Tolerance threshold for matching vectors in entropy.
            If None, defaults to 0.2 * standard deviation of the RR series.
        dfa_scales (Tuple[int, int], optional): The min and max box sizes (in beats) for 
            calculating the short-term fractal scaling exponent (alpha1). Defaults to (4, 16).
        plot (bool, optional): If True, renders a diagnostic plot showing the time series 
            and the log-log DFA regression line. Defaults to False.

    Returns:
        Dict[str, float]: A dictionary containing the nonlinear metrics:
            - ApEn: Approximate Entropy (measure of regularity/complexity).
            - SampEn: Sample Entropy (less biased measure of regularity).
            - DFA_alpha1: Short-term fractal scaling exponent.

    Raises:
        ValueError: If the RR series length is insufficient (< 100 beats) for 
            stable nonlinear dynamic estimation.
    """
    # Ensure RR intervals are scaled to milliseconds for entropy stability
    rr_ms = rr_series[:, 1] * 1000.0

    if len(rr_ms) < 100:
        raise ValueError(
            "RR series too short for reliable nonlinear HRV metrics (minimum ~100 points recommended)."
        )

    # Apply the widely accepted Pincus standard for tolerance (0.2 * SDNN)
    if r is None:
        r = 0.2 * np.std(rr_ms)
        logger.debug(f"Tolerance r not provided. Using r = 0.2 * std(RR) = {r:.2f} ms")

    # Entropy computations measuring the logarithmic likelihood that runs of 
    # patterns that are close remain close on the next incremental comparisons.
    apen = ant.app_entropy(rr_ms, order=m, metric='chebyshev', tolerance=r)
    sampen = ant.sample_entropy(rr_ms, order=m, tolerance=r)

    # DFA alpha1 computation to assess short-term fractal correlation properties
    nvals = list(range(dfa_scales[0], dfa_scales[1] + 1))
    dfa_alpha1 = nolds.dfa(rr_ms, nvals=nvals)

    metrics = {
        "ApEn": float(apen),
        "SampEn": float(sampen),
        "DFA_alpha1": float(dfa_alpha1)
    }

    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # RR Time Series
        axs[0].plot(rr_series[:, 0], rr_ms, lw=1)
        axs[0].set_title(f"RR Time Series (ApEn = {apen:.3f})")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("RR Interval (ms)")

        # Manual DFA computation explicitly reconstructed for visualizing the log-log fit
        integrated = np.cumsum(rr_ms - np.mean(rr_ms))
        flucts = []
        valid_scales = []
        for scale in nvals:
            segments = len(integrated) // scale
            if segments < 2:
                continue
            rms = []
            for i in range(segments):
                segment = integrated[i*scale:(i+1)*scale]
                x = np.arange(scale)
                # Linear detrending within each local bounding box
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                detrended = segment - trend
                rms.append(np.sqrt(np.mean(detrended**2)))
            flucts.append(np.mean(rms))
            valid_scales.append(scale)

        log_scales = np.log10(valid_scales)
        log_flucts = np.log10(flucts)
        axs[1].plot(log_scales, log_flucts, 'o-', label="Manual DFA fluctuation")
        
        # Plot the regression line defining the alpha1 scaling exponent
        slope, intercept = np.polyfit(log_scales, log_flucts, 1)
        axs[1].plot(log_scales, slope * log_scales + intercept, 'r--', label=f"Slope = {slope:.3f}")
        axs[1].set_title(f"DFA α1 (Scales {dfa_scales[0]}–{dfa_scales[1]} beats)")
        axs[1].set_xlabel("log(window size)")
        axs[1].set_ylabel("log(fluctuation)")
        axs[1].legend()

        # Display Entropy metadata cleanly
        axs[2].text(
            0.1, 0.5,
            f"Sample Entropy (SampEn): {sampen:.3f}\nTolerance (r): {r:.2f} ms",
            fontsize=14,
            verticalalignment='center'
        )
        axs[2].axis('off')

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    return metrics


def process_nonlinear_hrv_data(input_dir: str, output_csv: str) -> None:
    """
    Traverses the input cohort directory, computes geometric and nonlinear metrics 
    for each corrected M-ECG RR interval file, and aggregates them into a CSV.

    Args:
        input_dir (str): Root directory path containing the artifact-corrected 
            RR interval pickle files.
        output_csv (str): File path defining where the compiled CSV results will be saved.
    """
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        logger.error(f"Input directory does not exist or is not a directory: {input_dir}")
        return

    pattern = '*_rr_intervals_corrected.np.pkl'
    matching_files = sorted(list(input_path.rglob(pattern)))

    if not matching_files:
        logger.warning(f"No files matching '{pattern}' found in {input_dir}")
        return

    logger.info(f"Found {len(matching_files)} RR interval files matching pattern.")

    # List to accumulate dictionary rows to prevent DataFrame append fragmentation overhead
    results_list: List[Dict[str, Any]] = []
    
    # Regex to strictly extract the Subject ID from the M-ECG pipeline naming convention
    subject_regex = re.compile(r'([A-Za-z0-9]+)_rr_intervals_corrected\.np\.pkl$')

    for counter, file_path in enumerate(matching_files, start=1):
        # Extract SUBJECT_ID
        match = subject_regex.search(file_path.name)
        if match:
            subject_id = match.group(1)
        else:
            logger.warning(f"Could not parse SUBJECT_ID from {file_path.name}. Skipping.")
            continue

        try:
            with open(file_path, 'rb') as f:
                rr_intervals_corrected = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load data for {subject_id}: {e}")
            continue

        # ----------------------- Compute Metrics -----------------------
        poincare_metrics = compute_and_plot_poincare(
            rr_array=rr_intervals_corrected,
            plot_standard=False
        )
        
        nonlinear_metrics = compute_nonlinear_hrv_metrics(
            rr_series=rr_intervals_corrected, 
            plot=False
        )

        logger.info(f"[{counter}/{len(matching_files)}] Processed: {subject_id} | SD1/SD2: {poincare_metrics['SD1/SD2']:.3f} | SampEn: {nonlinear_metrics['SampEn']:.3f}")

        # ----------------------- Append to Results -----------------------
        results_list.append({
            'SUBJECT_ID': subject_id,
            "SD1 (ms)": poincare_metrics['SD1'],
            "SD2 (ms)": poincare_metrics['SD2'],
            "SD1/SD2 ratio": poincare_metrics['SD1/SD2'],
            "approximate entropy (ApEn)": nonlinear_metrics['ApEn'],
            "sample entropy (SampEn)": nonlinear_metrics['SampEn'],
            "DFA α1": nonlinear_metrics['DFA_alpha1']
        })

    # Construct DataFrame once at the end and push directly to disk
    if results_list:
        df = pd.DataFrame(results_list)
        
        # Ensure output directory tree exists
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False, header=True)
        logger.info(f"Successfully saved HRV metrics for {len(df)} subjects to: {output_path}")
    else:
        logger.warning("No valid data was processed. CSV not created.")


def main() -> None:
    """
    Parses command-line arguments to execute the nonlinear/geometric extraction pipeline.
    """
    parser = argparse.ArgumentParser(description="Extract Geometric and Nonlinear HRV Metrics from M-ECG RR Intervals.")
    parser.add_argument(
        '-i', '--input_dir', 
        type=str, 
        required=True, 
        help="Root directory containing the corrected RR interval pickle files."
    )
    parser.add_argument(
        '-o', '--output_csv', 
        type=str, 
        required=True, 
        help="File path where the compiled CSV results will be saved."
    )
    
    args = parser.parse_args()
    process_nonlinear_hrv_data(args.input_dir, args.output_csv)


if __name__ == "__main__":
    main()