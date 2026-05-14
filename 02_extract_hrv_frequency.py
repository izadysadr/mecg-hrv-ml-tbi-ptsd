#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Title: HRV Frequency-Domain Metrics Extraction from M-ECG-Derived RR Intervals
===============================================================================

Author:      AQIL IZADYSADR
Created:     June 27, 2025

Description:
------------
This production-ready script scans a directory of corrected RR interval files,
interpolates the RR intervals, computes power spectral density (PSD), extracts
standard HRV frequency-domain metrics (VLF, LF, HF bands), calculates LF/HF ratios,
identifies LF and HF peak frequencies, logs results, and saves the compiled metrics
to a CSV file.

Key Features:
-------------
1. Interpolates RR intervals using cubic splines to ensure evenly spaced time points.
2. Computes PSD via Welch’s method (0–0.4 Hz).
3. Extracts frequency-domain HRV metrics for each subject:
   - Absolute power (ms²) for VLF, LF, HF
   - Relative power (%) for VLF, LF, HF
   - LF/HF ratio
   - LF and HF peak frequencies (Hz)
   - Total power across all bands
4. Optional visualization of PSD with highlighted HRV bands and peak markers.
5. Handles multiple subjects automatically via directory traversal.
6. Memory-optimized and structured for large cohort scaling.

Inputs:
-------
- Directory containing pickle files of corrected RR intervals
  (pattern: '*_rr_intervals_corrected.np.pkl')
- Each pickle file should be a 2D NumPy array:
    col 0: time stamps (seconds)
    col 1: RR interval values (seconds)

Outputs:
--------
- CSV file containing frequency-domain HRV metrics for all subjects.

Usage:
------
Run via command line specifying the input directory and output CSV path:

    python m-ecg_frequency_domain_hrv_extraction.py -i /path/to/data -o /path/to/save/frequency_domain_hrv_metrics.csv

===============================================================================
"""

import argparse
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def interpolate_rr_and_compute_psd(
    rr_corrected: np.ndarray, 
    num_points: int = 850
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Interpolates unevenly sampled RR intervals and computes the Power Spectral Density (PSD).

    To perform frequency-domain analysis (like Welch's method), the non-equidistant 
    RR time series must be resampled to form a continuous, uniformly sampled signal. 
    This function utilizes cubic spline interpolation followed by PSD estimation.

    Args:
        rr_corrected (np.ndarray): A 2D array of shape (N, 2) where the first column 
            contains time stamps (seconds) and the second contains RR intervals (seconds).
        num_points (int, optional): The number of evenly spaced points for the interpolated 
            time vector. Defaults to 850.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]: A 5-tuple containing:
            - rr_interpolated: The interpolated RR interval array in milliseconds.
            - new_time: The newly generated evenly spaced time vector.
            - sampling_rate: The effective sampling frequency (Hz) of the interpolated signal.
            - psd: The estimated power spectral density values.
            - freqs: The array of sample frequencies corresponding to the PSD values.
    """
    # Extract time and RR values from input array
    rr_time = rr_corrected[:, 0]              # time in seconds
    rr_values_sec = rr_corrected[:, 1]        # RR intervals in seconds

    # Convert RR intervals to milliseconds for standard HRV analysis interpretation
    rr_values_ms = rr_values_sec * 1000

    # Create cubic spline interpolator to handle non-linear physiological variations
    interpolator = CubicSpline(rr_time, rr_values_ms)

    # Generate new evenly spaced time vector
    new_time = np.linspace(rr_time[0], rr_time[-1], num=num_points)

    # Compute interpolated RR intervals
    rr_interpolated = interpolator(new_time)

    # Estimate effective sampling rate of the interpolated signal required for Welch's method
    duration = new_time[-1] - new_time[0]
    sampling_rate = (len(new_time) - 1) / duration

    # Compute PSD using Welch's method, bounded exactly to the standard physiological
    # HRV frequency range (0–0.4 Hz) to avoid high-frequency noise integration.
    psd, freqs = mne.time_frequency.psd_array_welch(
        rr_interpolated,
        sfreq=sampling_rate,
        fmin=0,
        fmax=0.4,
        verbose=False
    )
    return rr_interpolated, new_time, sampling_rate, psd, freqs
    

def compute_hrv_band_powers_and_plot(
    freqs: np.ndarray, 
    psd: np.ndarray, 
    show_fig: bool = False, 
    figsize: Tuple[int, int] = (10, 6), 
    dpi: int = 300
) -> Dict[str, Any]:
    """
    Extracts standard HRV frequency-domain metrics and optionally renders the PSD plot.

    Integrates the PSD over standard Very Low Frequency (VLF), Low Frequency (LF), 
    and High Frequency (HF) bands using the trapezoidal rule to compute absolute power. 
    It also calculates relative powers, the LF/HF autonomic balance ratio, and identifies
    peak frequencies within the LF and HF bands.

    Args:
        freqs (np.ndarray): The frequency array corresponding to the PSD values.
        psd (np.ndarray): The power spectral density array.
        show_fig (bool, optional): Whether to generate and return a matplotlib figure 
            highlighting the HRV bands and peaks. Defaults to False.
        figsize (Tuple[int, int], optional): Dimensions of the figure. Defaults to (10, 6).
        dpi (int, optional): Resolution of the figure. Defaults to 300.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - absolute_power (Dict[str, float]): Absolute power (ms²) for VLF, LF, HF.
            - relative_power (Dict[str, float]): Relative power (%) for VLF, LF, HF.
            - lf_hf_ratio (float): The calculated LF/HF ratio.
            - total_power (float): Sum of absolute powers across all bands.
            - lf_peak_freq (float): Frequency (Hz) of maximum power in the LF band.
            - hf_peak_freq (float): Frequency (Hz) of maximum power in the HF band.
            - fig (Optional[plt.Figure]): The generated matplotlib figure, or None.
    """
    # Define standard physiological HRV frequency bands (Hz)
    bands = {
        "VLF": (0.003, 0.04),
        "LF": (0.04, 0.15),
        "HF": (0.15, 0.4)
    }

    # Compute absolute power for each band using trapezoidal numerical integration
    powers_abs = {}
    for band, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        power = trapezoid(psd[mask], freqs[mask])
        powers_abs[band] = power

    # Total power across all standard bands
    total_power = sum(powers_abs.values())

    # Compute relative power (%) for each band, handling division by zero
    powers_rel = {
        band: (power / total_power) * 100 if total_power != 0 else np.nan
        for band, power in powers_abs.items()
    }

    # Compute LF/HF ratio as a traditional index of sympathovagal balance
    lf_hf_ratio = powers_abs["LF"] / powers_abs["HF"] if powers_abs["HF"] != 0 else np.nan

    # Identify LF peak frequency
    lf_mask = (freqs >= bands["LF"][0]) & (freqs < bands["LF"][1])
    if np.any(lf_mask):
        lf_peak_idx = np.argmax(psd[lf_mask])
        lf_peak = float(freqs[lf_mask][lf_peak_idx])
        lf_peak_val = float(psd[lf_mask][lf_peak_idx])
    else:
        lf_peak = np.nan
        lf_peak_val = np.nan

    # Identify HF peak frequency
    hf_mask = (freqs >= bands["HF"][0]) & (freqs < bands["HF"][1])
    if np.any(hf_mask):
        hf_peak_idx = np.argmax(psd[hf_mask])
        hf_peak = float(freqs[hf_mask][hf_peak_idx])
        hf_peak_val = float(psd[hf_mask][hf_peak_idx])
    else:
        hf_peak = np.nan
        hf_peak_val = np.nan

    # Plot PSD with highlighted bands and LF/HF peaks if requested
    if show_fig:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(freqs, psd, label='PSD (linear)', color='black')

        colors = {"VLF": 'gray', "LF": 'yellow', "HF": 'orange'}
        for band, (fmin, fmax) in bands.items():
            ax.axvspan(fmin, fmax, color=colors[band], alpha=0.5, label=f"{band} Band")

        if not np.isnan(lf_peak):
            ax.axvline(lf_peak, color='blue', linestyle='--', label=f'LF Peak: {lf_peak:.3f} Hz')
            ax.annotate(f'{lf_peak:.3f} Hz', xy=(lf_peak, lf_peak_val), xytext=(lf_peak, lf_peak_val*1.1),
                        arrowprops=dict(arrowstyle='->', color='blue'), color='blue', fontsize=9, ha='center')

        if not np.isnan(hf_peak):
            ax.axvline(hf_peak, color='red', linestyle='--', label=f'HF Peak: {hf_peak:.3f} Hz')
            ax.annotate(f'{hf_peak:.3f} Hz', xy=(hf_peak, hf_peak_val), xytext=(hf_peak, hf_peak_val*1.1),
                        arrowprops=dict(arrowstyle='->', color='red'), color='red', fontsize=9, ha='center')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (ms²/Hz)')
        ax.set_title('Power Spectral Density with HRV Bands and Peak Frequencies')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
    else:
        fig = None

    return {
        "absolute_power": powers_abs,
        "relative_power": powers_rel,
        "lf_hf_ratio": lf_hf_ratio,
        "total_power": total_power,
        "lf_peak_freq": lf_peak,
        "hf_peak_freq": hf_peak,
        "fig": fig
    }


def process_frequency_hrv_data(input_dir: str, output_csv: str) -> None:
    """
    Traverses the input cohort directory, extracts M-ECG RR interval metrics, 
    and compiles the frequency-domain feature set into a CSV.

    Args:
        input_dir (str): Root directory path containing the corrected RR interval pickle files.
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

    # List to accumulate dictionary rows (highly optimized for memory over large cohorts)
    results_list: List[Dict[str, Any]] = []
    
    # Regex to extract Subject ID safely from the established naming convention
    subject_regex = re.compile(r'(.+)_rr_intervals_corrected\.np\.pkl$')

    for counter, file_path in enumerate(matching_files, start=1):
        # Extract SUBJECT_ID
        match = subject_regex.search(file_path.name)
        if match:
            subject_id = match.group(1)
        else:
            logger.warning(f"Skipping {file_path.name}: SUBJECT_ID not found.")
            continue

        try:
            with open(file_path, 'rb') as f:
                rr_intervals_corrected = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load data for {subject_id}: {e}")
            continue

        # -------------------------
        # --- Run HRV Pipeline ---
        # -------------------------
        
        # Interpolate RR intervals and compute PSD
        rr_interpolated, new_time, sampling_rate, psd, freqs = interpolate_rr_and_compute_psd(rr_intervals_corrected)
        
        # Compute HRV frequency-domain metrics
        hrv_results = compute_hrv_band_powers_and_plot(freqs, psd)

        # Log-transform absolute power values to address physiological skewness 
        # and ensure normal distribution for downstream parametric statistical testing.
        abs_power = hrv_results["absolute_power"]
        abs_power_ln = {band: np.log(power + 1e-10) for band, power in abs_power.items()}

        logger.info(f"[{counter}/{len(matching_files)}] Processed: {subject_id} | LF/HF Ratio: {hrv_results['lf_hf_ratio']:.3f}")

        # -------------------------
        # --- Append to Results ---
        # -------------------------
        results_list.append({
            'SUBJECT_ID': subject_id,
            'VLF - absolute power (ms²)': abs_power['VLF'],
            'LF - absolute power (ms²)': abs_power['LF'],
            'HF - absolute power (ms²)': abs_power['HF'],
            'VLF - absolute power ln(ms²)': abs_power_ln['VLF'],
            'LF - absolute power ln(ms²)': abs_power_ln['LF'],
            'HF - absolute power ln(ms²)': abs_power_ln['HF'],
            'VLF - relative power (%)': hrv_results["relative_power"]['VLF'],
            'LF - relative power (%)': hrv_results["relative_power"]['LF'],
            'HF - relative power (%)': hrv_results["relative_power"]['HF'],
            'LF/HF Ratio': hrv_results["lf_hf_ratio"],
            'Total Power (ms²)': hrv_results["total_power"],
            'LF Peak Frequency (Hz)': hrv_results["lf_peak_freq"],
            'HF Peak Frequency (Hz)': hrv_results["hf_peak_freq"]
        })

    # Construct DataFrame once at the end to prevent memory fragmentation
    if results_list:
        df = pd.DataFrame(results_list)
        
        # Ensure output directory exists
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False, header=True)
        logger.info(f"Successfully saved HRV metrics for {len(df)} subjects to: {output_path}")
    else:
        logger.warning("No valid data was processed. CSV not created.")


def main() -> None:
    """
    Parses command-line arguments to execute the M-ECG frequency-domain extraction pipeline.
    """
    parser = argparse.ArgumentParser(description="Extract Frequency-Domain HRV Metrics from M-ECG RR Intervals.")
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
    process_frequency_hrv_data(args.input_dir, args.output_csv)


if __name__ == "__main__":
    main()