#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Title: HRV Time-Domain Metrics Extraction from M-ECG RR Intervals
===============================================================================

Author:      AQIL IZADYSADR
Created:     June 20, 2025

Description:
------------
This production-ready script processes corrected RR interval data from multiple 
subjects to compute standard time-domain heart rate variability (HRV) metrics. 
The results are consolidated into a CSV file for downstream analyses.

Key Features:
-------------
1. Computes time-domain HRV metrics for each subject:
   - Mean RR interval (ms)
   - Mean heart rate (bpm)
   - SDNN (ms)
   - RMSSD (ms)
   - SDSD (ms)
   - NN50 count
   - pNN50 (%)
2. Handles multiple subjects automatically via directory traversal.
3. Memory-optimized for large cohort scaling.
4. Outputs results into a structured CSV file for further analysis.

Inputs:
-------
- Directory containing pickle files of corrected RR intervals
  (pattern: '*_rr_intervals_corrected.np.pkl')
- Each pickle file should be a 2D NumPy array:
    col 0: cumulative time (seconds)
    col 1: RR intervals (seconds)

Outputs:
--------
- CSV file containing time-domain HRV metrics for all subjects.

Usage:
------
Run via command line specifying the input directory and output CSV path:

    python m-ecg_time_domain_hrv_extraction.py -i /path/to/data -o /path/to/save/time_domain_hrv_metrics.csv

===============================================================================
"""

import argparse
import logging
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def compute_time_domain_metrics_ms(rr_array: np.ndarray) -> tuple:
    """
    Compute standard time-domain heart rate variability (HRV) metrics 
    from an RR interval array.

    Parameters
    ----------
    rr_array : np.ndarray, shape (n, 2)
        2D NumPy array containing RR interval data.
        - Column 0: cumulative time (seconds) [not used in calculations]
        - Column 1: RR intervals (seconds)

    Returns
    -------
    mean_rr_ms : float
        Mean RR interval in milliseconds.
    mean_hr_bpm : float
        Mean heart rate in beats per minute (bpm), computed as 60,000 / mean RR.
    sdnn_ms : float
        Standard deviation of RR intervals (SDNN) in milliseconds.
    rmssd_ms : float
        Root mean square of successive differences (RMSSD) in milliseconds.
    sdsd_ms : float
        Standard deviation of successive differences (SDSD) in milliseconds.
    nn50_count : int
        Number of successive RR interval differences greater than 50 ms.
    pnn50_percent : float
        Percentage of NN50 counts relative to total number of successive differences.
    """
    # Extract RR intervals (in seconds) from column 1
    rr_seconds = rr_array[:, 1]

    # Convert RR intervals to milliseconds for standard HRV metrics
    rr_ms = rr_seconds * 1000

    # Mean RR interval in ms
    mean_rr = np.mean(rr_ms)  

    # Mean heart rate in beats per minute (bpm)
    mean_hr = 60000 / mean_rr  

    # Standard deviation of RR intervals (ms) — reflects overall HRV
    sdnn = np.std(rr_ms, ddof=1)  

    # Successive differences between RR intervals
    diff_rr = np.diff(rr_ms)

    # Root mean square of successive differences (ms) — reflects short-term HRV
    rmssd = np.sqrt(np.mean(diff_rr ** 2))  

    # Standard deviation of successive differences (ms)
    sdsd = np.std(diff_rr, ddof=1)  

    # Count of successive differences greater than 50 ms
    nn50 = np.sum(np.abs(diff_rr) > 50)

    # Percentage of NN50 relative to total number of successive differences
    pnn50 = (nn50 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0.0

    return mean_rr, mean_hr, sdnn, rmssd, sdsd, nn50, pnn50


def process_hrv_data(input_dir: str, output_csv: str) -> None:
    """
    Traverses the input directory for RR interval files, computes metrics,
    and saves the compiled results to a CSV.
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

    logger.info(f"Found {len(matching_files)} files to process.")

    # List to accumulate dictionary rows (highly optimized for memory)
    results_list = []
    
    # Regex to extract Subject ID safely
    subject_regex = re.compile(r'(.+)_rr_intervals_corrected')

    for counter, file_path in enumerate(matching_files, start=1):
        # Extract SUBJECT_ID
        match = subject_regex.search(file_path.name)
        if match:
            subject_id = match.group(1)
        else:
            logger.warning(f"Could not extract SUBJECT_ID from {file_path.name}. Skipping.")
            continue

        try:
            with open(file_path, 'rb') as f:
                rr_corrected = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load data for {subject_id}: {e}")
            continue

        # Compute Metrics
        mean_rr, mean_hr, sdnn, rmssd, sdsd, nn50, pnn50 = compute_time_domain_metrics_ms(rr_corrected)

        logger.info(f"[{counter}/{len(matching_files)}] Processed: {subject_id}")
        logger.debug(f"{subject_id} Metrics - mean_rr: {mean_rr:.2f}, rmssd: {rmssd:.2f}, sdnn: {sdnn:.2f}")

        # Store results
        results_list.append({
            'SUBJECT_ID': subject_id,
            'mean_rr_ms - resting state': mean_rr,
            'mean_hr_bpm - resting state': mean_hr,
            'sdnn_ms - resting state': sdnn,
            'rmssd_ms - resting state': rmssd,
            'sdsd_ms - resting state': sdsd,
            'nn50_count - resting state': nn50,
            'pnn50_percent - resting state': pnn50
        })

    # Construct DataFrame once at the end
    if results_list:
        df = pd.DataFrame(results_list)
        
        # Ensure output directory exists
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False, header=True)
        logger.info(f"Successfully saved HRV metrics for {len(df)} subjects to: {output_path}")
    else:
        logger.warning("No valid data was processed. CSV not created.")


def main():
    parser = argparse.ArgumentParser(description="Extract Time-Domain HRV Metrics from M-ECG RR Intervals.")
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
    process_hrv_data(args.input_dir, args.output_csv)


if __name__ == "__main__":
    main()