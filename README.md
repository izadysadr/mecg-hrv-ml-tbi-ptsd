# Multivariate Machine Learning Analysis of MEG‑Derived Heart Rate Variability in TBI With and Without Comorbid PTSD

## Overview

This repository contains the **working analysis code** used in the study:

**"Multivariate Machine Learning Analysis of MEG‑Derived Heart Rate Variability in TBI Individuals With and Without Comorbid PTSD"**

The project investigates whether multivariate machine‑learning models applied to **heart rate variability (HRV)** features extracted from **MEG-derived ECG (M‑ECG)** signals can differentiate veterans with **TBI alone** from those with **comorbid PTSD (TBI+PTSD)**.

Rather than relying on isolated univariate HRV comparisons, this pipeline extracts **time-domain, frequency-domain, geometric, and nonlinear HRV metrics** and evaluates their **joint discriminative structure** using **nested cross-validated machine learning**, feature selection, and model interpretability techniques.

---

## Scientific Motivation

TBI and PTSD frequently co-occur and share overlapping symptomatology and autonomic dysfunction. Traditional HRV analyses often fail to detect robust group differences after correction for multiple comparisons. This project demonstrates that:

- **Multivariate HRV patterns** carry diagnostically relevant information even when
- **Univariate HRV effects are weak or absent**, and
- **Machine learning models** can uncover distributed autonomic signatures linked to comorbidity.

Manuscript Link:

DOI: 

PMID:

The code here reflects the **exact feature-generation backbone** that supports downstream modeling, inference, and interpretation.

---

## Dataset Summary

- **Population**: Veterans  
- **Groups**:
  - TBI only (TBI‑PTSD): *n = 42*  
  - TBI + PTSD: *n = 47*  
- **Signal Source**: MEG‑derived ECG (M‑ECG)  
- **Condition**: Resting state  
- **RR Interval Format**:
  - Pickled NumPy arrays
  - Shape: `(N, 2)`
    - Column 0: cumulative time (seconds)
    - Column 1: RR intervals (seconds)

---

## Repository Structure

```
.
├── m-ecg_time_domain_hrv_extraction.py
├── m-ecg_frequency_domain_hrv_extraction.py
├── m-ecg geometric and non linear hrv extraction.py
├── machine learning pipeline.py
├── README.md
```

Each script computes a **distinct class of HRV features** and outputs a subject-level CSV file suitable for **machine-learning ingestion**.

---

## Script Descriptions

### 1. Time-Domain HRV Extraction

**File**: `m-ecg_time_domain_hrv_extraction.py`

Computes classical time-domain HRV metrics from corrected RR intervals.

**Metrics extracted**:

- Mean RR interval (ms)
- Mean heart rate (bpm)
- SDNN (ms)
- RMSSD (ms)
- SDSD (ms)
- NN50 count
- pNN50 (%)

**Key properties**:

- Uses unbiased estimators where appropriate (`ddof=1`)
- Assumes RR intervals are in seconds and converts internally to milliseconds
- Processes all subjects automatically via directory traversal

**Output**:

- CSV file containing one row per subject

---

### 2. Frequency-Domain HRV Extraction

**File**: `m-ecg_frequency_domain_hrv_extraction.py`

Computes spectral HRV features using RR-interval-based power spectral density estimation.

**Metrics extracted** (typical):

- VLF power
- LF power
- HF power
- LF/HF ratio
- Normalized LF and HF components
- Total power

**Notes**:

- Designed for resting-state RR data
- Outputs **non-normalized absolute power metrics** unless explicitly specified
- Intended for multivariate modeling rather than standalone physiological inference

---

### 3. Poincaré & Nonlinear HRV Extraction

**File**: `m-ecg geometric and non linear hrv extraction.py`

Computes nonlinear and geometric HRV measures sensitive to **complexity, irregularity, and fractal structure**.

#### Poincaré Metrics

- SD1 (short-term variability)
- SD2 (long-term variability)
- SD1/SD2 ratio
- Ellipse area (S)

#### Nonlinear Metrics

- Approximate Entropy (ApEn)
- Sample Entropy (SampEn)
- Detrended Fluctuation Analysis (DFA α1)

**Key safeguards**:

- Automatic RR unit detection (seconds vs milliseconds)
- Minimum length checks for entropy and DFA stability
- Optional plotting disabled for batch processing

---

## Machine Learning Pipeline (Downstream)

**File**: `machine learning pipeline.py`

Although not included directly in this repository, the generated HRV feature tables are used in a downstream pipeline involving:

- **Nested cross-validation**
- **Boruta feature selection**
- **Random Forest & XGBoost classifiers**
- **ROC-AUC performance evaluation**
- **SHAP-based model interpretability**

This separation is intentional to maintain a clean boundary between:

> **Physiological feature extraction** → **Statistical learning and inference**

---

## Key Findings (Project Context)

- Multivariate HRV models achieved **modest but reliable above-chance discrimination**
- Informative features reflected:
  - Reduced parasympathetic modulation
  - Altered sympathovagal balance
  - Increased low-frequency dominance
  - Elevated heart rate complexity
- Univariate HRV differences did **not survive multiple-comparison correction**

---

# Dependencies

## Required Python Version

- Python ≥ 3.8  
  *(Recommended: Python ≥ 3.9 for full compatibility with scientific libraries)*

## Core Scientific & Data Libraries

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`

## Signal Processing / Neurophysiology

- `mne`  
  *(Used for Welch power spectral density estimation of interpolated RR intervals)*

## Nonlinear HRV Analysis

- `antropy`  
  *(Approximate Entropy, Sample Entropy)*
- `nolds`  
  *(Detrended Fluctuation Analysis, DFA α1)*

## Machine Learning & Model Interpretation

- `scikit-learn`
- `xgboost`
- `boruta` (or `boruta_py`, depending on installation)
- `shap`

## Standard Library Modules (No Installation Required)

- `os`, `re`, `pickle`, `pathlib`
- `logging`, `warnings`, `itertools`, `typing`

## Installation

To install all required dependencies, run:

```bash
pip install numpy scipy pandas matplotlib mne antropy nolds scikit-learn xgboost shap boruta
```

---

## Reproducibility Notes

- All scripts operate on **pre-corrected RR intervals**
- No subject labels are hard-coded
- All file discovery is pattern-based
- Randomness is not introduced at the feature-extraction stage

---

## Intended Use

This repository is intended for:

- Neurocardiac and psychophysiological research
- HRV feature engineering for machine learning
- MEG-derived ECG analysis pipelines
- Methodological replication and extension

It is **not** intended for clinical diagnosis.

---

## Citation

If you use or adapt this code, please cite the associated manuscript and acknowledge the original author.  

DOI: 

---

## Author

**Aqil Izadysadr**  
Neurocardiac Signal Analysis & Machine Learning
