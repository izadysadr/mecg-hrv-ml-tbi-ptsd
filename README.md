# Multivariate Machine Learning Analysis of M-ECG-derived Heart Rate Variability in TBI With and Without Comorbid PTSD 

---

## Overview

This repository contains the **working analysis code** used in the study:

**"Multivariate Machine Learning Analysis of M-ECG-derived Heart Rate Variability in TBI Individuals With and Without Comorbid PTSD"** 

The project investigates whether multivariate machine‑learning models applied to **heart rate variability (HRV)** features extracted from **MEG-derived electrocardiogram (M‑ECG)** signals can differentiate Veterans with **TBI alone** from those with **comorbid PTSD (TBI+PTSD)**.

Rather than relying on isolated univariate HRV comparisons, this pipeline extracts **time-domain, frequency-domain, geometric, and nonlinear HRV metrics** and evaluates their **joint discriminative structure** using **nested cross-validated machine learning**, feature selection, and model interpretability techniques.

---

## Scientific Motivation

TBI and PTSD frequently co-occur and share overlapping symptomatology and autonomic dysfunction. Traditional HRV analyses often fail to detect robust group differences after correction for multiple comparisons. This project demonstrates that:

**Multivariate HRV patterns** carry diagnostically relevant information even when univariate HRV effects are weak or absent.


**Machine learning models** can uncover distributed autonomic signatures linked to comorbidity.



Manuscript Link:

DOI: [To be added]

PMID: [To be added]

The code here reflects the **exact feature-generation backbone** and **statistical modeling pipeline** that supports downstream inference and interpretation.

---

## Dataset Summary

* 
**Population**: Veterans (drawn from the Chronic Effects of Neurotrauma Consortium Study 34).


* **Groups**:
* TBI only (TBI-alone): *n = 42* 


* TBI + PTSD: *n = 40* 




**Signal Source**: M-ECG 


**Condition**: Resting state (5 minutes) 


* **RR Interval Format**:
* Pickled NumPy arrays
* Shape: `(N, 2)`
* Column 0: cumulative time (seconds)
* Column 1: RR intervals (seconds)





---

## Repository Structure

```text
.
├── 01_extract_hrv_time.py
├── 02_extract_hrv_frequency.py
├── 03_extract_hrv_nonlinear.py
├── 04_ml_classification_pipeline.py
├── README.md

```

The scripts are sequentially numbered to reflect the execution order. The first three compute distinct classes of HRV features and output subject-level CSV files. The fourth script ingests these CSVs alongside clinical demographics to run the machine-learning pipeline.

---

## Script Descriptions & Usage

All scripts are configured for command-line execution, utilizing `argparse` to handle input directories and output destinations cleanly.

### 1. Time-Domain HRV Extraction

**File**: `01_extract_hrv_time.py`

Computes classical time-domain HRV metrics from corrected RR intervals.

**Metrics extracted**:

* Mean RR interval (ms), Mean heart rate (bpm), SDNN (ms), RMSSD (ms), SDSD (ms), NN50 count, pNN50 (%)

**Usage**:

```bash
python 01_extract_hrv_time.py -i /path/to/data -o /path/to/save/time_domain_hrv_metrics.csv

```

### 2. Frequency-Domain HRV Extraction

**File**: `02_extract_hrv_frequency.py`

Computes spectral HRV features using RR-interval-based power spectral density estimation (Welch's method) bounded to standard physiological ranges (0–0.4 Hz).

**Metrics extracted**:

* Absolute and relative power for VLF, LF, and HF bands
* LF/HF ratio
* LF and HF peak frequencies (Hz)
* Total power

**Usage**:

```bash
python 02_extract_hrv_frequency.py -i /path/to/data -o /path/to/save/frequency_domain_hrv_metrics.csv

```

### 3. Poincaré & Nonlinear HRV Extraction

**File**: `03_extract_hrv_nonlinear.py`

Computes nonlinear and geometric HRV measures sensitive to **complexity, irregularity, and fractal structure**.

**Metrics extracted**:

* **Poincaré Metrics**: SD1, SD2, SD1/SD2 ratio, Ellipse area (S)
* **Nonlinear Metrics**: Approximate Entropy (ApEn), Sample Entropy (SampEn), Detrended Fluctuation Analysis (DFA α1)

**Usage**:

```bash
python 03_extract_hrv_nonlinear.py -i /path/to/data -o /path/to/save/non_linear_hrv_metrics.csv

```

### 4. Machine Learning Classification Pipeline

**File**: `04_ml_classification_pipeline.py`

Ingests the previously generated HRV metrics alongside clinical demographic data to execute a rigorous, leakage-free modeling pipeline.

**Core Pipeline Features**:

**Nested Cross-Validation**: 5-fold outer CV for evaluation; repeated 3-fold inner CV for hyperparameter tuning.


**Feature Selection**: Correlation-based redundancy filtering followed by all-relevant wrapper-based selection using Boruta.


**Classification**: Optimized XGBoost and Random Forest models.


**Statistical Inference**: Exact paired permutation testing for AUC comparison, and nonparametric bootstrapping (n=10,000) for 95% confidence intervals.


**Explainability**: SHapley Additive exPlanations (SHAP) pooled across test folds for global feature importance and dependence visualization.



**Usage**:

```bash
python 04_ml_classification_pipeline.py \
    --demo_file /path/to/demographics.xlsx \
    --metrics_dir /path/to/HRV_Metrics_Folder \
    --output_dir /path/to/save/results

```

---

## Key Findings (Project Context)

* Both Random Forest and XGBoost classifiers achieved above-chance discrimination (Random Forest AUC = 0.663; XGBoost AUC = 0.635).


* Informative features driving the models included:
* Altered sympathovagal balance (LF/HF ratio) 


* Increased low-frequency proportion (LF % total power) 


* Greater heart rate complexity (ApEn) 




* Univariate HRV differences were subtle and did **not survive multiple-comparison correction** (FDR-adjusted p ≥ 0.13), highlighting the necessity of the multivariate approach.



---

## Dependencies

### Required Python Version

* Python ≥ 3.8 *(Recommended: Python ≥ 3.9 for full compatibility with scientific libraries)*

### Installation

To install all required dependencies, run:

```bash
pip install numpy scipy pandas matplotlib seaborn statsmodels mne antropy nolds scikit-learn xgboost shap boruta

```

**Library Roles**:

**Neurophysiology / Signal Processing**: `mne` 


**Nonlinear Dynamics**: `antropy`, `nolds` 


**Machine Learning**: `scikit-learn`, `xgboost`, `boruta` 


**Interpretability & Stats**: `shap`, `statsmodels`, `scipy` 


**Standard Libraries** (built-in): `os`, `re`, `pickle`, `pathlib`, `logging`, `warnings`, `itertools`, `typing`

---

## Reproducibility Notes

* All extraction scripts operate on **pre-corrected RR intervals**.
* No subject labels or paths are hard-coded; all file discovery is dynamic.
* Random seeds (e.g., `RANDOM_STATE = 42`) are strictly enforced throughout the ML pipeline for CV splits, Boruta shadow features, model initialization, and bootstrapping to guarantee reproducibility.


* The ML pipeline isolates imputation, correlation filtering, and feature selection entirely within the training folds to strictly prevent data leakage.



---

## Intended Use

This repository is intended for:

* Neurocardiac and psychophysiological research
* HRV feature engineering for machine learning
* M-ECG analysis pipelines
* Methodological replication and extension

*It is **not** intended for clinical diagnosis.*

---

## Citation

If you use or adapt this code, please cite the associated manuscript and acknowledge the original author:

Izadysadr, A., Bagherzadeh, H. S., Rowland, J., Stapleton-Kotloski, J. R., & Godwin, D. W. (2026). *Multivariate Machine Learning Analysis of M-ECG-derived Heart Rate Variability in TBI Individuals With and Without Comorbid PTSD*. 

DOI: [To be added]

---

## Author

**Aqil Izadysadr** Department of Neurology, Wake Forest School of Medicine 
Neurocardiac Signal Analysis & Machine Learning
