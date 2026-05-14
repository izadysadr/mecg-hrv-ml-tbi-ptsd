# Multivariate Machine Learning Analysis of M-ECG-derived Heart Rate Variability in TBI With and Without Comorbid PTSD

---

## Overview

This repository contains the working analysis code used in the study:

> **"Multivariate Machine Learning Analysis of M-ECG-derived Heart Rate Variability in TBI Individuals With and Without Comorbid PTSD"**

The project investigates whether multivariate machine-learning models applied to heart rate variability (HRV) features extracted from MEG-derived electrocardiogram (M-ECG) signals can differentiate Veterans with:

- **TBI alone**
- **TBI with comorbid PTSD (TBI+PTSD)**

Rather than relying solely on isolated univariate HRV comparisons, this pipeline extracts:

- Time-domain HRV metrics
- Frequency-domain HRV metrics
- Geometric HRV metrics
- Nonlinear HRV metrics

These features are evaluated using:

- Nested cross-validated machine learning
- Feature selection
- Statistical inference
- Model interpretability techniques

---

## Scientific Motivation

TBI and PTSD frequently co-occur and share overlapping symptomatology and autonomic dysfunction. Traditional HRV analyses often fail to detect robust group differences after correction for multiple comparisons.

This project demonstrates that:

- **Multivariate HRV patterns** contain diagnostically relevant information even when univariate effects are weak or absent.
- **Machine learning approaches** can uncover distributed autonomic signatures associated with psychiatric and neurological comorbidity.

### Manuscript Information

- DOI: *To be added*
- PMID: *To be added*

The code in this repository reflects the exact feature-generation backbone and statistical modeling pipeline used for downstream inference and interpretation.

---

## Dataset Summary

### Population

Veterans drawn from the **Chronic Effects of Neurotrauma Consortium (CENC) Study 34**

### Groups

| Group | Sample Size |
|---|---|
| TBI only (TBI-alone) | n = 42 |
| TBI + PTSD | n = 40 |

### Signal Source

- M-ECG (MEG-derived ECG)

### Recording Condition

- Resting-state
- 5-minute recordings

### RR Interval Format

Pickled NumPy arrays with shape:

```python
(N, 2)
```

| Column | Description |
|---|---|
| Column 0 | Cumulative time (seconds) |
| Column 1 | RR intervals (seconds) |

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

The scripts are sequentially numbered to reflect execution order.

- Scripts 1–3 compute HRV feature sets and generate subject-level CSV files.
- Script 4 ingests these features alongside demographic variables to execute the machine-learning workflow.

---

# Script Descriptions & Usage

All scripts support command-line execution using `argparse`.

---

## 1. Time-Domain HRV Extraction

### File

```text
01_extract_hrv_time.py
```

### Description

Computes classical time-domain HRV metrics from corrected RR intervals.

### Metrics Extracted

- Mean RR interval (ms)
- Mean heart rate (bpm)
- SDNN (ms)
- RMSSD (ms)
- SDSD (ms)
- NN50 count
- pNN50 (%)

### Usage

```bash
python 01_extract_hrv_time.py \
    -i /path/to/data \
    -o /path/to/save/time_domain_hrv_metrics.csv
```

---

## 2. Frequency-Domain HRV Extraction

### File

```text
02_extract_hrv_frequency.py
```

### Description

Computes spectral HRV features using Welch power spectral density estimation constrained to physiological frequency ranges (0–0.4 Hz).

### Metrics Extracted

- Absolute power:
  - VLF
  - LF
  - HF

- Relative power:
  - LF (%)
  - HF (%)

- LF/HF ratio
- LF peak frequency
- HF peak frequency
- Total power

### Usage

```bash
python 02_extract_hrv_frequency.py \
    -i /path/to/data \
    -o /path/to/save/frequency_domain_hrv_metrics.csv
```

---

## 3. Poincaré & Nonlinear HRV Extraction

### File

```text
03_extract_hrv_nonlinear.py
```

### Description

Computes nonlinear and geometric HRV measures sensitive to:

- Signal complexity
- Irregularity
- Fractal structure

### Metrics Extracted

#### Poincaré Metrics

- SD1
- SD2
- SD1/SD2 ratio
- Ellipse area (S)

#### Nonlinear Metrics

- Approximate Entropy (ApEn)
- Sample Entropy (SampEn)
- DFA α1

### Usage

```bash
python 03_extract_hrv_nonlinear.py \
    -i /path/to/data \
    -o /path/to/save/non_linear_hrv_metrics.csv
```

---

## 4. Machine Learning Classification Pipeline

### File

```text
04_ml_classification_pipeline.py
```

### Description

Ingests the previously generated HRV metrics alongside clinical demographic data to execute a rigorous, leakage-free modeling pipeline.

### Core Pipeline Features

- **Nested Cross-Validation**
  - 5-fold outer CV for evaluation
  - Repeated 3-fold inner CV for hyperparameter tuning

- **Feature Selection**
  - Correlation-based redundancy filtering
  - All-relevant wrapper-based selection using Boruta

- **Classification**
  - Optimized XGBoost
  - Random Forest models

- **Statistical Inference**
  - Exact paired permutation testing for AUC comparison
  - Nonparametric bootstrapping (`n = 10,000`) for 95% confidence intervals

- **Explainability**
  - SHapley Additive exPlanations (SHAP)
  - Global feature importance and dependence visualization pooled across test folds

### Usage

```bash
python 04_ml_classification_pipeline.py \
    --demo_file /path/to/demographics.xlsx \
    --metrics_dir /path/to/HRV_Metrics_Folder \
    --output_dir /path/to/save/results
```

---

# Key Findings

- Random Forest achieved:
  - **AUC = 0.663**

- XGBoost achieved:
  - **AUC = 0.635**

### Important Predictive Features

- LF/HF ratio
- LF % total power
- Approximate Entropy (ApEn)

### Interpretation

Univariate HRV differences were subtle and did **not survive multiple-comparison correction** (`FDR-adjusted p ≥ 0.13`), emphasizing the value of multivariate modeling approaches.

---

# Dependencies

## Python Version

- Python ≥ 3.8
- Recommended: Python ≥ 3.9

## Installation

```bash
pip install numpy scipy pandas matplotlib seaborn statsmodels \
mne antropy nolds scikit-learn xgboost shap boruta
```

---

## Library Roles

| Category | Libraries |
|---|---|
| Neurophysiology / Signal Processing | `mne` |
| Nonlinear Dynamics | `antropy`, `nolds` |
| Machine Learning | `scikit-learn`, `xgboost`, `boruta` |
| Explainability & Statistics | `shap`, `statsmodels`, `scipy` |

### Standard Libraries Used

```python
os
re
pickle
pathlib
logging
warnings
itertools
typing
```

---

# Reproducibility Notes

- All extraction scripts operate on pre-corrected RR intervals.
- No subject labels or paths are hard-coded.
- Random seeds (`RANDOM_STATE = 42`) are enforced throughout the ML workflow.
- Feature selection and preprocessing are isolated strictly within training folds to prevent data leakage.

---

# Intended Use

This repository is intended for:

- Neurocardiac research
- Psychophysiology
- HRV feature engineering
- M-ECG analysis pipelines
- Methodological replication and extension

> **Note:** This repository is not intended for clinical diagnosis.

---

# Citation

If you use or adapt this code, please cite the associated manuscript:

> Izadysadr, A., Bagherzadeh, H. S., Rowland, J., Stapleton-Kotloski, J. R., & Godwin, D. W. (2026). *Multivariate Machine Learning Analysis of M-ECG-derived Heart Rate Variability in TBI Individuals With and Without Comorbid PTSD*.

DOI: *To be added*

---

# Author

## Aqil Izadysadr

Department of Neurology  
Wake Forest School of Medicine

### Research Focus

- Neurocardiac Signal Analysis
- Heart Rate Variability
- Machine Learning
- Computational Neuroscience