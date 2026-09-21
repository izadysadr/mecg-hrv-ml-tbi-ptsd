# Multivariate Machine Learning Analysis of M-ECG-derived Heart Rate Variability in TBI Veterans, With and Without Current PTSD and Additional Psychiatric Comorbidity

## Overview

This repository contains the analysis scripts used to derive heart-rate-variability (HRV) features from corrected M-ECG RR intervals and to run the Random Forest classification pipeline used in the study.

The workflow is divided into two stages:

1. **HRV feature extraction** from corrected RR intervals across time-domain, frequency-domain, geometric, and nonlinear domains.
2. **Random Forest classification and interpretation** using nested cross-validation, training-fold correlation filtering, Boruta feature selection, hyperparameter tuning, bootstrap confidence intervals, and SHAP analysis.

The scripts are designed to be portable and do not contain study-specific local paths or participant identifiers. Dataset-specific paths, metadata column names, optional exclusions, and output locations are supplied at runtime.

The repository begins with **corrected RR intervals**. Extraction of M-ECG cardiac signals and RR-interval correction is handled separately by the M-ECG extraction framework:

https://github.com/izadysadr/MEG-HRV-Extraction

---

## Repository Structure

```text
.
├── 01_extract_hrv_time.py
├── 02_extract_hrv_frequency.py
├── 03_extract_hrv_nonlinear.py
├── 04_ml_classification_pipeline_rf.py
├── LICENSE
└── README.md
```

The first three scripts generate the **18 HRV features used in the analysis**. The fourth script merges those feature files with a metadata table and runs the Random Forest classification pipeline.

---

## Input RR-Interval Format

The three HRV extraction scripts expect pre-corrected RR interval files with names matching:

```text
*_rr_intervals_corrected.np.pkl
```

Each file should contain a two-column NumPy array:

```text
Column 0: cumulative time / time stamp in seconds
Column 1: corrected RR interval in seconds
```

The extraction scripts recursively discover matching files beneath the supplied input directory.

---

## HRV Features Used in the Analysis

The current scripts compute and export only the 18 HRV features used by the classification pipeline.

### Time-domain features (3)

* Mean RR interval (ms)
* SDNN (ms)
* RMSSD (ms)

### Frequency-domain features (9)

* VLF absolute power (ms²)
* LF absolute power (ms²)
* HF absolute power (ms²)
* LF relative power (% total power)
* HF relative power (% total power)
* LF/HF ratio
* Total power (ms²)
* LF peak frequency (Hz)
* HF peak frequency (Hz)

### Geometric and nonlinear features (6)

* SD1 (ms)
* SD2 (ms)
* SD1/SD2 ratio
* Approximate entropy (ApEn)
* Sample entropy (SampEn)
* DFA α1

---

## Script Descriptions and Usage

### 1. Time-Domain HRV Extraction

**File:** `01_extract_hrv_time.py`

Computes the three time-domain variables used in the study directly from corrected RR intervals:

* Mean RR interval
* SDNN
* RMSSD

**Usage:**

```bash
python 01_extract_hrv_time.py \\
    -i /path/to/corrected_rr_intervals \\
    -o "/path/to/hrv_metrics/time domain hrv metrics.csv"
```

---

### 2. Frequency-Domain HRV Extraction

**File:** `02_extract_hrv_frequency.py`

Performs frequency-domain analysis from corrected RR intervals. RR intervals are converted to milliseconds and resampled to **850 evenly spaced points using cubic-spline interpolation**. Power spectral density is estimated over **0–0.4 Hz using Welch's method**.

Frequency bands are defined as:

* VLF: 0.003–0.04 Hz
* LF: 0.04–0.15 Hz
* HF: 0.15–0.40 Hz

Absolute band powers are obtained by trapezoidal integration of the PSD. Total power is defined as VLF + LF + HF band power, and LF and HF relative power are calculated as percentages of that total. The script also derives the LF/HF ratio and LF/HF peak frequencies.

**Usage:**

```bash
python 02_extract_hrv_frequency.py \\
    -i /path/to/corrected_rr_intervals \\
    -o "/path/to/hrv_metrics/frequency domain hrv metrics.csv"
```

---

### 3. Geometric and Nonlinear HRV Extraction

**File:** `03_extract_hrv_nonlinear.py`

Computes the geometric and nonlinear variables used in the analysis:

* SD1
* SD2
* SD1/SD2 ratio
* Approximate entropy (ApEn)
* Sample entropy (SampEn)
* DFA α1

Entropy calculations use an embedding dimension of **2**. When no tolerance is supplied, the tolerance is set to **0.2 × the standard deviation of the RR series**. DFA α1 is calculated over scales of **4–16 beats**.

**Usage:**

```bash
python 03_extract_hrv_nonlinear.py \\
    -i /path/to/corrected_rr_intervals \\
    -o "/path/to/hrv_metrics/non-linear HRV metrics.csv"
```

Using the three output filenames shown above allows the machine-learning script to use its default metric-file settings. Alternative filenames can also be supplied to the machine-learning script through command-line arguments.

---

### 4. Random Forest Classification Pipeline

**File:** `04_ml_classification_pipeline_rf.py`

Runs the Random Forest-only classification and interpretation pipeline. The script is intentionally generic: metadata column names, subject identifier, exclusions, file locations, and metric filenames are supplied at runtime rather than embedded in the source code.

The pipeline includes:

* Five-fold stratified **outer cross-validation** for held-out model evaluation
* Repeated stratified **3-fold × 5-repeat inner cross-validation** for hyperparameter tuning
* Median imputation, when required, fitted only within each training fold
* Training-fold **Spearman correlation filtering** at `|ρ| > 0.90`
* Training-fold **Boruta feature selection**, retaining confirmed and tentative features
* Random Forest hyperparameter tuning with `GridSearchCV`
* Held-out predictions and probabilities from each outer fold
* ROC AUC, accuracy, precision, recall, F1-score, and a pooled confusion-matrix summary
* **10,000-sample bootstrap confidence intervals** for pooled performance metrics
* VIF summaries for features selected across folds
* Held-out **SHAP** analysis and bootstrap summaries of mean absolute SHAP importance
* Global SHAP violin/bar plots and dependence plots for selected features
* High-resolution LZW-compressed TIFF figure output

#### Cohort configuration

The metadata table must contain a subject-ID column and four binary columns supplied through the following arguments:

* `--group_status_column`
* `--primary_condition_column`
* `--comorbidity_column`
* `--lifetime_status_column`

The existing analysis logic constructs:

```text
Group 0: group_status = 0, primary_condition = 1,
         comorbidity = 0, lifetime_status = 0

Group 1: group_status = 1, primary_condition = 1,
         comorbidity = 1, lifetime_status = 1
```

Group 0 receives target label `0`; Group 1 receives target label `1`.

#### Required usage

```bash
python 04_ml_classification_pipeline_rf.py \\
    --demo_file /path/to/metadata.xlsx \\
    --metrics_dir /path/to/hrv_metrics \\
    --output_dir /path/to/results \\
    --group_status_column group_status \\
    --primary_condition_column primary_condition \\
    --comorbidity_column comorbidity_status \\
    --lifetime_status_column lifetime_status
```

#### Common optional arguments

```text
--id_column SUBJECT_ID
--age_column age
--sex_column sex
--exclude_ids ID001 ID002 ...
--time_metrics_file "time domain hrv metrics.csv"
--frequency_metrics_file "frequency domain hrv metrics.csv"
--nonlinear_metrics_file "non-linear HRV metrics.csv"
```

`--age_column` and `--sex_column` are optional and are used only for descriptive console summaries. `--exclude_ids` allows exclusions to be supplied at runtime without embedding participant identifiers in the public script.

Use:

```bash
python 04_ml_classification_pipeline_rf.py --help
```

for the complete command-line interface.

---

## Machine-Learning Feature Set

The classification script expects the 18 HRV variables generated by scripts 01–03 and renames them internally to concise display names before modeling.

No age or sex variable is included in the model feature matrix by default. Optional age and sex arguments are used only for descriptive output.

---

## SHAP Analysis

SHAP values are computed only for **held-out outer-fold observations** using the Random Forest model trained within that fold.

Because Boruta may retain different features across folds, global SHAP aggregation aligns features across outer folds. A feature that was absent from a fold's fitted model contributes a structural zero SHAP value for observations from that fold. Observed held-out feature values are retained separately for visualization.

Dependence plots include only held-out observations from folds in which the focal feature was actually retained by Boruta, preventing structural-zero SHAP values from being interpreted as genuine focal-feature relationships.

---

## Outputs

Depending on the supplied data and selected features, the Random Forest pipeline saves the following high-resolution, LZW-compressed TIFF figures:

* ROC curve
* SHAP violin plot
* SHAP mean-absolute-importance bar plot
* SHAP dependence plots for selected features

The script also prints analysis summaries to the console, including:

* Fold-wise model-performance summaries
* Bootstrap confidence intervals for pooled performance metrics
* Feature-selection frequencies
* Correlation-filter drop frequencies
* Pooled confusion-matrix summary
* VIF summary
* Bootstrapped SHAP importance summaries

---

## Dependencies

### Python

Python 3.9+ is recommended.

### Installation

```bash
pip install numpy pandas scipy matplotlib seaborn statsmodels mne antropy nolds scikit-learn shap boruta openpyxl
```

`openpyxl` is required when reading `.xlsx` metadata files with pandas.

### Main library roles

* **Numerical/data handling:** `numpy`, `pandas`, `scipy`
* **Frequency-domain HRV:** `mne`
* **Nonlinear HRV:** `antropy`, `nolds`
* **Machine learning:** `scikit-learn`, `boruta`
* **Interpretability:** `shap`
* **Statistics:** `statsmodels`, `scipy`
* **Visualization:** `matplotlib`, `seaborn`
* **Excel input:** `openpyxl`

---

## Reproducibility Notes

* Scripts 01–03 operate on **pre-corrected RR intervals** and do not perform M-ECG extraction or participant-specific R-peak quality control.
* The extraction scripts compute/export only the **18 HRV features used by the classification analysis**.
* The machine-learning script contains no embedded local data paths or participant IDs.
* Optional subject exclusions are provided at runtime.
* The Random Forest analysis uses a fixed random state of `42` for reproducible cross-validation, feature selection, model fitting, and bootstrap procedures where applicable.
* Imputation, correlation filtering, Boruta feature selection, and hyperparameter tuning are performed using training data within the nested cross-validation structure to avoid leakage into held-out outer folds.
* SHAP explanations are calculated from held-out outer-fold observations rather than training observations.

---

## Data Availability and Privacy

This repository contains analysis code only. Raw participant data and corrected participant-level RR interval files are not distributed with the repository.

Users applying the workflow to their own data should ensure that their metadata and RR-interval files follow the formats described above and that they have appropriate authorization to use those data.

---

## Intended Use

This repository is intended for methodological replication and research involving:

* Heart-rate-variability feature extraction
* M-ECG-derived RR interval analysis
* Multivariate HRV classification
* Nested cross-validation and feature-selection workflows
* Random Forest interpretation using SHAP

It is not intended for clinical diagnosis or clinical decision-making.

