"""
===============================================================================
Title: Random Forest HRV Classification Pipeline
===============================================================================

Author:      AQIL IZADYSADR

Description:
------------
Random Forest pipeline for binary classification using HRV features. The script
performs binary group separation, nested cross-validation, training-fold correlation
filtering, Boruta feature selection, Random Forest hyperparameter tuning, held-out
prediction, SHAP analysis, bootstrap confidence intervals, VIF assessment, and
high-resolution figure generation.

Dataset-specific file paths, the subject-ID column, and the binary analysis-group
column are supplied at runtime and are not embedded in the script.

Usage:
------
python 04_ml_classification_pipeline_rf.py \
    --demo_file /path/to/metadata.xlsx \
    --metrics_dir /path/to/hrv_metrics \
    --output_dir /path/to/results \
    --group_column analysis_group

The subject-ID column can optionally be specified if it differs from the default.

Expected metric filenames inside --metrics_dir:
    - time domain hrv metrics.csv
    - frequency domain hrv metrics.csv
    - non-linear HRV metrics.csv

===============================================================================
"""

# ============================
# --- Standard Libraries ---
# ============================
import argparse
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

# ============================
# --- Data Handling & Stats ---
# ============================
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ============================
# --- Machine Learning ---
# ============================
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve, auc
)

# ============================
# --- Visualization ---
# ============================
import matplotlib
matplotlib.use('Agg')  # PRODUCTION FIX: Use non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# ----------------------------
# --- Global Configurations ---
# ----------------------------

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Fix for NumPy Deprecation Warnings
np.int = int
np.float = float
np.bool = bool

# Force Pandas to display full DataFrame content in console
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# Set Plotting Style
sns.set(style="whitegrid")


# ============================
# --- Helper Functions ---
# ============================
def apply_figure_text_sizes(fig, label_size=12, tick_size=12, legend_size=12):
    """Apply the finalized figure text sizes."""
    for ax in fig.axes:
        ax.xaxis.label.set_size(label_size)
        ax.yaxis.label.set_size(label_size)
        ax.tick_params(axis='both', which='both', labelsize=tick_size)
        ax.title.set_fontsize(label_size)
        for txt in ax.texts:
            txt.set_fontsize(tick_size)
        legend = ax.get_legend()
        if legend is not None:
            for txt in legend.get_texts():
                txt.set_fontsize(legend_size)
            title = legend.get_title()
            if title is not None:
                title.set_fontsize(legend_size)


# ============================
# --- Main Pipeline ---
# ============================
def main(args):
    demo_file = Path(args.demo_file)
    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    id_column = args.id_column
    group_column = args.group_column

    # Keep save_dir as a string because the plotting helper uses os.path.join.
    save_dir = str(output_dir)
    print("Saving outputs to:", save_dir)

    # ----------------------------
    # --- Load Data ---
    # ----------------------------
    metadata = pd.read_excel(demo_file, sheet_name=0)

    # Validate configured metadata columns before group separation.
    required_metadata_columns = {
        id_column,
        group_column,
    }
    missing_metadata_columns = sorted(required_metadata_columns.difference(metadata.columns))
    if missing_metadata_columns:
        raise ValueError(
            "Metadata file is missing required column(s): "
            + ", ".join(missing_metadata_columns)
        )

    # ----------------------------
    # --- Load HRV Metrics ---
    # ----------------------------
    TIME_METRICS_FILE = "time domain hrv metrics.csv"
    FREQUENCY_METRICS_FILE = "frequency domain hrv metrics.csv"
    NONLINEAR_METRICS_FILE = "non-linear HRV metrics.csv"

    df_non_linear = pd.read_csv(metrics_dir / NONLINEAR_METRICS_FILE)
    df_freq = pd.read_csv(metrics_dir / FREQUENCY_METRICS_FILE)
    df_time = pd.read_csv(metrics_dir / TIME_METRICS_FILE)

    for metric_name, metric_df in (
        ("time-domain", df_time),
        ("frequency-domain", df_freq),
        ("nonlinear/geometric", df_non_linear),
    ):
        if id_column not in metric_df.columns:
            raise ValueError(
                f"{metric_name} HRV file is missing the configured ID column: {id_column}"
            )

    # Merge all HRV metrics using the configured subject identifier.
    df_all_metrics = (
        df_time
        .merge(df_freq, on=id_column)
        .merge(df_non_linear, on=id_column)
    )
    df_all_metrics = df_all_metrics.drop_duplicates(subset=id_column)

    # Merge HRV metrics with the metadata table.
    data = pd.merge(metadata, df_all_metrics, on=id_column, how='right')

    # ----------------------------
    # --- Construct Binary Cohorts ---
    # ----------------------------
    # Group definitions are supplied in the metadata through a pre-defined binary
    # analysis-group column. The clinical criteria used to derive those groups are
    # described in the corresponding manuscript rather than embedded here.
    group_1 = data[data[group_column] == 0].copy()
    group_2 = data[data[group_column] == 1].copy()

    if group_1.empty or group_2.empty:
        raise ValueError(
            "The configured group column must contain both binary values 0 and 1."
        )

    print(f"Group 1: {len(group_1)}")
    print(f"Group 2: {len(group_2)}")

    # ----------------------------
    # --- Define HRV Columns ---
    # ----------------------------
    hrv_columns = [
        # ======================
        # Time-domain features
        # ======================
        "mean_rr_ms - resting state",    # average RR interval (ms)
        "sdnn_ms - resting state",       # standard deviation of NN intervals (overall HRV)
        "rmssd_ms - resting state",      # root mean square of successive differences (short-term vagal tone)

        # ======================
        # Frequency-domain features
        # ======================
        "VLF - absolute power (ms²)",    # very low frequency power
        "LF - absolute power (ms²)",     # low frequency power
        "HF - absolute power (ms²)",     # high frequency power
        
        "LF - relative power (%)",       # relative LF power
        "HF - relative power (%)",       # relative HF power
        
        "LF/HF Ratio",                   # sympathovagal balance (LF/HF ratio)
        "Total Power (ms²)",             # total spectral power
        
        "LF Peak Frequency (Hz)",        # dominant frequency in LF band
        "HF Peak Frequency (Hz)",        # dominant frequency in HF band

        # ======================
        # Geometric / Poincaré features
        # ======================
        "SD1 (ms)",                      # short-term variability (Poincaré SD1)
        "SD2 (ms)",                      # long-term variability (Poincaré SD2)
        "SD1/SD2 ratio",                 # ratio of short- to long-term variability

        # ======================
        # Nonlinear / complexity features
        # ======================
        "approximate entropy (ApEn)",    # regularity/complexity of RR intervals
        "sample entropy (SampEn)",       # robust complexity measure of RR intervals
        "DFA α1"                         # short-term fractal scaling exponent
    ]

    # ----------------------------
    # --- Prepare Data for Modeling ---
    # ----------------------------
    # Combine groups and assign target labels
    model_data = pd.concat([
        group_1.assign(target=0),
        group_2.assign(target=1)
    ], ignore_index=True)

    # Original HRV columns
    original_columns = hrv_columns

    # New, user-friendly names
    new_names = [
        "Mean RR (ms)",
        "SDNN (ms)",
        "RMSSD (ms)",
        "VLF (ms²)",
        "LF (ms²)",
        "HF (ms²)",
        "LF (% total power)",
        "HF (% total power)",
        "LF/HF ratio",
        "Total Power (ms²)",
        "LF peak frequency (Hz)",
        "HF peak frequency (Hz)",
        "SD1 (ms)",
        "SD2 (ms)",
        "SD1/SD2 ratio",
        "ApEn",
        "SampEn",
        "DFA α1"
    ]

    # Create mapping dictionary
    rename_dict = dict(zip(original_columns, new_names))

    # Rename columns in the analysis dataset
    model_data = model_data.rename(columns=rename_dict)

    # HRV features used in the primary analysis.
    hrv_features = new_names.copy()

    # Primary analysis: 18 HRV features only.
    model_features = hrv_features.copy()

    # Sensitivity analyses: uncomment ONE line and rerun the same pipeline.
    # Age and Sex remain separate columns in the metadata table.
    # model_features = hrv_features + ["Age"]
    # model_features = hrv_features + ["Sex"]

    # Select candidate features for modeling
    X_full = model_data[model_features]
    y_full = model_data['target']
    random_state = 42

    # ----------------------------
    # --- Cross-Validation Settings ---
    # ----------------------------
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    inner_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=random_state)

    # ----------------------------
    # --- Hyperparameter Grids ---
    # ----------------------------
    rf_param_grid = {
        'n_estimators': [100, 200], 'max_depth': [3, 5], 'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 5], 'max_features': ['sqrt']
    }

    # ----------------------------
    # --- Initialize Containers ---
    # ----------------------------
    aucs_rf = []
    prec_rf = []
    rec_rf = []
    f1_rf = []
    acc_rf = []

    selected_features_across_folds = []
    all_shap_vals = []
    all_X_test_sel = []
    # Full held-out feature values (before correlation/Boruta filtering), retained only
    # so SHAP plot colors use each participant's actual/imputed HRV value rather than
    # an artificial zero when a feature was not selected in that fold.
    all_X_test_full = []

    correlation_drop_counts = defaultdict(int)

    # =========================
    # Missing Data Summary
    # =========================
    missing_summary = X_full.isna().sum().to_frame(name='missing_count')
    missing_summary['missing_pct'] = 100 * missing_summary['missing_count'] / len(X_full)
    has_missing = X_full.isna().any().any()  

    print("\n=== Missing Data Summary ===")
    print(missing_summary.sort_values('missing_pct', ascending=False))

    # ----------------------------
    # --- Containers for bootstrapping metrics ---
    # ----------------------------
    all_test_preds_rf = []
    all_test_probs_rf = []
    all_test_labels = []

    # ----------------------------
    # --- Outer CV Loop ---
    # ----------------------------
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_full, y_full), 1):
        
        # --- Split Data ---
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]
        
        # --- Track fold sample counts --- 
        if 'fold_sample_summary' not in locals():
            fold_sample_summary = []
        
        fold_sample_summary.append({
            'fold': fold_idx,
            'train_total': len(y_train),
            'test_total': len(y_test),
            'train_class0': (y_train==0).sum(),
            'train_class1': (y_train==1).sum(),
            'test_class0': (y_test==0).sum(),
            'test_class1': (y_test==1).sum()
        })
        
        print(f"\n--- Fold {fold_idx} Sample Counts ---")
        print(f"Total train: {fold_sample_summary[-1]['train_total']}, class 0: {fold_sample_summary[-1]['train_class0']}, class 1: {fold_sample_summary[-1]['train_class1']}")
        print(f"Total test: {fold_sample_summary[-1]['test_total']}, class 0: {fold_sample_summary[-1]['test_class0']}, class 1: {fold_sample_summary[-1]['test_class1']}")

        # --- Imputation ---
        if has_missing:
            imputer = SimpleImputer(strategy='median')
            X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
            X_test_imp  = pd.DataFrame(imputer.transform(X_test),  columns=X_test.columns)
        else:
            X_train_imp = X_train.copy()
            X_test_imp  = X_test.copy()

        # --- Correlation Filter ---
        corr_matrix = X_train_imp.corr(method='spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
        
        for f in to_drop:
            correlation_drop_counts[f] += 1
        
        X_train_filt = X_train_imp.drop(columns=to_drop)
        X_test_filt = X_test_imp.drop(columns=to_drop)
        print(f"\nFold {fold_idx}, dropped: {to_drop}")

        # --- Feature Selection with Boruta ---
        rf_bor = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1)
        boruta = BorutaPy(rf_bor, n_estimators='auto', max_iter=50, alpha=0.1, random_state=random_state)

        boruta.fit(X_train_filt.values, y_train.values)
        feats = X_train_filt.columns[boruta.support_ | boruta.support_weak_]

        if len(feats) == 0:
            continue

        selected_features_across_folds.append(list(feats))
        
        print(f"\n--- Fold {fold_idx} Boruta Feature Selection ---")
        print(f"Number of selected features: {len(feats)}")
        for f in feats:
            print(f"  - {f}")
            
        X_train_sel = X_train_filt[feats]
        X_test_sel = X_test_filt[feats]

        # ----------------------------
        # --- Model Training ---
        # ----------------------------
        # Random Forest
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=random_state, n_jobs=-1),
            rf_param_grid, scoring='roc_auc', cv=inner_cv, n_jobs=-1
        )
        rf_grid.fit(X_train_sel, y_train)
        best_rf = rf_grid.best_estimator_
        
        print(f"\nFold {fold_idx} Best Parameters:")
        print(f"  RF: {rf_grid.best_params_}")

        # ----------------------------
        # --- Model Evaluation ---
        # ----------------------------
        # Random Forest metrics
        y_prob_rf = best_rf.predict_proba(X_test_sel)[:, 1]
        y_pred_rf = best_rf.predict(X_test_sel)
        
        aucs_rf.append(roc_auc_score(y_test, y_prob_rf))
        prec_rf.append(precision_score(y_test, y_pred_rf, zero_division=0))
        rec_rf.append(recall_score(y_test, y_pred_rf, zero_division=0))
        f1_rf.append(f1_score(y_test, y_pred_rf, zero_division=0))
        acc_rf.append(accuracy_score(y_test, y_pred_rf))
        
        all_test_preds_rf.append(y_pred_rf)
        all_test_probs_rf.append(y_prob_rf)
        all_test_labels.append(y_test.values)
        
        # ----------------------------
        # --- SHAP Analysis ---
        # ----------------------------
        explainer_rf = shap.TreeExplainer(best_rf)
        shap_vals_rf_out = explainer_rf.shap_values(X_test_sel)
        
        # Extract positive class SHAP values
        if isinstance(shap_vals_rf_out, list):
            shap_vals_rf = shap_vals_rf_out[1]
        elif len(np.array(shap_vals_rf_out).shape) == 3:
            shap_vals_rf = shap_vals_rf_out[:, :, 1]
        else:
            shap_vals_rf = shap_vals_rf_out
            
        all_shap_vals.append(shap_vals_rf)
        all_X_test_sel.append(X_test_sel)
        all_X_test_full.append(X_test_imp.copy())
        
        print(f"\nFold {fold_idx} SHAP computed for {len(feats)} features.\n")
        print("="*50 + "\n")


    # =========================
    # Correlation Filter Drop Frequency
    # =========================

    print("\n=== Correlation Filter Drop Frequency Across Outer Folds ===")
    for feature, count in sorted(correlation_drop_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: dropped in {count} of {fold_idx} folds")

    stable_dropped = [f for f, c in correlation_drop_counts.items() if c == fold_idx]
    unstable_features = [f for f, c in correlation_drop_counts.items() if 0 < c < fold_idx]

    print("\nFeatures dropped in ALL folds (highly redundant):")
    print(stable_dropped)
    print("\nFeatures inconsistently dropped across folds (unstable correlation structure):")
    print(unstable_features)


    # =========================
    # --- Performance metrics 
    # =========================

    print("=== Model Metrics Across Folds ===")

    print("\n--- Random Forest ---")
    print(f"AUC:       {np.round(aucs_rf, 3)} | mean ± std: {np.mean(aucs_rf):.3f} ± {np.std(aucs_rf):.3f}")
    print(f"Accuracy:  {np.round(acc_rf, 3)} | mean ± std: {np.mean(acc_rf):.3f} ± {np.std(acc_rf):.3f}")
    print(f"Precision: {np.round(prec_rf, 3)} | mean ± std: {np.mean(prec_rf):.3f} ± {np.std(prec_rf):.3f}")
    print(f"Recall:    {np.round(rec_rf, 3)} | mean ± std: {np.mean(rec_rf):.3f} ± {np.std(rec_rf):.3f}")
    print(f"F1:        {np.round(f1_rf, 3)} | mean ± std: {np.mean(f1_rf):.3f} ± {np.std(f1_rf):.3f}")

    # Concatenate all outer fold predictions and probabilities
    y_true_all      = np.concatenate(all_test_labels)
    y_pred_rf_all   = np.concatenate(all_test_preds_rf)
    y_prob_rf_all   = np.concatenate(all_test_probs_rf)


    # -----------------------------
    # High-resolution figure parameters
    # -----------------------------
    # The figure dimensions, text sizes, resolution, and original aspect ratios
    # are retained from the finalized plotting configuration.
    fontsize = 12
    line_width = 4
    grid_alpha = 0.3
    figure_dpi = 600

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'axes.linewidth': 3,      # original setting
        'grid.linewidth': 2,      # original setting
        'legend.fontsize': fontsize,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
    })

    def save_tiff(base_name):
        """Save the current figure as a flattened, LZW-compressed TIFF at 600 dpi."""
        tif_path = os.path.join(save_dir, f"{base_name}.tif")
        plt.savefig(
            tif_path,
            format='tiff',
            dpi=figure_dpi,
            transparent=False,
            pil_kwargs={'compression': 'tiff_lzw'}
        )
        print(f"Saved: {tif_path}")

    # Final figure geometry. Aspect ratios are unchanged.
    roc_figsize = (7.5, 7.5 * 12 / 11)
    violin_figsize = (7.5, 4.0)
    bar_figsize = (7.5, 5.0)
    dependence_figsize = (7.5, 5.0)

    # =====================================
    #        ROC CURVE GENERATION
    # =====================================
    plt.close('all')
    plt.figure(figsize=roc_figsize)

    fpr_rf, tpr_rf, _ = roc_curve(y_true_all, y_prob_rf_all)
    auc_rf = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})', linewidth=line_width, color='darkorange')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, label='Random Guessing')

    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.grid(True, linestyle='--', alpha=grid_alpha)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=True)
    fig = plt.gcf()
    fig.set_size_inches(*roc_figsize, forward=True)
    apply_figure_text_sizes(fig, label_size=12, tick_size=12, legend_size=12)
    plt.tight_layout()
    save_tiff('ROC_Curve_Random_Forest')
    # plt.show()  # PRODUCTION FIX: Removed execution blocking call
    plt.close()

    # -----------------------------
    # Bootstrapped metrics
    # -----------------------------
    n_bootstraps = 10000
    rng = np.random.default_rng(seed=42)

    def bootstrap_metrics(y_true, y_pred, y_prob=None, n_bootstraps=10000):
        boot_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        if y_prob is not None:
            boot_metrics['auc'] = []
        
        n_samples = len(y_true)
        
        for _ in range(n_bootstraps):
            indices = rng.integers(0, n_samples, n_samples)
            y_true_bs = y_true[indices]
            y_pred_bs = y_pred[indices]
            
            boot_metrics['accuracy'].append(accuracy_score(y_true_bs, y_pred_bs))
            boot_metrics['precision'].append(precision_score(y_true_bs, y_pred_bs, zero_division=0))
            boot_metrics['recall'].append(recall_score(y_true_bs, y_pred_bs, zero_division=0))
            boot_metrics['f1'].append(f1_score(y_true_bs, y_pred_bs, zero_division=0))
            
            if y_prob is not None:
                y_prob_bs = y_prob[indices]
                if len(np.unique(y_true_bs)) < 2:
                    continue
                boot_metrics['auc'].append(roc_auc_score(y_true_bs, y_prob_bs))
        
        results = {}
        for metric in boot_metrics:
            vals = np.array(boot_metrics[metric])
            mean_val = vals.mean()
            ci_lower = np.percentile(vals, 2.5)
            ci_upper = np.percentile(vals, 97.5)
            results[metric] = (mean_val, ci_lower, ci_upper)
        
        return results

    print("\n--- Bootstrapped Metrics (95% CI) ---")
    for name, y_p_cls, y_p_prob in zip(['Random Forest'], [y_pred_rf_all], [y_prob_rf_all]):
        print(f"\n{name}:")
        boot_res = bootstrap_metrics(y_true_all, y_p_cls, y_prob=y_p_prob, n_bootstraps=n_bootstraps)
        for metric, (mean_val, ci_lower, ci_upper) in boot_res.items():
            print(f"  {metric}: {mean_val:.3f} (95% CI: {ci_lower:.3f} – {ci_upper:.3f})")

    flat_feats = [f for sublist in selected_features_across_folds for f in sublist]
    feat_counts = pd.Series(flat_feats).value_counts()
    print("\nFeature selection frequencies across folds (Boruta):")
    print(feat_counts)

    # Confusion matrices (pooled)
    conf_matrix_rf_pooled = confusion_matrix(y_true_all, y_pred_rf_all)

    print("\n=== Random Forest Pooled Confusion Matrix Summary ===")

    TP = np.diag(conf_matrix_rf_pooled)
    FP = conf_matrix_rf_pooled.sum(axis=0) - TP
    FN = conf_matrix_rf_pooled.sum(axis=1) - TP
    TN = conf_matrix_rf_pooled.sum() - (TP + FP + FN)

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

    summary_df = pd.DataFrame({
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'Sensitivity (Recall)': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1 Score': f1
    }, index=[f'Class {i}' for i in range(conf_matrix_rf_pooled.shape[0])])

    summary_df = summary_df.round(3)
    print(summary_df)

    # =====================================
    #        Feature Set VIF 
    # =====================================
    important_features = list(set([feat for sublist in selected_features_across_folds for feat in sublist]))

    X_reduced = model_data[important_features].dropna()
    X_reduced_scaled = pd.DataFrame(StandardScaler().fit_transform(X_reduced), columns=important_features)
    vif_clean = pd.DataFrame({"feature": X_reduced_scaled.columns, "VIF":[variance_inflation_factor(X_reduced_scaled.values, i) for i in range(X_reduced_scaled.shape[1])]})
    print("\nReduced Feature Set VIF:")
    print(vif_clean.sort_values("VIF", ascending=False))

    # =====================================
    #            SHAP Aggregation & Viz
    # =====================================

    print(f"\n{'='*50}")
    print(f"        SHAP Aggregation: Random Forest")
    print(f"{'='*50}")

    all_shap_arrays = []
    for vals, X_df in zip(all_shap_vals, all_X_test_sel):
        df = pd.DataFrame(np.abs(vals), columns=X_df.columns, index=X_df.index)
        all_shap_arrays.append(df)

    pooled_shap_df = pd.concat(all_shap_arrays, axis=0).fillna(0)
    all_feats = pooled_shap_df.columns.tolist()

    np.random.seed(42)
    n_bootstrap = 10000
    boot_means = defaultdict(list)
    for _ in range(n_bootstrap):
        sampled = pooled_shap_df.sample(n=len(pooled_shap_df), replace=True)
        for feat in all_feats:
            boot_means[feat].append(sampled[feat].mean())

    shap_summary = pd.DataFrame({
        'mean': {f: np.mean(v) for f, v in boot_means.items()},
        'lower': {f: np.percentile(v, 2.5) for f, v in boot_means.items()},
        'upper': {f: np.percentile(v, 97.5) for f, v in boot_means.items()}
    }).sort_values("mean", ascending=True)

    print(f"\nRandom Forest Bootstrapped SHAP Feature Importance (mean ± 95% CI, pooled test sets):\n")
    for feat in shap_summary.index:
        mean_val = shap_summary.loc[feat, 'mean']
        lower_val = shap_summary.loc[feat, 'lower']
        upper_val = shap_summary.loc[feat, 'upper']
        print(f"{feat:<30}: {mean_val:.4f} [{lower_val:.4f}, {upper_val:.4f}]")

    # =====================================
    #        SHAP VISUALIZATIONS
    # =====================================
    all_feats_vis = sorted({feat for X_df in all_X_test_sel for feat in X_df.columns})

    shap_aligned_list = []
    X_test_aligned_list = []

    for shap_vals_fold, X_df_fold, X_full_fold in zip(all_shap_vals, all_X_test_sel, all_X_test_full):
        shap_fold_df = pd.DataFrame(shap_vals_fold, columns=X_df_fold.columns, index=X_df_fold.index)
        shap_fold_aligned = shap_fold_df.reindex(columns=all_feats_vis, fill_value=0)
        shap_aligned_list.append(shap_fold_aligned.values)

        X_fold_actual = X_full_fold.reindex(columns=all_feats_vis)
        X_test_aligned_list.append(X_fold_actual)

    pooled_shap_vals_aligned = np.vstack(shap_aligned_list)
    pooled_X_test_aligned = pd.concat(X_test_aligned_list, axis=0, ignore_index=True)

    if pooled_shap_vals_aligned.shape != pooled_X_test_aligned.shape:
        raise ValueError(
            "Aligned SHAP and feature-value matrices have different shapes: "
            f"{pooled_shap_vals_aligned.shape} vs {pooled_X_test_aligned.shape}"
        )

    # Violin plot (Figure 3)
    plt.close('all')
    shap.plots.violin(pooled_shap_vals_aligned,
                      pooled_X_test_aligned,
                      plot_type='violin',
                      show=False,
                      title=None,
                      plot_size=violin_figsize
                      )
    fig = plt.gcf()
    fig.set_size_inches(*violin_figsize, forward=True)
    plt.gca().set_title("")
    apply_figure_text_sizes(fig, label_size=8, tick_size=8, legend_size=8)
    plt.tight_layout()
    save_tiff('shap_violin_plot_Random_Forest')
    # plt.show()  # PRODUCTION FIX: Removed execution blocking call
    plt.close()

    # Clean Feature Importance Bar Plot
    plt.close('all')
    plt.figure(figsize=bar_figsize)
    mean_abs_shap = np.mean(np.abs(pooled_shap_vals_aligned), axis=0)
    shap_df_bar = pd.DataFrame({
        'feature': pooled_X_test_aligned.columns,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=True)

    sns.barplot(x='mean_abs_shap', y='feature', data=shap_df_bar, palette='viridis')
    plt.xlabel('Mean Absolute SHAP Value (Impact on Model Output)', fontsize=fontsize)
    plt.ylabel('')
    fig = plt.gcf()
    fig.set_size_inches(*bar_figsize, forward=True)
    apply_figure_text_sizes(fig, label_size=12, tick_size=12, legend_size=12)
    plt.tight_layout()
    save_tiff('shap_bar_importance_Random_Forest')
    # plt.show()  # PRODUCTION FIX: Removed execution blocking call
    plt.close()

    # =====================================
    #        SHAP DEPENDENCE PLOTS
    # =====================================
    dependence_specs = [
        ("LF (% total power)", None),
        ("ApEn", None),
        ("LF/HF ratio", None)
    ]

    for main_feat, color_feat in dependence_specs:
        main_values = []
        main_shap_values = []
        color_values = []

        for shap_vals_fold, X_df_fold, X_full_fold in zip(all_shap_vals, all_X_test_sel, all_X_test_full):
            if main_feat not in X_df_fold.columns:
                continue

            if color_feat is not None and color_feat not in X_df_fold.columns:
                print(
                    f"Skipping one fold for {main_feat}: {color_feat} was not "
                    "retained in that fold-specific model."
                )
                continue

            main_col_idx = X_df_fold.columns.get_loc(main_feat)
            main_shap_values.extend(shap_vals_fold[:, main_col_idx])
            main_values.extend(X_full_fold[main_feat].to_numpy())

            if color_feat is not None:
                color_values.extend(X_full_fold[color_feat].to_numpy())

        if len(main_values) == 0:
            print(f"Skipping dependence plot for {main_feat}: no eligible held-out observations.")
            continue

        main_values = np.asarray(main_values)
        main_shap_values = np.asarray(main_shap_values)

        if color_feat is None:
            feature_matrix = main_values.reshape(-1, 1)
            shap_matrix = main_shap_values.reshape(-1, 1)
            feature_names_dep = [main_feat]
            interaction_index = None
            output_base = f"shap_dependence_Random_Forest_{re.sub(r'[^\w\-_.]', '_', main_feat)}"
        else:
            color_values = np.asarray(color_values)
            feature_matrix = np.column_stack([main_values, color_values])
            shap_matrix = np.column_stack([main_shap_values, np.zeros_like(main_shap_values)])
            feature_names_dep = [main_feat, color_feat]
            interaction_index = 1

            safe_main = re.sub(r"[^\w\-_.]", "_", main_feat)
            safe_color = re.sub(r"[^\w\-_.]", "_", color_feat)
            output_base = f"shap_dependence_Random_Forest_{safe_main}_colored_by_{safe_color}"

        plt.close('all')
        plt.figure(figsize=dependence_figsize)
        shap.dependence_plot(
            ind=0,
            shap_values=shap_matrix,
            features=feature_matrix,
            feature_names=feature_names_dep,
            interaction_index=interaction_index,
            dot_size=40,
            show=False
        )

        fig = plt.gcf()
        fig.set_size_inches(*dependence_figsize, forward=True)
        plt.gca().set_title("")
        apply_figure_text_sizes(fig, label_size=12, tick_size=12, legend_size=12)
        plt.tight_layout()
        save_tiff(output_base)
        # plt.show()  # PRODUCTION FIX: Removed execution blocking call
        plt.close()


# ============================
# --- Entry Point ---
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Random Forest HRV binary-classification pipeline."
    )
    parser.add_argument(
        "-d", "--demo_file",
        type=str,
        required=True,
        help="Path to the metadata file."
    )
    parser.add_argument(
        "-m", "--metrics_dir",
        type=str,
        required=True,
        help="Directory containing the HRV metric files."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Directory in which analysis outputs will be saved."
    )
    parser.add_argument(
        "--id_column",
        type=str,
        default="SUBJECT_ID",
        help="Identifier column shared by the metadata and HRV metric files."
    )
    parser.add_argument(
        "--group_column",
        type=str,
        required=True,
        help="Binary analysis-group column (0 = Group 1, 1 = Group 2)."
    )

    args = parser.parse_args()
    main(args)