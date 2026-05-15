#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Title: HRV-Based Machine Learning Pipeline for TBI/PTSD Classification 
===============================================================================

Author:      AQIL IZADYSADR
Created:     August 25, 2025

Description:
------------
This production-ready script implements a comprehensive nested cross-validation
machine learning pipeline for distinguishing traumatic brain injury (TBI)
participants with PTSD from TBI participants without PTSD using resting-state
heart rate variability (HRV) metrics.

Usage:
------
Run via command line specifying the inputs and output directory:

    python 04_ml_classification_pipeline.py \
        --demo_file /path/to/demographics.xlsx \
        --metrics_dir /path/to/HRV_metrics_folder \
        --output_dir /path/to/save/results

===============================================================================
"""

import argparse
import itertools
import logging
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from boruta import BorutaPy
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =============================================================================
# --- CONFIGURATION & LOGGING ---
# =============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
)
logger = logging.getLogger(__name__)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# Compatibility patch for older libraries
np.int = int
np.float = float
np.bool = bool

# Global reproducibility seed
RANDOM_STATE = 42

# =============================================================================
# --- GLOBAL VARIABLES ---
# =============================================================================

HRV_COLUMNS = [
    "mean_rr_ms - resting state", "sdnn_ms - resting state", "rmssd_ms - resting state",
    "VLF - absolute power (ms²)", "LF - absolute power (ms²)", "HF - absolute power (ms²)",
    "LF - relative power (%)", "HF - relative power (%)", "LF/HF Ratio", "Total Power (ms²)",
    "LF Peak Frequency (Hz)", "HF Peak Frequency (Hz)",
    "SD1 (ms)", "SD2 (ms)", "SD1/SD2 ratio",
    "approximate entropy (ApEn)", "sample entropy (SampEn)", "DFA α1"
]

NEW_NAMES = [
    "Mean RR (ms)", "SDNN (ms)", "RMSSD (ms)",
    "VLF (ms²)", "LF (ms²)", "HF (ms²)",
    "LF (% total power)", "HF (% total power)", "LF/HF ratio", "Total Power (ms²)",
    "LF peak frequency (Hz)", "HF peak frequency (Hz)",
    "SD1 (ms)", "SD2 (ms)", "SD1/SD2 ratio",
    "ApEn", "SampEn", "DFA α1"
]

RENAME_DICT = dict(zip(HRV_COLUMNS, NEW_NAMES))

XGB_PARAM_GRID = {
    'max_depth': [3, 5], 'min_child_weight': [1, 3], 'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8], 'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1],
    'reg_alpha': [0, 0.1, 1], 'reg_lambda': [1, 5, 10], 'gamma': [0, 0.1, 0.5]
}

RF_PARAM_GRID = {
    'n_estimators': [100, 200], 'max_depth': [3, 5], 'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 5], 'max_features': ['sqrt']
}


# =============================================================================
# --- PIPELINE FUNCTIONS ---
# =============================================================================

def load_and_prepare_data(demo_file: Path, metrics_dir: Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Loads clinical demographics and HRV metric datasets, merges them, and isolates cohorts.

    This function reads time-domain, frequency-domain, and non-linear HRV metrics, merges
    them with clinical demographics via Subject_ID, and specifically extracts cohorts
    of TBI participants with and without current PTSD. 

    Args:
        demo_file (Path): Absolute or relative path to the clinical demographics Excel file.
        metrics_dir (Path): Directory containing the structured HRV metric CSV files.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame]: A 3-tuple containing:
            - X_full: The feature matrix containing only renamed HRV metrics.
            - y_full: The binary target variable (1 for PTSD, 0 for no PTSD).
            - data_tbi: The complete, un-subsetted merged dataframe for downstream statistical use.
    """
    logger.info("Loading demographic and clinical datasets...")
    df = pd.read_excel(demo_file)
    
    df_non_linear = pd.read_csv(metrics_dir / 'non-linear HRV metrics.csv')
    df_freq = pd.read_csv(metrics_dir / 'frequency domain hrv metrics.csv')
    df_time = pd.read_csv(metrics_dir / 'time domain hrv metrics.csv')

    df_all_metrics = (
        df_time
        .merge(df_freq, on='Subject_ID')
        .merge(df_non_linear, on='Subject_ID')
    ).drop_duplicates(subset="Subject_ID")

    df_merged = pd.merge(df, df_all_metrics, on='Subject_ID', how='right')

    # Strict cohort isolation based on current psychological and historical diagnoses
    TBI_and_no_PTSD = df_merged[
        (df_merged["PTSD_current"] == 0) & (df_merged["TBI"] == 1) &
        (df_merged["CurrentPsychDisorder"] == 0) & (df_merged["PTSD_Life"] == 0)
    ]
    TBI_and_PTSD = df_merged[
        (df_merged["PTSD_current"] == 1) & (df_merged["TBI"] == 1) &
        (df_merged["CurrentPsychDisorder"] == 1) & (df_merged["PTSD_Life"] == 1)
    ]

    logger.info(f"TBI_and_no_PTSD_merged: {len(TBI_and_no_PTSD)}")
    logger.info(f"TBI_and_PTSD_merged: {len(TBI_and_PTSD)}")

    # Group Demographic Summaries
    logger.info("\nTBI and PTSD:")
    logger.info(f"Average age: {TBI_and_PTSD['demogageyears'].mean()}")
    logger.info("Sex distribution (%):")
    logger.info((TBI_and_PTSD['sex'].value_counts(normalize=True) * 100).round(2).to_string())

    logger.info("\n--- TBI without PTSD ---")
    logger.info(f"Average age: {round(TBI_and_no_PTSD['demogageyears'].mean(), 2)}")
    logger.info("Sex distribution (%):")
    logger.info((TBI_and_no_PTSD['sex'].value_counts(normalize=True) * 100).round(2).to_string())

    data_tbi = pd.concat([
        TBI_and_PTSD.assign(target=1),
        TBI_and_no_PTSD.assign(target=0)
    ], ignore_index=True)

    data_tbi = data_tbi.rename(columns=RENAME_DICT)
    X_full = data_tbi[NEW_NAMES]
    y_full = data_tbi['target']

    missing_summary = X_full.isna().sum().to_frame(name='missing_count')
    missing_summary['missing_pct'] = 100 * missing_summary['missing_count'] / len(X_full)
    
    logger.info("\n=== Missing Data Summary ===")
    logger.info(missing_summary.sort_values('missing_pct', ascending=False).to_string())

    return X_full, y_full, data_tbi

def bootstrap_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: Optional[np.ndarray] = None, 
    n_bootstraps: int = 10000
) -> Dict[str, Tuple[float, float, float]]:
    """
    Computes bootstrapped performance metrics to generate 95% confidence intervals.

    Args:
        y_true (np.ndarray): Array of true binary ground-truth labels.
        y_pred (np.ndarray): Array of binary predicted labels from the model.
        y_prob (Optional[np.ndarray]): Array of predicted probabilities, required for AUC.
        n_bootstraps (int): The number of bootstrap sampling iterations. Default is 10000.

    Returns:
        Dict[str, Tuple[float, float, float]]: A dictionary mapping metric strings 
        ('accuracy', 'precision', 'recall', 'f1', 'auc') to a tuple containing the 
        (mean value, 2.5th percentile CI limit, 97.5th percentile CI limit).
    """
    rng = np.random.default_rng(seed=RANDOM_STATE)
    boot_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    if y_prob is not None:
        boot_metrics['auc'] = []

    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        indices = rng.integers(0, n_samples, n_samples)
        y_true_bs, y_pred_bs = y_true[indices], y_pred[indices]

        boot_metrics['accuracy'].append(accuracy_score(y_true_bs, y_pred_bs))
        boot_metrics['precision'].append(precision_score(y_true_bs, y_pred_bs, zero_division=0))
        boot_metrics['recall'].append(recall_score(y_true_bs, y_pred_bs, zero_division=0))
        boot_metrics['f1'].append(f1_score(y_true_bs, y_pred_bs, zero_division=0))

        if y_prob is not None:
            y_prob_bs = y_prob[indices]
            # Handle edge cases where bootstrap sample contains only one class
            if len(np.unique(y_true_bs)) < 2:
                continue
            boot_metrics['auc'].append(roc_auc_score(y_true_bs, y_prob_bs))

    results = {}
    for metric in boot_metrics:
        vals = np.array(boot_metrics[metric])
        results[metric] = (vals.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5))

    return results

def run_nested_cv(X_full: pd.DataFrame, y_full: pd.Series) -> Dict[str, Any]:
    """
    Executes a rigorous nested cross-validation pipeline for feature selection and modeling.

    This function prevents data leakage by restricting feature correlation reduction, 
    Boruta feature selection, and hyperparameter tuning exclusively to the training
    splits of each outer cross-validation fold. 

    Args:
        X_full (pd.DataFrame): The standardized feature matrix of HRV metrics.
        y_full (pd.Series): The binary target variable (1 for PTSD, 0 for no PTSD).

    Returns:
        Dict[str, Any]: A dictionary aggregating pooled predictions, performance metrics,
        selected features, and SHAP values across all outer CV folds for downstream evaluation.
    """
    # Outer CV for generalized performance evaluation; Inner CV for hyperparameter tuning
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=RANDOM_STATE)

    results: Dict[str, Any] = {
        'rf': {'aucs': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'bal_acc': [], 'preds': [], 'probs': [], 'conf': []},
        'xgb': {'aucs': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'bal_acc': [], 'preds': [], 'probs': [], 'conf': []},
        'y_true': [], 'selected_features': [], 'shap_vals': [], 'X_test_sel': [], 'corr_drops': defaultdict(int)
    }

    has_missing = X_full.isna().any().any()

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_full, y_full), 1):
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

        logger.info(f"\n--- Fold {fold_idx} Sample Counts ---")
        logger.info(f"Total train: {len(y_train)}, class 0: {(y_train == 0).sum()}, class 1: {(y_train == 1).sum()}")
        logger.info(f"Total test: {len(y_test)}, class 0: {(y_test == 0).sum()}, class 1: {(y_test == 1).sum()}")

        # Ensure imputation happens strictly within the CV loop to prevent data leakage
        if has_missing:
            imputer = SimpleImputer(strategy='median')
            X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
            X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
        else:
            X_train_imp, X_test_imp = X_train.copy(), X_test.copy()

        # Drop highly correlated features to reduce multicollinearity before Boruta selection.
        # Spearman is used over Pearson to correctly capture non-linear monotonic relationships.
        corr_matrix = X_train_imp.corr(method='spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
        
        for f in to_drop:
            results['corr_drops'][f] += 1
            
        X_train_filt = X_train_imp.drop(columns=to_drop)
        X_test_filt = X_test_imp.drop(columns=to_drop)
        logger.info(f"\nFold {fold_idx}, dropped: {to_drop}")

        # Apply Boruta feature selection to capture all relevant features, 
        # ensuring comprehensive biomarker discovery prior to model training.
        rf_bor = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
        boruta = BorutaPy(rf_bor, n_estimators='auto', max_iter=50, alpha=0.1, random_state=RANDOM_STATE)
        boruta.fit(X_train_filt.values, y_train.values)
        
        feats = X_train_filt.columns[boruta.support_ | boruta.support_weak_]
        if len(feats) == 0:
            continue

        results['selected_features'].append(list(feats))
        
        logger.info(f"\n--- Fold {fold_idx} Boruta Feature Selection ---")
        logger.info(f"Number of selected features: {len(feats)}")
        for f in feats:
            logger.info(f"  - {f}")

        X_train_sel, X_test_sel = X_train_filt[feats], X_test_filt[feats]

        # Hyperparameter tuning executed exclusively on the isolated training subset
        xgb_grid = GridSearchCV(xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', random_state=RANDOM_STATE, n_jobs=-1),
                                XGB_PARAM_GRID, scoring='roc_auc', cv=inner_cv, n_jobs=-1)
        xgb_grid.fit(X_train_sel, y_train)

        rf_grid = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                               RF_PARAM_GRID, scoring='roc_auc', cv=inner_cv, n_jobs=-1)
        rf_grid.fit(X_train_sel, y_train)

        logger.info(f"\nFold {fold_idx} Best Parameters:")
        logger.info(f"  RF: {rf_grid.best_params_}")
        logger.info(f"  XGB: {xgb_grid.best_params_}")

        # Final evaluation on the unseen outer validation fold
        for model, name in zip([xgb_grid.best_estimator_, rf_grid.best_estimator_], ['xgb', 'rf']):
            y_prob = model.predict_proba(X_test_sel)[:, 1]
            y_pred = model.predict(X_test_sel)
            
            results[name]['aucs'].append(roc_auc_score(y_test, y_prob))
            results[name]['acc'].append(accuracy_score(y_test, y_pred))
            results[name]['prec'].append(precision_score(y_test, y_pred, zero_division=0))
            results[name]['rec'].append(recall_score(y_test, y_pred, zero_division=0))
            results[name]['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            results[name]['bal_acc'].append(balanced_accuracy_score(y_test, y_pred))
            results[name]['conf'].append(confusion_matrix(y_test, y_pred))
            results[name]['preds'].append(y_pred)
            results[name]['probs'].append(y_prob)

        results['y_true'].append(y_test.values)

        # Generate local interpretability (SHAP) arrays for the held-out test subset
        explainer = shap.TreeExplainer(xgb_grid.best_estimator_)
        results['shap_vals'].append(explainer.shap_values(X_test_sel))
        results['X_test_sel'].append(X_test_sel)
        
        logger.info(f"\nFold {fold_idx} SHAP computed for {len(feats)} features.\n")
        logger.info("=" * 50 + "\n")

    return results

def compute_statistics(
    results: Dict[str, Any], 
    data_tbi: pd.DataFrame, 
    X_full: pd.DataFrame, 
    y_full: pd.Series, 
    output_dir: Path
) -> None:
    """
    Computes rigorous statistical comparisons, including exact permutation tests and VIF scores.

    Aggregates nested CV results to provide holistic evaluations, such as the exact
    paired permutation test between model AUCs, unadjusted Mann-Whitney U testing on the 
    full sample space, and variance inflation factor computations for the final feature subset.

    Args:
        results (Dict[str, Any]): Dictionary of aggregated nested CV outputs.
        data_tbi (pd.DataFrame): The full, unmodified merged demographic/metric dataframe.
        X_full (pd.DataFrame): The full standardized feature matrix.
        y_full (pd.Series): The full target variable series.
        output_dir (Path): Path to output directory for saving tabular CSV statistics.
    """
    
    # Correlation Filter Stability Summary
    logger.info("\n=== Correlation Filter Drop Frequency Across Outer Folds ===")
    n_folds = len(results['y_true'])
    for feature, count in sorted(results['corr_drops'].items(), key=lambda x: x[1], reverse=True):
        logger.info(f"{feature}: dropped in {count} of {n_folds} folds")

    stable_dropped = [f for f, c in results['corr_drops'].items() if c == n_folds]
    unstable_features = [f for f, c in results['corr_drops'].items() if 0 < c < n_folds]
    
    logger.info("\nFeatures dropped in ALL folds (highly redundant):")
    logger.info(str(stable_dropped))
    logger.info("\nFeatures inconsistently dropped across folds (unstable correlation structure):")
    logger.info(str(unstable_features))

    # Model Metrics Across Folds
    logger.info("\n=== Model Metrics Across Folds ===")
    for model_name in ['rf', 'xgb']:
        display_name = 'RF' if model_name == 'rf' else 'XGB'
        logger.info(f"\n--- {display_name} ---")
        m = results[model_name]
        logger.info(f"AUC: {np.round(m['aucs'], 3)} | mean ± std: {np.mean(m['aucs']):.3f} ± {np.std(m['aucs']):.3f}")
        logger.info(f"Accuracy: {np.round(m['acc'], 3)} | mean ± std: {np.mean(m['acc']):.3f} ± {np.std(m['acc']):.3f}")
        logger.info(f"Precision: {np.round(m['prec'], 3)} | mean ± std: {np.mean(m['prec']):.3f} ± {np.std(m['prec']):.3f}")
        logger.info(f"Recall: {np.round(m['rec'], 3)} | mean ± std: {np.mean(m['rec']):.3f} ± {np.std(m['rec']):.3f}")
        logger.info(f"F1: {np.round(m['f1'], 3)} | mean ± std: {np.mean(m['f1']):.3f} ± {np.std(m['f1']):.3f}")
        logger.info(f"Balanced Accuracy: {np.round(m['bal_acc'], 3)} | mean ± std: {np.mean(m['bal_acc']):.3f} ± {np.std(m['bal_acc']):.3f}")

    # Perform an exact paired permutation test to rigorously evaluate if the AUC 
    # difference between XGBoost and Random Forest models is statistically significant.
    logger.info("\n=== Exact Paired Permutation Test (XGB vs RF AUC) ===")
    auc_diffs = np.array(results['xgb']['aucs']) - np.array(results['rf']['aucs'])
    obs_diff = np.mean(auc_diffs)
    logger.info(f"Observed Mean AUC Difference (XGB - RF): {obs_diff:.4f}")
    
    permuted_diffs = [np.mean(auc_diffs * np.array(signs)) for signs in itertools.product([1, -1], repeat=len(auc_diffs))]
    p_value_exact = np.mean(np.abs(permuted_diffs) >= np.abs(obs_diff))
    logger.info(f"Exact Two-Sided p-value ({2**len(auc_diffs)} permutations): {p_value_exact:.4f}\n")

    # Bootstrapping Metrics
    y_true_all = np.concatenate(results['y_true'])
    y_pred_rf_all = np.concatenate(results['rf']['preds'])
    y_pred_xgb_all = np.concatenate(results['xgb']['preds'])
    y_prob_rf_all = np.concatenate(results['rf']['probs'])
    y_prob_xgb_all = np.concatenate(results['xgb']['probs'])

    logger.info("\n--- Bootstrapped Metrics (95% CI) ---")
    for name, y_p_cls, y_p_prob in zip(['Random Forest', 'XGBoost'], [y_pred_rf_all, y_pred_xgb_all], [y_prob_rf_all, y_prob_xgb_all]):
        logger.info(f"\n{name}:")
        boot_res = bootstrap_metrics(y_true_all, y_p_cls, y_prob=y_p_prob)
        for metric, (mean_val, ci_lower, ci_upper) in boot_res.items():
            logger.info(f"  {metric}: {mean_val:.3f} (95% CI: {ci_lower:.3f} – {ci_upper:.3f})")

    # Full Dataset Stats (Mann-Whitney U)
    group1, group2 = X_full[y_full == y_full.unique()[0]], X_full[y_full == y_full.unique()[1]]
    comp = []
    for f in X_full.columns:
        u_stat, p_val = mannwhitneyu(group1[f], group2[f], alternative='two-sided')
        comp.append({
            'Feature': f, 
            f'Mean_{y_full.unique()[0]}': group1[f].mean(), 
            f'Mean_{y_full.unique()[1]}': group2[f].mean(),
            'U_stat': u_stat, 
            'p_val': p_val, 
            'r_rb': 1 - (2 * u_stat) / (len(group1) * len(group2))
        })
    
    comp_df = pd.DataFrame(comp)
    comp_df['p_adj'] = multipletests(comp_df['p_val'], method='fdr_bh')[1]
    comp_df.to_csv(output_dir / "full_dataset_all_features_comparison.csv", index=False)
    
    logger.info("\nGroup-wise feature comparison (Mann–Whitney U, FDR-corrected p-values & effect size r_rb, top 10 by p_adj):")
    logger.info(comp_df.sort_values('p_adj').head(10).to_string())

    # Feature Selection Stability
    flat_feats = [f for sublist in results['selected_features'] for f in sublist]
    logger.info("\nFeature selection frequencies across folds (Boruta):")
    logger.info(pd.Series(flat_feats).value_counts().to_string())

    # Pooled Confusion Matrices
    for pooled_matrix, name in zip([confusion_matrix(y_true_all, y_pred_rf_all), confusion_matrix(y_true_all, y_pred_xgb_all)], ['RF', 'XGB']):
        logger.info(f"\n=== {name} Pooled Confusion Matrix Summary ===")
        TP = np.diag(pooled_matrix)
        FP = pooled_matrix.sum(axis=0) - TP
        FN = pooled_matrix.sum(axis=1) - TP
        TN = pooled_matrix.sum() - (TP + FP + FN)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

        summary_df = pd.DataFrame({
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
            'Sensitivity (Recall)': sensitivity, 'Specificity': specificity,
            'Precision': precision, 'F1 Score': f1
        }, index=[f'Class {i}' for i in range(pooled_matrix.shape[0])]).round(3)
        logger.info("\n" + summary_df.to_string())

    # Variance Inflation Factor (VIF) evaluation on the finally selected variables
    # to quantify the severity of residual multicollinearity.
    important_features = list(set(feat for sublist in results['selected_features'] for feat in sublist))
    if important_features:
        X_reduced_scaled = pd.DataFrame(StandardScaler().fit_transform(data_tbi[important_features].dropna()), columns=important_features)
        vif_clean = pd.DataFrame({
            "feature": X_reduced_scaled.columns,
            "VIF": [variance_inflation_factor(X_reduced_scaled.values, i) for i in range(X_reduced_scaled.shape[1])]
        })
        logger.info("\nReduced Feature Set VIF:")
        logger.info(vif_clean.sort_values("VIF", ascending=False).to_string())
        vif_clean.to_csv(output_dir / "vif_multicollinearity_summary.csv", index=False)

    # Bootstrapped SHAP Confidence Intervals
    all_shap_arrays, all_X_test_dfs = [], []
    for vals, X_df in zip(results['shap_vals'], results['X_test_sel']):
        all_shap_arrays.append(pd.DataFrame(np.abs(vals), columns=X_df.columns, index=X_df.index))
        all_X_test_dfs.append(X_df)
        
    pooled_shap_df = pd.concat(all_shap_arrays, axis=0).fillna(0)
    all_feats = pooled_shap_df.columns.tolist()

    np.random.seed(42)
    boot_means = defaultdict(list)
    for _ in range(10000):
        sampled = pooled_shap_df.sample(n=len(pooled_shap_df), replace=True)
        for feat in all_feats:
            boot_means[feat].append(sampled[feat].mean())

    shap_summary = pd.DataFrame({
        'mean': {f: np.mean(v) for f, v in boot_means.items()},
        'lower': {f: np.percentile(v, 2.5) for f, v in boot_means.items()},
        'upper': {f: np.percentile(v, 97.5) for f, v in boot_means.items()}
    }).sort_values("mean", ascending=True)

    logger.info("\nBootstrapped SHAP Feature Importance (mean ± 95% CI, pooled test sets):\n")
    for feat in shap_summary.index:
        logger.info(f"{feat:<30}: {shap_summary.loc[feat, 'mean']:.4f} [{shap_summary.loc[feat, 'lower']:.4f}, {shap_summary.loc[feat, 'upper']:.4f}]")


def generate_plots(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Generates and saves global Interpretability and ROC visualizations.

    Aggregates local SHAP explanations across all test folds to generate global
    feature importance, violin, and dependence plots, ensuring valid global
    interpretability despite the nested cross-validation architecture.

    Args:
        results (Dict[str, Any]): Dictionary of aggregated nested CV outputs.
        output_dir (Path): Output directory where rendered JPEG files are saved.
    """
    y_true_all = np.concatenate(results['y_true'])
    
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 20, 'axes.linewidth': 3, 'grid.linewidth': 2
    })

    # --- ROC Curve ---
    plt.figure(figsize=(11, 12))
    for model_probs, color, label in [(results['xgb']['probs'], 'royalblue', 'XGBoost'), (results['rf']['probs'], 'darkorange', 'Random Forest')]:
        fpr, tpr, _ = roc_curve(y_true_all, np.concatenate(model_probs))
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc(fpr, tpr):.3f})', linewidth=4, color=color)

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, label='Random Guessing')
    plt.xlabel('False Positive Rate'), plt.ylabel('True Positive Rate'), plt.title('ROC Curve - Model Comparison')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / 'ROC_Curve_Comparison.jpg', format='jpg', dpi=600)
    plt.close()

    # --- SHAP Aligning & Pooling ---
    all_feats = sorted({feat for X_df in results['X_test_sel'] for feat in X_df.columns})
    shap_aligned, X_aligned = [], []
    
    for shap_fold, X_df_fold in zip(results['shap_vals'], results['X_test_sel']):
        shap_aligned.append(pd.DataFrame(shap_fold, columns=X_df_fold.columns).reindex(columns=all_feats, fill_value=0).values)
        X_aligned.append(X_df_fold.reindex(columns=all_feats, fill_value=0))

    pooled_shap, pooled_X = np.vstack(shap_aligned), pd.concat(X_aligned, axis=0)

    # --- SHAP Violin ---
    plt.figure(figsize=(20, 12))
    shap.plots.violin(pooled_shap, pooled_X, plot_type='violin', show=False, title='SHAP Summary Plot', plot_size=(15, 8))
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_violin_plot.jpg', format='jpg', dpi=600)
    plt.close()

    # --- SHAP Bar ---
    plt.figure(figsize=(12, 8))
    shap_df_bar = pd.DataFrame({'feature': pooled_X.columns, 'mean_abs_shap': np.mean(np.abs(pooled_shap), axis=0)}).sort_values('mean_abs_shap')
    sns.barplot(x='mean_abs_shap', y='feature', data=shap_df_bar, palette='viridis')
    plt.title('Global SHAP Feature Importance'), plt.xlabel('Mean Absolute SHAP Value'), plt.ylabel('')
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_bar_importance.jpg', format='jpg', dpi=600)
    plt.close()

    # --- SHAP Dependence ---
    top_features = shap_df_bar.sort_values("mean_abs_shap", ascending=False)["feature"].head(3).tolist()
    shap_expl = shap.Explanation(values=pooled_shap, data=pooled_X.values, feature_names=pooled_X.columns)

    for feat_x, feat_color in itertools.combinations(top_features, 2):
        shap.plots.scatter(shap_expl[:, feat_x], color=shap_expl[:, feat_color], dot_size=40, show=False)
        safe_x, safe_c = re.sub(r"[^\w\-_.]", "_", feat_x), re.sub(r"[^\w\-_.]", "_", feat_color)
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_dependence_{safe_x}_vs_{safe_c}.jpg", dpi=600, bbox_inches="tight")
        plt.close()

def main() -> None:
    """
    Parses command-line arguments and orchestrates the HRV machine learning pipeline.
    """
    parser = argparse.ArgumentParser(description="Run HRV Classification Pipeline.")
    parser.add_argument('-d', '--demo_file', type=str, required=True, help="Path to clinical demographics excel file.")
    parser.add_argument('-m', '--metrics_dir', type=str, required=True, help="Directory containing HRV metric CSVs.")
    parser.add_argument('-o', '--output_dir', type=str, required=True, help="Directory to save plots and stat results.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_full, y_full, data_tbi = load_and_prepare_data(Path(args.demo_file), Path(args.metrics_dir))
    results = run_nested_cv(X_full, y_full)
    compute_statistics(results, data_tbi, X_full, y_full, out_dir)
    generate_plots(results, out_dir)


if __name__ == "__main__":
    main()