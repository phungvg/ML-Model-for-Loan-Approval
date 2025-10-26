#!/usr/bin/env python3
"""
Lending Club – single-file trainer (leakage-safe, profit-aware)

What this script does:
- Load the LendingClub CSV/CSV.GZ.
- Normalize/clean target; treat "", "NA", "None" as missing.
- Drop obvious post-outcome leakage columns (pymnt*, total_rec*, out_prncp*, hardship*, settlement*, etc.)
- Drop ultra-sparse cols (>80% NaN) and super high-cardinality categoricals (>200 uniques).
- Build a Pipeline: numeric (median impute + RobustScaler), categorical (most_frequent + OneHotEncoder).
- 5-fold CV to select among DecisionTree, RandomForest (tuned), LogisticRegression.
- Prefer an *out-of-time split* if `issue_d` exists; otherwise use random stratified split.
- Train the winner, then scan thresholds to MAXIMIZE PROFIT (using constants below).
- Save artifacts to outputs/: confusion_matrix.png, pr_curve.png, metrics.json, threshold_scan.csv, feature_importances.csv (if available).
- Optional: probability calibration (`--calibrate`).

Author: Phung (Victoria) Vuong
"""

import argparse
import json
import os
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, precision_recall_curve, auc, accuracy_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV


# ----------------------------
# Configuration / constants
# ----------------------------
DEFAULT_DATA = "/Users/panda/Documents/School/UIC/Semester/Now/Fall 2025/CS_218/lending_club_ml/data/project1_lending_club_data.csv.gz"
DEFAULT_SAVE_DIR = "outputs"
RANDOM_STATE = 42

# Business constants (edit as needed)
AVG_LOAN_AMT = 15268.0       # average principal used when per-loan amount isn't available
FULLY_PAID_RETURN = 1.30     # fully-paid brings back 130% of principal
OPERATING_COST = 1000.0      # fixed ops cost per approved loan

# Target setup
TARGET_COL = "loan_status"
POS_LABEL = "Fully Paid"     # we predict P(Fully Paid)


# ------------------------------------------------------------------------------
# Loading / cleaning
# ------------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Read a CSV (optionally gzipped), normalize target, coerce empty strings to NaN."""
    compression = "infer" if path.endswith(".gz") else None
    df = pd.read_csv(path, compression=compression, low_memory=False)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {path}")
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    df = df.replace({"": np.nan, "NA": np.nan, "None": np.nan})
    print(f"[load] shape={df.shape}, columns={len(df.columns)}")
    return df


def basic_leakage_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that reveal outcomes after origination: payment/recovery/principal/hardship/etc.
    Also drop some very high-cardinality free-text columns and LC grade buckets to keep modeling clean.
    """
    leak_keywords = [
        "pymnt", "payment", "recover", "last_pymnt", "next_pymnt",
        "total_rec", "collection", "chargeoff", "charged_off", "settlement",
        "out_prncp", "out_prncp_inv", "hardship", "debt_settlement"
    ]
    explicit_drop = [
        # underwriting proxies / internal buckets (optional to drop)
        "grade", "sub_grade",
        # often post-origination pull; conservative to drop
        "last_credit_pull_d",
        # extreme-cardinality text (explodes OHE)
        "emp_title", "title", "desc"
    ]
    drop_cols = [c for c in explicit_drop if c in df.columns]
    for c in df.columns:
        if c == TARGET_COL:
            continue
        if any(k in c.lower() for k in leak_keywords):
            drop_cols.append(c)
    if drop_cols:
        uniq = list(dict.fromkeys(drop_cols))  # dedupe, preserve order
        print(f"[leakage] dropping {len(uniq)} cols (examples): {uniq[:8]}{' ...' if len(uniq)>8 else ''}")
        df = df.drop(columns=uniq, errors="ignore")
    else:
        print("[leakage] none detected by keyword scan")
    return df


def drop_very_missing(df: pd.DataFrame, thresh: float = 0.80) -> pd.DataFrame:
    """Drop columns with > thresh fraction missing."""
    frac = df.isna().mean()
    drop = frac[frac > thresh].index.tolist()
    if drop:
        print(f"[missing] dropping {len(drop)} ultra-sparse columns (> {int(thresh*100)}% NaN)")
        df = df.drop(columns=drop)
    else:
        print("[missing] no ultra-sparse columns to drop")
    return df


def drop_high_cardinality_cats(df: pd.DataFrame, max_unique: int = 200) -> pd.DataFrame:
    """Drop categorical columns with > max_unique distinct values (avoids massive OHE blow-up)."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    too_big = [c for c in cat_cols if c != TARGET_COL and df[c].nunique(dropna=True) > max_unique]
    if too_big:
        print(f"[cardinality] dropping {len(too_big)} high-cardinality categoricals (> {max_unique} uniques): "
              f"{too_big[:6]}{' ...' if len(too_big)>6 else ''}")
        df = df.drop(columns=too_big)
    return df


def split_features(df: pd.DataFrame) -> Tuple[list, list]:
    """Return (categorical_cols, numeric_cols) excluding the target."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for lst in (cat_cols, num_cols):
        if TARGET_COL in lst:
            lst.remove(TARGET_COL)
    return cat_cols, num_cols


# ------------------------------------------------------------------------------
# Preprocessing / models
# ------------------------------------------------------------------------------
def build_preprocessor(cat_cols: list, num_cols: list) -> ColumnTransformer:
    """Numeric: median impute + RobustScaler; Categorical: most_frequent + OHE (with rare-category handling if available)."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)
    except TypeError:
        print("[preproc] OneHotEncoder(min_frequency=...) not supported; using plain OHE.")
        ohe = OneHotEncoder(handle_unknown="ignore")
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe)
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    return pre


def make_models() -> Dict[str, Any]:
    """Three baselines; RF slightly tuned and class-weighted."""
    return {
        "dt": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=100,
            min_samples_split=200,
            ccp_alpha=0.001,
            random_state=RANDOM_STATE
        ),
        "rf": RandomForestClassifier(
            n_estimators=600,
            max_depth=16,
            min_samples_leaf=80,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "logreg": LogisticRegression(
            max_iter=2000, solver="saga", penalty="l2"
        )
    }


# ------------------------------------------------------------------------------
# Profit-aware thresholding
# ------------------------------------------------------------------------------
def profit_given_threshold(
    y_true_txt: np.ndarray,
    p_fp: np.ndarray,
    threshold: float,
    loan_amounts: Optional[np.ndarray] = None
) -> Dict[str, float]:
    approve = (p_fp >= threshold).astype(int)            # 1 = approve
    y_is_fp = (y_true_txt == POS_LABEL).astype(int)      # 1 = Fully Paid

    E = int(((approve == 1) & (y_is_fp == 1)).sum())     # approved & fully paid
    L = int(((approve == 1) & (y_is_fp == 0)).sum())     # approved & charged off
    M = int(((approve == 0) & (y_is_fp == 1)).sum())     # rejected but would have paid
    S = int(((approve == 0) & (y_is_fp == 0)).sum())     # rejected & would have defaulted

    if loan_amounts is None:
        # uniform loan sizing
        revenue = E * AVG_LOAN_AMT * FULLY_PAID_RETURN
        total_costs = (E + L) * (AVG_LOAN_AMT + OPERATING_COST)
    else:
        la = np.asarray(loan_amounts)
        revenue = float((approve * y_is_fp * la * FULLY_PAID_RETURN).sum())
        total_costs = float((approve * (la + OPERATING_COST)).sum())

    profit = revenue - total_costs
    acc = (E + S) / max(E + L + M + S, 1)
    return {
        "profit": float(profit),
        "approved": int(E + L),
        "rejected": int(M + S),
        "E": E, "L": L, "M": M, "S": S,
        "accuracy": float(acc)
    }


def choose_best_threshold(
    y_true_txt: np.ndarray,
    proba_fp: np.ndarray,
    loan_amounts: Optional[np.ndarray] = None
) -> Dict[str, float]:
    scan = np.linspace(0.0, 1.0, 101)  # 0.00..1.00
    best = None
    for t in scan:
        m = profit_given_threshold(y_true_txt, proba_fp, float(t), loan_amounts)
        if (best is None) or (m["profit"] > best["profit"]):
            best = {"threshold": float(t), **m}
    return best


def dump_threshold_scan(
    y_true_txt: np.ndarray,
    proba_fp: np.ndarray,
    loan_amounts: Optional[np.ndarray],
    save_dir: str
) -> None:
    rows = []
    for t in np.linspace(0.0, 1.0, 101):
        m = profit_given_threshold(y_true_txt, proba_fp, float(t), loan_amounts)
        rows.append({"threshold": float(t), **m})
    pd.DataFrame(rows).to_csv(os.path.join(save_dir, "threshold_scan.csv"), index=False)


# ------------------------------------------------------------------------------
# Utilities: feature names / importances
# ------------------------------------------------------------------------------
def get_feature_names(pre: ColumnTransformer) -> Optional[np.ndarray]:
    """Try to get full transformed feature names from the ColumnTransformer."""
    try:
        return pre.get_feature_names_out()
    except Exception:
        # attempt manual build (best effort)
        names = []
        for name, trans, cols in pre.transformers_:
            if name == 'remainder' and trans == 'drop':
                continue
            if hasattr(trans, 'get_feature_names_out'):
                try:
                    sub = trans.get_feature_names_out(cols)
                except Exception:
                    sub = np.array(cols, dtype=object)
            else:
                sub = np.array(cols, dtype=object)
            names.extend([f"{name}__{s}" for s in sub])
        return np.array(names, dtype=object) if names else None


def save_feature_importances(model: Pipeline, X_train: pd.DataFrame, save_dir: str) -> None:
    """If model provides importances/coeffs, save top features."""
    os.makedirs(save_dir, exist_ok=True)
    try:
        pre = model.named_steps["pre"]
        clf = model.named_steps["clf"]
        feat_names = get_feature_names(pre)
        if feat_names is None:
            feat_names = np.array([f"f{i}" for i in range(pre.transform(X_train.iloc[:1]).shape[1])])

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            # logistic regression: use magnitude of coefficients
            coef = clf.coef_.ravel()
            importances = np.abs(coef) / (np.abs(coef).sum() + 1e-12)
        else:
            print("[features] classifier has no importances/coefficients; skipping.")
            return

        df_imp = pd.DataFrame({"feature": feat_names, "importance": importances})
        df_imp = df_imp.sort_values("importance", ascending=False)
        df_imp.to_csv(os.path.join(save_dir, "feature_importances.csv"), index=False)
        print("[features] top 25:")
        print(df_imp.head(25).to_string(index=False))
    except Exception as e:
        print(f"[features] could not save feature importances: {e}")


# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------
def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, save_dir: str) -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)

    classes = list(model.named_steps["clf"].classes_)
    class_index = classes.index(POS_LABEL)
    proba = model.predict_proba(X_test)[:, class_index]

    y_true_bin = (y_test == POS_LABEL).astype(int)
    y_pred_default = (proba >= 0.5).astype(int)

    roc = roc_auc_score(y_true_bin, proba)
    prec, rec, _ = precision_recall_curve(y_true_bin, proba)
    pr_auc = auc(rec, prec)
    acc = accuracy_score(y_true_bin, y_pred_default)

    loan_amounts = X_test["loan_amnt"].values if "loan_amnt" in X_test.columns else None
    best = choose_best_threshold(y_test.values, proba, loan_amounts)
    y_pred_opt = (proba >= best["threshold"]).astype(int)
    cm = confusion_matrix(y_true_bin, y_pred_opt, labels=[0, 1])

    # Confusion matrix image
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Charged Off", "Fully Paid"])
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix @ threshold={best['threshold']:.2f}")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()

    # PR curve image
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall (Fully Paid)")
    plt.ylabel("Precision (Fully Paid)")
    plt.title(f"PR Curve (AUC={pr_auc:.3f})")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "pr_curve.png"), bbox_inches="tight")
    plt.close()

    # Threshold scan CSV
    dump_threshold_scan(y_test.values, proba, loan_amounts, save_dir)

    print("\n=== Test Metrics (threshold=0.50) ===")
    print(f"Accuracy: {acc:.3f} | ROC-AUC: {roc:.3f} | PR-AUC: {pr_auc:.3f}")
    print("\nClassification report (0=Charged Off, 1=Fully Paid):")
    print(classification_report(y_true_bin, y_pred_default, digits=3))

    print("\n=== Profit-Optimized Decision ===")
    print(f"Best threshold: {best['threshold']:.2f}")
    print(f"Profit: ${best['profit']:,.0f}")
    print(f"Approved: {best['approved']:,} | Rejected: {best['rejected']:,}")
    print(f"S={best['S']:,}, L={best['L']:,}, M={best['M']:,}, E={best['E']:,}")

    results = {
        "accuracy@0.5": float(acc),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "best_threshold": best["threshold"],
        "profit_at_best_threshold": best["profit"],
        "confusion_counts_at_best_threshold": {"S": best["S"], "L": best["L"], "M": best["M"], "E": best["E"]}
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Lending Club – single-file trainer")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to CSV/CSV.GZ")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR, help="Where to write artifacts")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction if random split is used")
    parser.add_argument("--cv-subsample", type=int, default=200000,
                        help="Rows to use for CV model selection (speed). Use 0 for all.")
    parser.add_argument("--time-cutoff", type=str, default="2017-12-31",
                        help="Cutoff date for out-of-time split if issue_d exists (format YYYY-MM-DD)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Wrap the chosen model with CalibratedClassifierCV (isotonic, cv=3)")
    args = parser.parse_args()

    print(f"[config] data={args.data}")
    df = load_data(args.data)

    # Restrict to two classes we model (Fully Paid vs Charged Off)
    before = len(df)
    df = df[df[TARGET_COL].isin([POS_LABEL, "Charged Off"])].copy()
    print(f"[filter] kept {len(df):,}/{before:,} rows with target in {{'Fully Paid','Charged Off'}}")

    # Parse issue_d for potential temporal split (do not use as a feature)
    if "issue_d" in df.columns:
        df["issue_d_parsed"] = pd.to_datetime(df["issue_d"], errors="coerce", format="%b-%Y")
    else:
        df["issue_d_parsed"] = pd.NaT

    # Clean columns
    df = basic_leakage_filter(df)
    df = drop_high_cardinality_cats(df, max_unique=200)
    df = drop_very_missing(df, thresh=0.80)

    # Build feature lists
    cat_cols, num_cols = split_features(df)
    print(f"[cols] categorical={len(cat_cols)}, numeric={len(num_cols)}")

    # Prefer temporal split if we have usable issue_d_parsed
    temporal = df["issue_d_parsed"].notna().any()
    if temporal:
        cutoff = pd.Timestamp(args.time_cutoff)
        train_mask = (df["issue_d_parsed"] <= cutoff)
        test_mask = (df["issue_d_parsed"] > cutoff)
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            print("[split] temporal split had empty side; falling back to random split")
            temporal = False

    # Separate features/target (drop helper col)
    X = df.drop(columns=[TARGET_COL, "issue_d_parsed"], errors="ignore")
    y = df[TARGET_COL]

    if temporal:
        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]
        print(f"[split:time] train={len(X_train):,} (≤ {args.time_cutoff}), test={len(X_test):,} (> {args.time_cutoff})")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE
        )
        print(f"[split:random] train={len(X_train):,}, test={len(X_test):,}")

    # Preprocessor and candidate models
    pre = build_preprocessor(cat_cols, num_cols)
    models = make_models()

    # Optional subsample for faster CV
    if args.cv_subsample and len(X_train) > args.cv_subsample:
        rs = np.random.RandomState(RANDOM_STATE)
        idx = rs.choice(len(X_train), size=args.cv_subsample, replace=False)
        X_cv, y_cv = X_train.iloc[idx], y_train.iloc[idx]
        print(f"[cv] subsample used for model selection: {len(X_cv):,} rows")
    else:
        X_cv, y_cv = X_train, y_train
        print("[cv] using full training set for model selection")

    # Model selection by CV ROC-AUC
    best_name, best_auc = None, -np.inf
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(pipe, X_cv, y_cv, scoring="roc_auc", cv=cv, n_jobs=-1)
        mean_auc = scores.mean()
        print(f"[CV] {name}: ROC-AUC={mean_auc:.4f}")
        if mean_auc > best_auc:
            best_auc, best_name = mean_auc, name

    print(f"\n[selected] {best_name} (CV ROC-AUC={best_auc:.4f})")

    # Final pipeline (optionally calibrated)
    base_clf = models[best_name]
    clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3) if args.calibrate else base_clf
    final_model = Pipeline([("pre", pre), ("clf", clf)])
    final_model.fit(X_train, y_train)

    # Save model artifact (optional)
    os.makedirs(args.save_dir, exist_ok=True)
    try:
        import joblib
        joblib.dump(final_model, os.path.join(args.save_dir, f"model_{best_name}{'_cal' if args.calibrate else ''}.joblib"))
        print(f"[save] model saved to {os.path.join(args.save_dir, f'model_{best_name}{'_cal' if args.calibrate else ''}.joblib')}")
    except Exception as e:
        print(f"[save] skipping joblib save (optional): {e}")

    # Evaluate & save plots/metrics
    _ = evaluate(final_model, X_test, y_test, save_dir=args.save_dir)

    # Feature importances (best-effort)
    save_feature_importances(final_model, X_train, save_dir=args.save_dir)

    print(f"\nDone. Artifacts in: {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main()
