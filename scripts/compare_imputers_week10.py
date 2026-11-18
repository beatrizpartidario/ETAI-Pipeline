"""Week 10 Part A1: paired CV comparison of imputers on COMPAS development rows.

Usage: python scripts/compare_imputers_week10.py
Optional: python scripts/compare_imputers_week10.py --config config.yaml
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (clean_dataset, drop_duplicate_rows, add_engineered_features,
                               split_features_target, split_dev_test)
from src.model import build_pipeline
from src.evaluate import build_scorers, cross_validate_pipeline, corrected_paired_test


def compare(config):
    if config["data"]["target"] != "two_year_recid":
        raise ValueError("The Week 10 imputer comparison is intended for COMPAS.")
    cleaned = clean_dataset(load_data(config["data"]["path"]), config["diagnostics"])
    cleaned = drop_duplicate_rows(cleaned, config["diagnostics"].get("id_column"))
    engineered = config.get("feature_engineering", {}).get("enabled", False)
    cleaned = add_engineered_features(cleaned, enabled=engineered)
    X, y, extras = split_features_target(cleaned, config["data"],
                                         config["preprocessing"].get("mnar_indicator_sources", []))
    X_dev, _, y_dev, _, _, _ = split_dev_test(X, y, extras, config["test_set"]["size"],
                                              config["test_set"]["random_state"])
    cv_config = config["cv"]
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=cv_config.get("shuffle", True),
                         random_state=cv_config.get("random_state", 42) if cv_config.get("shuffle", True) else None)
    prep_base = dict(config["preprocessing"])
    if engineered:
        prep_base["numeric_features"] = prep_base["numeric_features"] + config["feature_engineering"]["numeric_features"]
    scorer = build_scorers(["roc_auc"])
    scores = {}
    for imputer in ["median", "knn", "iterative", "random_forest"]:
        prep = {**prep_base, "imputation": {**prep_base.get("imputation", {}), "numeric_strategy": imputer}}
        pipeline = build_pipeline(prep, config["model"], config.get("feature_selection", {"enabled": False}))
        print(f"Evaluating {imputer} ...", flush=True)
        fold, _ = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scorer, "roc_auc", n_jobs=1)
        scores[imputer] = fold["validation"].to_numpy()
        print(f"  ROC-AUC: {scores[imputer].mean():.3f} ± {scores[imputer].std(ddof=1):.3f}", flush=True)
    rows = []
    for name, values in scores.items():
        vs_median = corrected_paired_test(values, scores["median"], test_train_ratio=1/(cv.get_n_splits()-1)) if name != "median" else None
        rows.append({"imputer": name, "mean_roc_auc": values.mean(), "std_roc_auc": values.std(ddof=1),
                     "diff_vs_median": vs_median["mean_diff"] if vs_median else 0,
                     "corrected_p_vs_median": vs_median["p_value"] if vs_median else None,
                     **{f"fold_{i}": v for i,v in enumerate(values, 1)}})
    table = pd.DataFrame(rows)
    path = Path(config.get("output", {}).get("results_dir", "results")) / "week10_imputer_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    print("\n", table[["imputer", "mean_roc_auc", "std_roc_auc", "diff_vs_median", "corrected_p_vs_median"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved comparison: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        compare(yaml.safe_load(f))