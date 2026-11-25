"""Week 11: COMPAS anomaly features, same-fold ROC-AUC, and top-5% recidivism.

Run from the project folder:
    python scripts/compare_anomalies_week11.py
    python scripts/compare_anomalies_week11.py --config config.yaml

The CV comparison fits the full pipeline independently within each training fold.
The descriptive top-5% analysis fits an unsupervised detector on development rows
only, without touching the locked test set; it does not replace CV evaluation.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (AnomalyScore, add_engineered_features, build_preprocessor,
                               clean_dataset, drop_duplicate_rows, split_dev_test,
                               split_features_target)
from src.model import build_pipeline
from src.evaluate import build_scorers, cross_validate_pipeline, corrected_paired_test


def compare(config: dict):
    if config["data"]["target"] != "two_year_recid":
        raise ValueError("Week 11 anomaly comparison is for COMPAS.")
    df = clean_dataset(load_data(config["data"]["path"]), config["diagnostics"])
    df = drop_duplicate_rows(df, config["diagnostics"].get("id_column"))
    engineered = config.get("feature_engineering", {}).get("enabled", False)
    df = add_engineered_features(df, enabled=engineered)
    prep_cfg = dict(config["preprocessing"])
    if engineered:
        prep_cfg["numeric_features"] = (prep_cfg["numeric_features"]
                                        + config["feature_engineering"]["numeric_features"])
    X, y, extras = split_features_target(df, config["data"], prep_cfg.get("mnar_indicator_sources", []))
    X_dev, _, y_dev, _, _, _ = split_dev_test(X, y, extras, config["test_set"]["size"],
                                             config["test_set"]["random_state"])
    cv_cfg = config["cv"]
    cv = StratifiedKFold(n_splits=cv_cfg["n_splits"], shuffle=cv_cfg.get("shuffle", True),
                         random_state=cv_cfg.get("random_state", 42) if cv_cfg.get("shuffle", True) else None)
    anomaly_options = ["none", "isolation_forest", "lof"]
    score_by_method = {}
    rows = []
    base_anomaly = config.get("anomaly", {})
    for name in anomaly_options:
        anomaly_cfg = {**base_anomaly, "method": name}
        pipeline = build_pipeline(prep_cfg, config["model"], config.get("feature_selection", {}),
                                  anomaly_config=anomaly_cfg)
        print(f"Evaluating {name} ...", flush=True)
        folds, _ = cross_validate_pipeline(pipeline, X_dev, y_dev, cv,
                                           build_scorers(["roc_auc"]), "roc_auc", n_jobs=1)
        values = folds["validation"].to_numpy()
        score_by_method[name] = values
        print(f"  ROC-AUC: {values.mean():.3f} ± {values.std(ddof=1):.3f}", flush=True)

    for name in anomaly_options:
        values = score_by_method[name]
        test = (corrected_paired_test(values, score_by_method["none"],
                                      test_train_ratio=1 / (cv.get_n_splits() - 1))
                if name != "none" else None)
        rows.append({"anomaly": name, "mean_roc_auc": values.mean(),
                     "std_roc_auc": values.std(ddof=1),
                     "diff_vs_none": test["mean_diff"] if test else 0.0,
                     "corrected_p_vs_none": test["p_value"] if test else np.nan,
                     **{f"fold_{i}": float(v) for i, v in enumerate(values, 1)}})

    output_dir = Path(config.get("output", {}).get("results_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    path = output_dir / "week11_anomaly_comparison.csv"
    table.to_csv(path, index=False)
    print("\n" + table[["anomaly", "mean_roc_auc", "std_roc_auc", "diff_vs_none",
                         "corrected_p_vs_none"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved comparison: {path}")

    # Descriptive question from the notebook: top 5% most unusual people vs everyone else.
    # No labels are used in fitting the detector; no held-out test rows are used.
    Z = build_preprocessor(prep_cfg).fit_transform(X_dev, y_dev)
    scores_rows = []
    for name in ["isolation_forest", "lof"]:
        detector = AnomalyScore(method=name, n_neighbors=base_anomaly.get("n_neighbors", 20),
                                random_state=base_anomaly.get("random_state", 42))
        scores = np.asarray(detector.fit_transform(Z))[:, -1]
        n_top = max(1, int(np.ceil(0.05 * len(scores))))
        # Exactly ceil(5%) rows; threshold ties broken by row order.
        top = np.zeros(len(scores), dtype=bool)
        top[np.argsort(-scores, kind="stable")[:n_top]] = True
        targets = np.asarray(y_dev, dtype=int)
        scores_rows.append({"anomaly": name, "top_n": n_top,
                            "recidivism_top_5pct": float(targets[top].mean()),
                            "recidivism_rest_95pct": float(targets[~top].mean()),
                            "recidivism_overall": float(targets.mean())})
    rates = pd.DataFrame(scores_rows)
    rates_path = output_dir / "week11_top5_reoffending.csv"
    rates.to_csv(rates_path, index=False)
    print("\n" + rates.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved top-5% comparison: {rates_path}")
    print("Note: these top-5% rates are descriptive on development data; CV scores come from separate fold-safe fits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        compare(yaml.safe_load(f))
