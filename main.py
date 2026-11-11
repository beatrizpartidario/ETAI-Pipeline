"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full (deliberately simple) pipeline:
    load config -> load data -> preprocess -> split -> train
    -> evaluate (train & test) -> save results
"""
"""
Entry point: `python main.py`.

load config -> load and clean data -> drop duplicate rows (training only) -> features/target
-> lock the test set -> cross-validate the pipeline on the development set
-> [if tuning is enabled: nested CV of the tuning procedure, then tuning on all development rows]
-> out-of-fold classification and fairness reports -> refit on all development rows -> save the report
"""

from datetime import datetime
import argparse
import os
import pandas as pd
import shutil
import tempfile

import optuna
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import clean_dataset, drop_duplicate_rows, add_engineered_features, split_features_target, split_dev_test
from src.model import build_pipeline
from src.evaluate import (build_scorers, cross_validate_pipeline, cv_report, oof_classification_report,
                          fairness_report)
from src.tuning import tune_pipeline, nested_cross_validate, tuning_report
from src.results import save_run, save_scores


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path="config.yaml"):
    config = load_config(config_path)
    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config["model"]["type"]

    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))

    # Week 6: deterministic features are calculated row by row, without learning dataset statistics.
    feature_enabled = config.get("feature_engineering", {}).get("enabled", False)
    df_clean = add_engineered_features(df_clean, enabled=feature_enabled)

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    # From here on, every decision uses the development set only; the test set stays locked.
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )
    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                         random_state=cv_config.get("random_state") if shuffle else None)
    metrics = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]
    scorers = build_scorers(metrics)
    n_jobs = cv_config.get("n_jobs", 1)
    print(f"Model: {model_type}   |   selection metric: {selection_metric}   |   "
          f"{cv_config['n_splits']}-fold CV")
    print(f"Data:  {len(X_dev)} development rows   |   {len(X_test)} test rows locked away (not used)")

    prep_config = dict(config["preprocessing"])
    if feature_enabled:
        prep_config["numeric_features"] = (prep_config["numeric_features"]
                                            + config["feature_engineering"]["numeric_features"])
    selection_config = config.get("feature_selection", {"enabled": False})
    cache_dir = tempfile.mkdtemp(prefix="pipeline_cache_") if selection_config.get("enabled") else None
    pipeline = build_pipeline(prep_config, config["model"], selection_config, memory=cache_dir)
    floor_pipeline = build_pipeline(prep_config, {"type": "dummy"})
    floor_scores, _ = cross_validate_pipeline(floor_pipeline, X_dev, y_dev, cv,
                                              {selection_metric: scorers[selection_metric]}, selection_metric, n_jobs)
    fold_scores, oof_proba = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scorers, selection_metric, n_jobs)
    floor, untuned = floor_scores["validation"].mean(), fold_scores["validation"].mean()

    print(f"\nDefault hyperparameters ({config_path})")
    print(f"  {selection_metric}: {untuned:.3f} ± {fold_scores['validation'].std(ddof=1):.3f}   "
          f"train-validation gap: {fold_scores['gap'].mean():+.3f}   no-model floor: {floor:.3f}")
    report = (f"Hyperparameters from {config_path}: {config['model'].get('params')}\n\n"
              + cv_report(fold_scores, metrics, selection_metric, "Cross-validation")
              + f"\n\nNo-model floor (dummy, same folds): {selection_metric} = {floor:.3f}")
    final_scores = fold_scores

    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)
    if tuning_enabled:
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(f"tuning is enabled but config.yaml has no tuning.search_spaces for "
                             f"'{model_type}'. Add one, or set tuning.enabled: false.")
        search_space = dict(search_spaces[model_type])
        if selection_config.get("enabled"):
            search_space.update(selection_config.get("search_space", {}))
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed)

        # Honest estimate on the same outer folds; its out-of-fold probabilities feed the reports.
        print(f"\nTuning ({n_trials} trials, nested CV) ... this takes a while", flush=True)
        final_scores, oof_proba = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scorers, selection_metric,
            search_space, n_trials, tuning_seed, n_jobs=n_jobs,
        )
        print(f"  {selection_metric}: {final_scores['validation'].mean():.3f} ± "
              f"{final_scores['validation'].std(ddof=1):.3f}   (honest estimate of the tuned model)")
        report += "\n\n" + cv_report(final_scores, metrics, selection_metric,
                                     "Nested cross-validation (the tuning procedure, estimated honestly)")

        # The hyperparameters that are kept: the same procedure, once, on all development rows.
        pipeline, study = tune_pipeline(pipeline, X_dev, y_dev, inner_cv, scorers[selection_metric],
                                        search_space, n_trials, tuning_seed, n_jobs=n_jobs)
        print("  Chosen hyperparameters:")
        for name, value in study.best_params.items():
            print(f"    {name.replace('model__', ''):<20s} {value:.4g}" if isinstance(value, float)
                  else f"    {name.replace('model__', ''):<20s} {value}")
        report += "\n\n" + tuning_report(study, final_scores, selection_metric)

    print(f"\nAll metrics ({'tuned model' if tuning_enabled else 'default hyperparameters'}, mean ± std)")
    for name in metrics:
        print(f"  {name:<18s} {final_scores[name].mean():.3f} ± {final_scores[name].std(ddof=1):.3f}")
    report += "\n\n" + oof_classification_report(y_dev, oof_proba)
    report += "\n" + fairness_report(y_dev, oof_proba, extras_dev, sensitive_attr=config["data"]["sensitive_attr"])

    pipeline.fit(X_dev, y_dev)
    # Option A: score all label-free rows with the SAME fitted preprocessing/model.
    prediction_config = config.get("prediction", {})
    if prediction_config.get("path"):
        new_raw = load_data(prediction_config["path"])
        new_clean = clean_dataset(new_raw, config["diagnostics"])
        new_clean = add_engineered_features(new_clean, enabled=feature_enabled)
        id_col = config["diagnostics"].get("id_column", "id")
        if id_col not in new_clean.columns:
            raise ValueError(f"Prediction input is missing ID column {id_col!r}")
        if config["data"]["target"] in new_clean.columns:
            raise ValueError("Prediction input must not contain the target column")
        X_new, new_y, _ = split_features_target(new_clean, config["data"], mnar_sources)
        # Fail on unexpected columns instead of silently dropping them.
        if set(X_new.columns) != set(X_dev.columns):
            raise ValueError(f"Prediction feature mismatch: missing={set(X_dev.columns)-set(X_new.columns)}, "
                             f"extra={set(X_new.columns)-set(X_dev.columns)}")
        X_new = X_new.loc[:, X_dev.columns]
        proba = pipeline.predict_proba(X_new)[:, 1]
        prediction_threshold = prediction_config.get("threshold", 0.5)
        output_path = prediction_config.get("output", "results/adult_predictions.csv")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        predictions = pd.DataFrame({id_col: new_clean[id_col].to_numpy(),
                                    "probability": proba,
                                    "prediction": (proba >= prediction_threshold).astype(int)})
        assert len(predictions) == len(new_raw), "Predictions must preserve every input row"
        predictions.to_csv(output_path, index=False)
        print(f"\nPredictions: {len(predictions)} rows written to {output_path}")
        report += f"\nPredictions: {len(predictions)} rows written to {output_path}\n"

    if selection_config.get("enabled"):
        names = pipeline.named_steps["prep"].get_feature_names_out()
        mask = pipeline.named_steps["select"].get_support()
        kept = list(names[mask])
        feature_summary = f"Features kept: {len(kept)}/{len(names)}: {kept}"
        print("\n" + feature_summary)
        report += "\n" + feature_summary + "\n"
    refit = f"Final model: {model_type} refit on all {len(X_dev)} development rows."
    report += "\n" + refit + "\n"

    row = {"run": run, "data": config["data"]["path"], "model": model_type, "tuned": tuning_enabled,
           "encoder": config["preprocessing"]["encoder"], "scaler": config["preprocessing"]["scaler"],
           "engineered_features": feature_enabled,
           "feature_selection": selection_config.get("enabled", False),
           "selection_min_votes": pipeline.named_steps["select"].min_votes if selection_config.get("enabled") else "",
           "params": str(config["model"].get("params")), "tuned_params": str(study.best_params) if tuning_enabled else "",
           "metric": selection_metric, "mean": final_scores["validation"].mean(),
           "std": final_scores["validation"].std(ddof=1), "untuned": untuned, "floor": floor,
           **{f"fold_{i}": score for i, score in enumerate(final_scores["validation"], start=1)}}
    runs = save_scores(config.get("output", {}).get("scoreboard", "results/runs.csv"), row)
    same = runs[(runs["data"] == row["data"]) & (runs["metric"] == selection_metric)]
    board = (same.sort_values("mean", ascending=False)
             [["run", "model", "tuned", "encoder", "scaler", "mean", "std", "untuned", "floor"]]
             .rename(columns={"mean": selection_metric})
             .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nScoreboard (runs on the same data and metric; best first)\n{board}")
    report += f"\nScoreboard:\n{board}\n"

    path = save_run(config.get("output", {}).get("results_dir", "results"), run, config, report)
    print(f"\n{refit}\nFull report (per-fold tables, tuning, confusion matrix, fairness): {path}")
    if cache_dir:
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ETAI pipeline")
    parser.add_argument("config_path", nargs="?", default=None,
                        help="Optional config path, e.g. config_adult.yaml")
    parser.add_argument("--config", default=None, help="YAML configuration to run")
    args = parser.parse_args()
    main(args.config or args.config_path or "config.yaml")

