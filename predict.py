"""Predict one output row per input CSV row using a saved, already-fitted pipeline.

Examples:
    python main.py
    python predict.py data/compas_new_people.csv predictions.csv
"""
import argparse

import joblib
import pandas as pd
import yaml

from src.preprocessing import clean_dataset, add_engineered_features, split_features_target


def predict(df: pd.DataFrame, pipeline, config: dict) -> pd.DataFrame:
    """Returns the id, the probability of the positive class and the prediction (at 0.5) for every row."""
    id_column = config["diagnostics"]["id_column"]
    if id_column not in df.columns:
        raise ValueError(f"Missing ID column: {id_column}")
    clean = clean_dataset(df, config["diagnostics"])
    clean = add_engineered_features(clean, enabled=config.get("feature_engineering", {}).get("enabled", False))
    X, _, _ = split_features_target(clean, config["data"], config["preprocessing"].get("mnar_indicator_sources", []))
    expected = list(pipeline.feature_names_in_)
    missing = sorted(set(expected) - set(X.columns))
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = X.loc[:, expected]
    probability = pipeline.predict_proba(X)[:, 1]
    result = pd.DataFrame({"id": df[id_column].to_numpy(),
                           "probability": probability,
                           "prediction": (probability >= 0.5).astype(int)})
    assert len(result) == len(df), "Every input row must receive a prediction"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a CSV with the model saved by main.py.")
    parser.add_argument("input", help="CSV with the same columns as the training data (no target needed)")
    parser.add_argument("output", nargs="?", default="predictions.csv", help="where to write the predictions")
    parser.add_argument("--config", default="config.yaml", help="config used to train the saved model")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    pipeline = joblib.load(config.get("output", {}).get("model_path", "models/model.joblib"))
    predictions = predict(pd.read_csv(args.input), pipeline, config)
    predictions.to_csv(args.output, index=False)
    print(f"{len(predictions)} rows predicted -> {args.output}")