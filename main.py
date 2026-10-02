"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full (deliberately simple) pipeline:
    load config -> load data -> preprocess -> split -> train
    -> evaluate (train & test) -> save results
"""
import yaml

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data import load_data

from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    split_features_target,
    build_preprocessor,
    split_dev_test,
)

from src.model import build_model

from src.evaluate import (
    cross_validate_pipeline,
    cv_report,
    oof_classification_report,
    fairness_report,
)

from src.results import save_run


# Configuration (Loads all project settings from config.yaml)

def load_config(
    path: str = "config.yaml",
) -> dict:

    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():

    # Load configuration
    config = load_config()
    print(config["model"])

    # Load Data (Reads the original dataset)

    df_raw = load_data(
        config["data"]["path"]
    )


    # Clean Data (Applies data cleaning rules before splitting the dataset)

    df_clean = clean_dataset(
        df_raw,
        config["diagnostics"],
    )


    # Remove Duplicates (Removes exact duplicates and repeated IDs from training data)

    df_clean = drop_duplicate_rows(
        df_clean,
        config["diagnostics"].get(
            "id_column"
        ),
    )


    # Features (Separates model features, target and evaluation columns)

    mnar_sources = (
        config["preprocessing"].get(
            "mnar_indicator_sources",
            [],
        )
    )

    X, y, extras = split_features_target(
        df_clean,
        config["data"],
        mnar_sources,
    )


    # Development and Test Split (Sets the final test set aside before cross-validation)

    (
        X_dev,
        X_test,
        y_dev,
        y_test,
        extras_dev,
        extras_test,
    ) = split_dev_test(
        X,
        y,
        extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )


    # Pipeline (Combines preprocessing and model in one pipeline)

    pipeline = Pipeline([
        (
            "prep",
            build_preprocessor(
                config["preprocessing"]
            ),
        ),
        (
            "model",
            build_model(
                config["model"]
            ),
        ),
    ])


    # Cross-Validation (Creates stratified folds using the settings in config.yaml)

    cv_config = config["cv"]

    shuffle = cv_config.get(
        "shuffle",
        True,
    )

    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=(
            cv_config.get("random_state")
            if shuffle
            else None
        ),
    )

    scoring = cv_config.get(
        "scoring",
        "accuracy",
    )


    # Run Cross-Validation

    fold_scores, y_oof = cross_validate_pipeline(
        pipeline,
        X_dev,
        y_dev,
        cv,
        scoring,
        n_jobs=cv_config.get(
            "n_jobs",
            1,
        ),
    )


    # Cross-Validation Report

    report = cv_report(
        fold_scores,
        scoring,
    )


    # Classification Report

    report += (
        "\n\n"
        + oof_classification_report(
            y_dev,
            y_oof,
        )
    )


    # Fairness Report

    report += (
        "\n"
        + fairness_report(
            y_dev,
            y_oof,
            extras_dev,
            sensitive_attr=config["data"]["sensitive_attr"],
        )
    )


    # Final Model

    final_model = pipeline.fit(
        X_dev,
        y_dev,
    )

    refit = (
        f"Final model: "
        f"{config['model']['type']} "
        f"refit on all "
        f"{len(X_dev)} development rows."
    )

    print(refit)

    report += (
        "\n"
        + refit
        + "\n"
    )


    # Locked Test Set

    locked = (
        f"Locked test set: "
        f"{len(X_test)} rows set aside, "
        f"not evaluated. "
        f"Development set: "
        f"{len(X_dev)} rows."
    )

    print(locked)

    report += (
        "\n"
        + locked
        + "\n"
    )


    # Save Results

    results_dir = (
        config.get(
            "output",
            {},
        ).get(
            "results_dir",
            "results",
        )
    )

    path = save_run(
        results_dir,
        config,
        report,
    )

    print(
        f"Full results saved to {path}"
    )


if __name__ == "__main__":
    main()