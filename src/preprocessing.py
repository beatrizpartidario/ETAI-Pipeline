"""
Preprocessing -- deliberately minimal for week 2.

This is intentionally the weakest part of the pipeline:
    - missing values are simply dropped (no imputation strategy)
    - categorical columns are one-hot encoded with no thought given to unseen categories or cardinality
    - a single train/test split is used (no cross-validation)

You will replace this with something better in the coming weeks.

One thing that is NOT naive, on purpose: `sensitive_attr` (race) is kept out of the model's input features entirely. It's split alongside the data so it's still available afterwards -- not to train on, but to check whether the model treats different groups differently. See src/evaluate.py:fairness_report.
"""

"""

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(
    df: pd.DataFrame,
    target: str,
    sensitive_attr: str,
    drop_columns: list,
    test_size: float,
    random_state: int,
):
    # naive: just drop rows with any missing values
    df = df.dropna()

    y = df[target]

    # kept aside for fairness auditing after training -- never used as a model input
    extras = df[[sensitive_attr, "score_text"]].copy()

    columns_to_exclude = [target, sensitive_attr] + [
        c for c in drop_columns if c in df.columns
    ]
    X = df.drop(columns=columns_to_exclude)

    # naive: one-hot encode all non-numeric columns, no further thought
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, extras_test  

"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

from category_encoders import CountEncoder


# Categorialc Data Cleaning and Standardization
## For each categorical column defined in the config, removes extra spaces, converts temporarily to lowercase and maps known variants to one canonical value

def canonicalize_categories(
    df: pd.DataFrame,
    columns_and_maps: dict,
) -> pd.DataFrame:

    out = df.copy()

    for column, mapping in columns_and_maps.items():

        if column not in out.columns:
            continue

        normalized = (
            out[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        out[column] = (
            normalized
            .map(mapping)
            .fillna(out[column])
        )

    return out


# Numerical Data Cleaning and Standardization
## Checks numeric columns against validity rules from the config and turns impossible values into NaN

def flag_invalid_values(
    df: pd.DataFrame,
    validity_rules: dict,
) -> pd.DataFrame:

    out = df.copy()

    for column, rule in validity_rules.items():

        if column not in out.columns:
            continue

        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        )

        if "min" in rule:
            out.loc[
                out[column] < rule["min"],
                column,
            ] = np.nan

        if "max" in rule:
            out.loc[
                out[column] > rule["max"],
                column,
            ] = np.nan

    return out


# Cleaning Pipeline
## Applies placeholder cleaning, numeric conversion, validity checks, category normalization and redundant-column removal

def clean_dataset(
    df: pd.DataFrame,
    diagnostics_config: dict,
) -> pd.DataFrame:

    out = df.copy()

    placeholder_tokens = set(
        diagnostics_config.get(
            "placeholder_tokens",
            [],
        )
    )

    # Placeholder values -> NaN
    normalized_placeholders = {
        str(token).strip().lower()
        for token in placeholder_tokens
    }

    for column in out.columns:

        normalized = (
            out[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        placeholder_mask = normalized.isin(
            normalized_placeholders
        )

        out.loc[
            placeholder_mask,
            column,
        ] = np.nan

    # Numeric-looking columns stored as text
    for column in diagnostics_config.get(
        "numeric_text_columns",
        [],
    ):

        if column not in out.columns:
            continue

        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        )

    # Invalid values -> NaN
    out = flag_invalid_values(
        out,
        diagnostics_config.get(
            "validity_rules",
            {},
        ),
    )

    # Normalize categorical values
    out = canonicalize_categories(
        out,
        diagnostics_config.get(
            "canonical_categories",
            {},
        ),
    )

    # Redundant columns
    redundant_columns = diagnostics_config.get(
        "redundant_columns",
        [],
    )

    out = out.drop(
        columns=[
            column
            for column in redundant_columns
            if column in out.columns
        ]
    )

    return out


# Duplicate Rows
## Removes exact duplicates and repeated IDs

def drop_duplicate_rows(
    df: pd.DataFrame,
    id_column: str | None = None,
) -> pd.DataFrame:

    out = df.drop_duplicates()

    if id_column and id_column in out.columns:

        out = out.drop_duplicates(
            subset=id_column,
            keep="first",
        )

    return out

# Missingness Indicators
## Creates an extra binary column to preserve the information that a value was originally missing

def add_missingness_indicators(
    df: pd.DataFrame,
    indicator_sources: list,
) -> pd.DataFrame:

    out = df.copy()

    for column in indicator_sources:

        if column not in out.columns:
            continue

        out[f"{column}_was_missing"] = (
            out[column]
            .isna()
            .astype(int)
        )

    return out


# Features
## Separates the target, model features and extra columns used for evaluation but not for training

def split_features_target(
    df: pd.DataFrame,
    data_config: dict,
    mnar_indicator_sources: list,
):

    out = add_missingness_indicators(
        df,
        mnar_indicator_sources,
    )

    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]

    drop_columns = data_config.get(
        "drop_columns",
        [],
    )

    # Target
    y = (
        out[target]
        if target in out.columns
        else None
    )

    # Extra columns kept for evaluation
    extras_columns = [
        column
        for column in [
            sensitive_attr,
            "score_text",
        ]
        if column in out.columns
    ]

    extras = (
        out[extras_columns].copy()
        if extras_columns
        else None
    )

    # Columns excluded from model input
    columns_to_exclude = (
        set(drop_columns)
        | {
            target,
            sensitive_attr,
            "score_text",
        }
    )

    # Model features
    feature_columns = [
        column
        for column in out.columns
        if column not in columns_to_exclude
    ]

    X = out[feature_columns]

    return X, y, extras


# Encoders and Scalers
## Defines the preprocessing options that can be selected through the config

SCALERS = {

    "none": "passthrough",

    "standard": StandardScaler,

    "minmax": MinMaxScaler,

    "robust": RobustScaler,
}


ENCODERS = {

    "onehot": lambda seed: OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    ),

    "ordinal": lambda seed: OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    ),

    "count": lambda seed: CountEncoder(
        handle_unknown=0,
        handle_missing=0,
    ),

    "target": lambda seed: TargetEncoder(
        target_type="binary",
        cv=5,
        shuffle=True,
        random_state=seed,
    ),
}


# Preprocessing Function

def build_preprocessor(
    preprocessing_config: dict,
) -> ColumnTransformer:

    # Features
    numeric_features = preprocessing_config.get(
        "numeric_features",
        [],
    )

    categorical_features = preprocessing_config.get(
        "categorical_features",
        [],
    )

    indicator_sources = preprocessing_config.get(
        "mnar_indicator_sources",
        [],
    )

    # Imputation configuration
    imputation_config = preprocessing_config.get(
        "imputation",
        {},
    )

    # Encoder and scaler selected in config.yaml
    encoder_name = preprocessing_config["encoder"]
    scaler_name = preprocessing_config["scaler"]

    # Create scaler
    scaler_factory = SCALERS[scaler_name]

    scaler = (
        scaler_factory()
        if callable(scaler_factory)
        else scaler_factory
    )

    # Create encoder
    random_state = preprocessing_config.get(
        "random_state",
        42,
    )

    encoder = ENCODERS[encoder_name](
        random_state
    )

    # Numeric preprocessing
    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy=imputation_config.get(
                    "numeric_strategy",
                    "median",
                )
            ),
        ),
        (
            "scaler",
            scaler,
        ),
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy=imputation_config.get(
                    "categorical_strategy",
                    "most_frequent",
                )
            ),
        ),
        (
            "encoder",
            encoder,
        ),
    ])

    # Missingness indicator columns
    indicator_columns = [
        f"{column}_was_missing"
        for column in indicator_sources
    ]

    # Apply a different preprocessing pipeline to each group of columns
    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
        (
            "indicators",
            "passthrough",
            indicator_columns,
        ),
    ])


# Test Split

def split_dev_test(
    X,
    y,
    extras,
    test_size: float,
    random_state: int,
):

    (
        X_dev,
        X_test,
        y_dev,
        y_test,
        extras_dev,
        extras_test,
    ) = train_test_split(
        X,
        y,
        extras,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return (
        X_dev,
        X_test,
        y_dev,
        y_test,
        extras_dev,
        extras_test,
    )


