from __future__ import annotations

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlpm.config.settings import settings

FEATURE_COLUMNS = [
    "abs_gap_bps",
    "market_prob_centered",
    "model_prob_centered",
]


def train_baseline(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict[str, object]:
    if train_df.empty or valid_df.empty:
        raise ValueError("Training and validation datasets must be non-empty.")
    if "target" not in train_df.columns or "target" not in valid_df.columns:
        raise ValueError("Expected a 'target' column for supervised training.")

    mlflow.set_tracking_uri(settings().mlflow_tracking_uri)
    with mlflow.start_run(run_name="baseline_logistic_regression"):
        preprocess = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    FEATURE_COLUMNS,
                )
            ]
        )
        model = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("classifier", LogisticRegression(max_iter=500)),
            ]
        )
        model.fit(train_df[FEATURE_COLUMNS], train_df["target"])
        predictions = model.predict(valid_df[FEATURE_COLUMNS])
        report = classification_report(valid_df["target"], predictions, output_dict=True)

        mlflow.log_params({"model_type": "logistic_regression", "features": ",".join(FEATURE_COLUMNS)})
        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.sklearn.log_model(model, "model")
        return {"model": model, "report": report}
