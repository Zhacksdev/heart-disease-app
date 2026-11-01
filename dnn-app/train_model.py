"""Train and evaluate a compact DNN on the Cleveland heart dataset.

The 80:20 seed-42 test split matches the companion LLM practical. Imputation,
one-hot encoding, and scaling are fitted only on the training split.
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "heart_disease.csv"
ARTIFACTS = ROOT / "artifacts"
SEED = 42
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def main() -> None:
    data = pd.read_csv(DATA_PATH, na_values="?")
    if list(data.columns) != FEATURES + ["num"]:
        raise ValueError(f"Kolom dataset harus {FEATURES + ['num']}")
    data["target"] = (pd.to_numeric(data["num"], errors="raise") > 0).astype(int)
    X, y = data[FEATURES], data["target"]
    train_idx, test_idx = train_test_split(
        data.index, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
        alpha=0.001, batch_size=32, learning_rate_init=0.001,
        max_iter=500, early_stopping=True, validation_fraction=0.2,
        n_iter_no_change=25, tol=1e-4, random_state=SEED,
    )
    model.fit(X_train_ready, y_train)
    predicted = model.predict(X_test_ready)
    probabilities = model.predict_proba(X_test_ready)[:, list(model.classes_).index(1)]
    metrics = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "test_size": 0.2,
        "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
        "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision_recall_f1_positive_class": [float(x) for x in precision_recall_fscore_support(y_test, predicted, average="binary", zero_division=0)[:3]],
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, predicted, labels=[0, 1], target_names=["No Heart Disease", "Heart Disease"], output_dict=True, zero_division=0),
        "false_negative_count": int(((y_test == 1) & (predicted == 0)).sum()),
        "threshold": 0.5, "early_stopping": bool(model.early_stopping),
        "best_validation_score": float(model.best_validation_score_),
        "epochs_trained": int(model.n_iter_),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(preprocessor, ARTIFACTS / "preprocessor.joblib")
    joblib.dump(model, ARTIFACTS / "dnn_model.joblib")
    # Test rows are saved for reproducible comparisons with the LLM practical.
    pd.DataFrame({"row_index": test_idx}).to_csv(ARTIFACTS / "test_indices_seed42.csv", index=False)
    pd.DataFrame({"actual": y_test, "prediction": predicted,
                  "probability_heart_disease": probabilities}, index=test_idx).to_csv(ARTIFACTS / "test_predictions.csv", index_label="row_index")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    info = {
        "model_name": "scikit-learn MLPClassifier DNN", "framework": "scikit-learn",
        "hidden_layers": [32, 16], "activation": "relu", "output": "binary probability via predict_proba",
        "loss": "binary cross-entropy (sklearn MLPClassifier log loss)", "optimizer": "adam",
        "features": FEATURES, "numeric_features": NUMERIC, "categorical_features": CATEGORICAL,
        "target_mapping": {"0": "No Heart Disease (num=0)", "1": "Heart Disease (num=1,2,3,4)"},
        "positive_class": 1, "threshold": 0.5, "random_seed": SEED,
        "preprocessing": "median numeric imputation + standard scaling; mode categorical imputation + one-hot encoding; fitted on training split only",
        "libraries": {"python": platform.python_version(), "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "numpy": np.__version__},
        "metrics_file": "metrics.json", "artifacts": ["dnn_model.joblib", "preprocessor.joblib"],
    }
    (ARTIFACTS / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("Dataset dimensions:", data.shape)
    print("Missing by column:", data[FEATURES + ["num"]].isna().sum().to_dict())
    print("Target distribution:", y.value_counts().sort_index().to_dict())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
