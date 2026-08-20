import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = "data/transactions.csv"
MODEL_PATH = "model/risk_model.joblib"

FEATURES = [
    "amount",
    "hour",
    "distance_from_usual_km",
    "account_age_days",
    "transactions_24h",
    "failed_attempts",
    "is_international",
    "is_new_device",
]

TARGET = "high_risk"


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Dataset not found. Run generate_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURES)
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(classification_report(y_test, predictions))
    print(
        "ROC-AUC:",
        round(roc_auc_score(y_test, probabilities), 4)
    )

    os.makedirs("model", exist_ok=True)

    artifact = {
        "pipeline": pipeline,
        "features": FEATURES,
        "thresholds": {
            "medium": 0.35,
            "high": 0.70
        }
    }

    joblib.dump(artifact, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
