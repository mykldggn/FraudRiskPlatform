import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_risk_platform.config import FEATURE_COLUMNS, TARGET_COLUMN


def build_model() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer([("numeric", numeric_pipeline, FEATURE_COLUMNS)])
    classifier = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.06, random_state=42)
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def train_model(df: pd.DataFrame) -> tuple[Pipeline, dict[str, float]]:
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURE_COLUMNS],
        df[TARGET_COLUMN],
        test_size=0.25,
        stratify=df[TARGET_COLUMN],
        random_state=42,
    )
    model = build_model()
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = evaluate_probabilities(y_test, probabilities)
    return model, metrics


def evaluate_probabilities(y_true: pd.Series, probabilities) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    target_precision_idx = next((idx for idx, value in enumerate(precision) if value >= 0.65), len(precision) - 1)
    selected_threshold = thresholds[max(0, min(target_precision_idx, len(thresholds) - 1))] if len(thresholds) else 0.5

    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "recall_at_65_precision": float(recall[target_precision_idx]),
        "threshold_at_65_precision": float(selected_threshold),
    }


def save_model(model: Pipeline, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)

