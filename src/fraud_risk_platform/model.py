import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_risk_platform.config import FEATURE_COLUMNS, TARGET_COLUMN


def build_model(model_type: str = "hist_gradient_boosting") -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer([("numeric", numeric_pipeline, FEATURE_COLUMNS)])

    if model_type == "logistic_regression":
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    elif model_type == "xgboost":
        classifier = _build_xgboost_classifier()
    else:
        classifier = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.06, random_state=42)

    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def train_model(df: pd.DataFrame, model_type: str = "hist_gradient_boosting") -> tuple[Pipeline, dict[str, float]]:
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURE_COLUMNS],
        df[TARGET_COLUMN],
        test_size=0.25,
        stratify=df[TARGET_COLUMN],
        random_state=42,
    )
    model = build_model(model_type=model_type)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = evaluate_probabilities(y_test, probabilities)
    metrics["model"] = model_type
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


def benchmark_models(df: pd.DataFrame) -> pd.DataFrame:
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURE_COLUMNS],
        df[TARGET_COLUMN],
        test_size=0.25,
        stratify=df[TARGET_COLUMN],
        random_state=42,
    )
    model_types = ["logistic_regression", "hist_gradient_boosting"]
    if _xgboost_available():
        model_types.append("xgboost")

    rows = []
    for model_type in model_types:
        model = build_model(model_type=model_type)
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        rows.append({"model": model_type, **evaluate_probabilities(y_test, probabilities)})
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False)


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return False
    return True


def _build_xgboost_classifier():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Install optional dependencies with `pip install -r requirements-optional.txt`.") from exc

    return XGBClassifier(
        n_estimators=220,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )
