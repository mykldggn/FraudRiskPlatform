import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from typing import Optional

from fraud_risk_platform.config import FEATURE_COLUMNS, TARGET_COLUMN


def global_feature_attribution(model, df: pd.DataFrame, max_rows: int = 500) -> pd.DataFrame:
    sample = df.sample(min(max_rows, len(df)), random_state=42) if len(df) > max_rows else df.copy()
    if TARGET_COLUMN in sample.columns and sample[TARGET_COLUMN].nunique() > 1:
        fallback = _permutation_attribution(model, sample)
    else:
        fallback = _local_probability_attribution(model, sample[FEATURE_COLUMNS].head(25))

    shap_values = _try_shap_attribution(model, sample[FEATURE_COLUMNS])
    if shap_values is not None:
        return shap_values
    return fallback


def explain_transaction(model, row: pd.Series) -> list[str]:
    baseline = pd.DataFrame([row[FEATURE_COLUMNS].to_dict()])
    base_probability = float(model.predict_proba(baseline)[:, 1][0])
    contributions = []

    for feature in FEATURE_COLUMNS:
        perturbed = baseline.copy()
        perturbed[feature] = 0
        probability = float(model.predict_proba(perturbed)[:, 1][0])
        contributions.append((feature, base_probability - probability))

    return [
        f"{feature} increased risk by {impact:.3f}"
        for feature, impact in sorted(contributions, key=lambda item: abs(item[1]), reverse=True)[:3]
    ]


def _try_shap_attribution(model, features: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        import shap
    except ImportError:
        return None

    try:
        transformed = model.named_steps["preprocess"].transform(features)
        classifier = model.named_steps["classifier"]
        explainer = shap.Explainer(classifier, transformed)
        values = explainer(transformed)
        raw_values = np.asarray(values.values)
        if raw_values.ndim == 3:
            raw_values = raw_values[:, :, -1]
        feature_names = _feature_names(model)
        importance = np.abs(raw_values).mean(axis=0)
        return _format_attribution(feature_names, importance, "shap")
    except Exception:
        return None


def _permutation_attribution(model, sample: pd.DataFrame) -> pd.DataFrame:
    result = permutation_importance(
        model,
        sample[FEATURE_COLUMNS],
        sample[TARGET_COLUMN],
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
    )
    return _format_attribution(FEATURE_COLUMNS, result.importances_mean, "permutation_importance")


def _local_probability_attribution(model, features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in features.iterrows():
        baseline = pd.DataFrame([row[FEATURE_COLUMNS].to_dict()])
        base_probability = float(model.predict_proba(baseline)[:, 1][0])
        for feature in FEATURE_COLUMNS:
            perturbed = baseline.copy()
            perturbed[feature] = 0
            probability = float(model.predict_proba(perturbed)[:, 1][0])
            rows.append({"feature": feature, "importance": abs(base_probability - probability)})
    result = pd.DataFrame(rows).groupby("feature", as_index=False)["importance"].mean()
    result["method"] = "probability_delta"
    return result.sort_values("importance", ascending=False)


def _feature_names(model) -> list[str]:
    try:
        names = model.named_steps["preprocess"].get_feature_names_out()
        return [name.replace("numeric__", "") for name in names]
    except Exception:
        return FEATURE_COLUMNS


def _format_attribution(feature_names, importance, method: str) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": list(feature_names), "importance": importance, "method": method})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
