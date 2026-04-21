from fraud_risk_platform.data import generate_synthetic_transactions
from fraud_risk_platform.model import train_model
from fraud_risk_platform.scoring import precision_at_k, score_transactions


def test_scoring_adds_risk_outputs():
    df = generate_synthetic_transactions(n_rows=500, seed=7)
    model, _ = train_model(df)
    scored = score_transactions(model, df, threshold=0.5)

    assert "fraud_probability" in scored.columns
    assert "risk_score" in scored.columns
    assert "alert" in scored.columns
    assert scored["fraud_probability"].between(0, 1).all()


def test_precision_at_k_returns_fraction():
    df = generate_synthetic_transactions(n_rows=500, seed=11)
    model, _ = train_model(df)
    scored = score_transactions(model, df, threshold=0.5)

    value = precision_at_k(scored, k=25)
    assert 0 <= value <= 1

