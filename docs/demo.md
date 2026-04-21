# Demo Assets

The project includes generated dashboard assets for use in the README and portfolio site:

- `docs/assets/dashboard-screenshot.png`
- `docs/assets/fraud-risk-demo.gif`

Regenerate them after rerunning model scoring:

```bash
python scripts/render_demo_assets.py
```

The assets are based on the scored sample transactions in `data/processed/scored_transactions.csv`.
