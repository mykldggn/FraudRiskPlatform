import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_risk_platform.config import SCORED_DATA_PATH

ASSET_DIR = ROOT / "docs" / "assets"
SCREENSHOT_PATH = ASSET_DIR / "dashboard-screenshot.png"
GIF_PATH = ASSET_DIR / "fraud-risk-demo.gif"


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    scored = pd.read_csv(SCORED_DATA_PATH)
    frames = [_render_frame(scored, threshold) for threshold in [0.35, 0.5, 0.7]]
    frames[1].save(SCREENSHOT_PATH)
    frames[0].save(GIF_PATH, save_all=True, append_images=frames[1:], duration=950, loop=0)
    print(f"Wrote {SCREENSHOT_PATH}")
    print(f"Wrote {GIF_PATH}")


def _render_frame(scored: pd.DataFrame, threshold: float) -> Image.Image:
    width, height = 1400, 900
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    title_font = _font(44, bold=True)
    heading_font = _font(26, bold=True)
    body_font = _font(22)
    small_font = _font(18)

    alerts = scored["fraud_probability"] >= threshold
    p100 = scored.sort_values("fraud_probability", ascending=False).head(100)["is_fraud"].mean()

    draw.text((54, 42), "Fraud Risk Monitoring Platform", fill="#111827", font=title_font)
    draw.text((56, 96), f"Analyst review dashboard - threshold {threshold:.2f}", fill="#475569", font=body_font)

    metrics = [
        ("Transactions", f"{len(scored):,}"),
        ("Alert rate", f"{alerts.mean():.1%}"),
        ("Avg risk", f"{scored['risk_score'].mean():.1f}"),
        ("P95 risk", f"{scored['risk_score'].quantile(0.95):.1f}"),
        ("Precision@100", f"{p100:.1%}"),
    ]
    for index, (label, value) in enumerate(metrics):
        x = 56 + index * 260
        _card(draw, x, 150, 230, 116)
        draw.text((x + 22, 174), label, fill="#64748b", font=small_font)
        draw.text((x + 22, 204), value, fill="#0f172a", font=heading_font)

    _card(draw, 56, 310, 590, 495)
    draw.text((86, 336), "Risk Score Distribution", fill="#111827", font=heading_font)
    bins = scored["risk_score"].round(-1).clip(0, 100).value_counts().sort_index()
    max_count = max(float(bins.max()), 1)
    for i, (score, count) in enumerate(bins.items()):
        if i > 10:
            break
        bar_height = int(340 * count / max_count)
        x = 96 + i * 45
        y = 745 - bar_height
        draw.rectangle((x, y, x + 28, 745), fill="#2563eb")
        draw.text((x - 4, 760), str(int(score)), fill="#64748b", font=small_font)

    _card(draw, 690, 310, 650, 495)
    draw.text((720, 336), "Top Flagged Transactions", fill="#111827", font=heading_font)
    draw.text((720, 384), "ID", fill="#64748b", font=small_font)
    draw.text((900, 384), "Risk", fill="#64748b", font=small_font)
    draw.text((1010, 384), "Amount", fill="#64748b", font=small_font)
    draw.text((1148, 384), "Reason", fill="#64748b", font=small_font)

    top = scored.sort_values("fraud_probability", ascending=False).head(7)
    for row_index, (_, row) in enumerate(top.iterrows()):
        y = 426 + row_index * 48
        draw.line((720, y - 14, 1300, y - 14), fill="#e2e8f0", width=1)
        draw.text((720, y), str(row["transaction_id"])[:15], fill="#0f172a", font=small_font)
        draw.text((900, y), f"{row['risk_score']:.1f}", fill="#dc2626", font=small_font)
        draw.text((1010, y), f"${row['amount']:,.0f}", fill="#0f172a", font=small_font)
        draw.text((1148, y), str(row["review_reason"])[:24], fill="#475569", font=small_font)

    return image


def _card(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, height: int) -> None:
    draw.rounded_rectangle((x, y, x + width, y + height), radius=12, fill="#ffffff", outline="#e2e8f0", width=2)


def _font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
