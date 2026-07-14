"""
Shared Plotly theme and figure-saving utilities.

Import this module first in any chapter script — registering the "iml" template
as the default ensures all figures share consistent styling without per-call setup.
"""
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

# ── Color palette ──────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#2563EB",  # blue
    "secondary": "#7C3AED",  # purple
    "accent":    "#DB2777",  # pink
    "positive":  "#059669",  # green
    "negative":  "#DC2626",  # red
    "neutral":   "#6B7280",  # gray
    "palette": [
        "#2563EB", "#7C3AED", "#DB2777",
        "#059669", "#D97706", "#0891B2",
        "#9333EA", "#EA580C",
    ],
}

# ── Register custom template ───────────────────────────────────────────────────
def _build_template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family="Inter, Arial, sans-serif", size=14, color="#1F2937"),
        title=dict(font=dict(size=22, color="#111827"), x=0.5, xanchor="center"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        colorway=COLORS["palette"],
        xaxis=dict(
            gridcolor="#F3F4F6",
            linecolor="#D1D5DB",
            tickfont=dict(size=12),
            showgrid=True,
            zeroline=False,
            automargin=True,
        ),
        yaxis=dict(
            gridcolor="#F3F4F6",
            linecolor="#D1D5DB",
            tickfont=dict(size=12),
            showgrid=True,
            zeroline=False,
            automargin=True,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E5E7EB",
            borderwidth=1,
            font=dict(size=12),
        ),
        margin=dict(l=70, r=50, t=90, b=70),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#D1D5DB",
            font_size=13,
        ),
    )
    return t


pio.templates["iml"] = _build_template()
pio.templates.default = "iml"

# ── Figure saving ──────────────────────────────────────────────────────────────
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def save_figure(fig: go.Figure, chapter: str, name: str, show: bool = False) -> None:
    """Save a Plotly figure as interactive HTML and (if kaleido is available) PNG.

    HTML files are self-contained and openable in any browser.
    PNG files (1200×700 px, 2× scale) are suitable for slide embedding.
    """
    out_dir = FIGURES_DIR / chapter
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / f"{name}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"  saved → figures/{chapter}/{name}.html")

    png_path = out_dir / f"{name}.png"
    try:
        fig.write_image(str(png_path), width=1200, height=700, scale=2)
        print(f"  saved → figures/{chapter}/{name}.png")
    except Exception:
        print(f"  (PNG skipped — install kaleido for static export)")

    if show:
        fig.show()
