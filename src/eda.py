"""
eda.py
------
All plotting / EDA helper functions.
Each function returns a matplotlib Figure so app.py can call st.pyplot(fig).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Dark theme defaults ────────────────────────────────────────────────────────
BG      = "#0b0c10"
PANEL   = "#16171e"
BORDER  = "#2d2f42"
TEXT    = "#c9d6ff"
MUTED   = "#8892a4"
ACCENT1 = "#a18cd1"   # purple
ACCENT2 = "#6ee7b7"   # green
ACCENT3 = "#fca5a5"   # red/pink
PALETTE = [ACCENT1, "#7b9fff", ACCENT2, "#fbbf24", ACCENT3, "#f472b6", "#38bdf8"]


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.spines[:].set_color(BORDER)
    ax.tick_params(colors=TEXT, labelsize=8)
    if title:  ax.set_title(title, color=TEXT, fontsize=10, pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def fig_price_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    fig.patch.set_facecolor(PANEL)
    ax.hist(df["price"], bins=70, color=ACCENT1, edgecolor="none", alpha=0.85)
    _style_ax(ax, "Price Distribution", "Price (USD)", "Count")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return fig


def fig_carat_vs_price(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    fig.patch.set_facecolor(PANEL)
    sc = ax.scatter(df["carat"], df["price"],
                    c=df["clarity_num"], cmap="cool",
                    alpha=0.35, s=7, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Clarity grade", color=MUTED, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=MUTED, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)
    _style_ax(ax, "Carat vs Price  (coloured by Clarity)", "Carat", "Price (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return fig


def fig_avg_price_by_cut(df: pd.DataFrame) -> plt.Figure:
    order    = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    avg_vals = df.groupby("cut")["price"].mean().reindex(order)
    fig, ax  = plt.subplots(figsize=(6, 3.6))
    fig.patch.set_facecolor(PANEL)
    colors   = sns.color_palette("cool", len(order))
    bars     = ax.bar(avg_vals.index, avg_vals.values,
                      color=colors, edgecolor="none", width=0.6)
    for bar, val in zip(bars, avg_vals.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 60,
                f"${val:,.0f}", ha="center", color=TEXT, fontsize=8)
    _style_ax(ax, "Average Price by Cut Quality", "Cut", "Avg Price (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return fig


def fig_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    num_cols = ["carat", "depth", "table", "x", "y", "z",
                "price", "cut_num", "color_num", "clarity_num"]
    corr     = df[num_cols].corr()
    fig, ax  = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(PANEL)
    sns.heatmap(corr, ax=ax, cmap="RdPu", annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.4, linecolor=PANEL,
                cbar_kws={"shrink": 0.75})
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_title("Feature Correlation Matrix", color=TEXT, fontsize=10, pad=8)
    fig.tight_layout()
    return fig


def fig_clarity_violin(df: pd.DataFrame) -> plt.Figure:
    order   = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
    palette = sns.color_palette("cool", len(order))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    fig.patch.set_facecolor(PANEL)
    sns.violinplot(data=df, x="clarity", y="price", order=order,
                   palette=palette, ax=ax, linewidth=0.7, inner="quartile")
    _style_ax(ax,
              "Price Distribution by Clarity  (I1 = Included  →  IF = Internally Flawless)",
              "Clarity Grade", "Price (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return fig


def fig_color_boxplot(df: pd.DataFrame) -> plt.Figure:
    order   = ["J", "I", "H", "G", "F", "E", "D"]
    palette = sns.color_palette("magma", len(order))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor(PANEL)
    sns.boxplot(data=df, x="color", y="price", order=order,
                palette=palette, ax=ax, linewidth=0.8,
                flierprops={"marker": "o", "markersize": 2,
                            "markerfacecolor": MUTED, "alpha": 0.3})
    _style_ax(ax, "Price Range by Color Grade  (J = Least Colorless  →  D = Colorless)",
              "Color Grade", "Price (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return fig


def fig_actual_vs_predicted(y_test, y_pred) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    fig.patch.set_facecolor(PANEL)
    ax.scatter(y_test, y_pred, alpha=0.3, s=7, color=ACCENT1, linewidths=0)
    lims = [min(float(y_test.min()), y_pred.min()),
            max(float(y_test.max()), y_pred.max())]
    ax.plot(lims, lims, "--", color=ACCENT3, linewidth=1.5, label="Perfect fit")
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
    _style_ax(ax, "Actual vs Predicted Price", "Actual Price ($)", "Predicted Price ($)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return fig


def fig_residuals(y_test, y_pred) -> plt.Figure:
    residuals = np.array(y_test) - y_pred
    fig, ax   = plt.subplots(figsize=(5.5, 5))
    fig.patch.set_facecolor(PANEL)
    ax.hist(residuals, bins=55, color=ACCENT2, edgecolor="none", alpha=0.85)
    ax.axvline(0, color=ACCENT3, linewidth=1.5, linestyle="--", label="Zero error")
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
    _style_ax(ax, "Residual Distribution", "Residual ($)", "Frequency")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return fig


def fig_feature_impact(model, input_values: list, feature_names: list,
                       feature_labels: dict) -> plt.Figure:
    """Bar chart of each feature's dollar contribution to the prediction."""
    impacts = np.array(model.coef_) * np.array(input_values)
    labels  = [feature_labels.get(f, f) for f in feature_names]
    order   = np.argsort(impacts)
    sorted_labels  = [labels[i]  for i in order]
    sorted_impacts = [impacts[i] for i in order]
    colors = ["#fca5a5" if v < 0 else "#6ee7b7" for v in sorted_impacts]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor(PANEL)
    ax.barh(sorted_labels, sorted_impacts, color=colors, edgecolor="none", height=0.55)
    ax.axvline(0, color=BORDER, linewidth=1)
    _style_ax(ax, "Feature Dollar Contribution to Predicted Price",
              "Impact on Price ($)", "")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return fig


def fig_coefficients(model, feature_names: list, feature_labels: dict) -> plt.Figure:
    coefs   = model.coef_
    labels  = [feature_labels.get(f, f) for f in feature_names]
    order   = np.argsort(coefs)
    sorted_labels = [labels[i] for i in order]
    sorted_coefs  = [coefs[i]  for i in order]
    colors = ["#fca5a5" if v < 0 else ACCENT1 for v in sorted_coefs]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor(PANEL)
    ax.barh(sorted_labels, sorted_coefs, color=colors, edgecolor="none", height=0.55)
    ax.axvline(0, color=BORDER, linewidth=1)
    _style_ax(ax, "Regression Coefficients (β)", "Coefficient Value", "")
    fig.tight_layout()
    return fig
