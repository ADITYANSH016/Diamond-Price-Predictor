"""
app.py
------
Entry-point for the Streamlit web application.
Run with:  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import streamlit as st

from data_generator import generate_diamonds
from model          import train, save_model, load_model, model_exists, load_data
from eda            import (fig_price_distribution, fig_carat_vs_price,
                             fig_avg_price_by_cut, fig_correlation_heatmap,
                             fig_clarity_violin, fig_color_boxplot,
                             fig_actual_vs_predicted, fig_residuals,
                             fig_feature_impact, fig_coefficients)
from utils          import (CUT_OPTIONS, COLOR_OPTIONS, CLARITY_OPTIONS,
                             CUT_MAP, COLOR_MAP, CLARITY_MAP,
                             FEATURE_COLS, FEATURE_LABELS,
                             build_input_row, price_band)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💎 Diamond Price Predictor | MLR",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  h1, h2, h3                 { font-family: 'Playfair Display', serif; }
  .stApp                     { background: #0b0c10; }

  .hero { text-align:center; padding: 1.5rem 0 0.5rem; }
  .hero h1 {
    font-family:'Playfair Display',serif; font-size:2.9rem; font-weight:700;
    background: linear-gradient(135deg,#c9d6ff,#e2e2e2,#a18cd1);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
  }
  .hero p { color:#8892a4; font-size:.95rem; font-weight:300; margin-top:.3rem; }

  .kpi-row { display:flex; gap:1rem; margin:1.2rem 0; }
  .kpi {
    flex:1; background:linear-gradient(145deg,#16171e,#1d1f2a);
    border:1px solid #2d2f42; border-radius:14px;
    padding:1.1rem 1.4rem; text-align:center;
  }
  .kpi-label { color:#8892a4; font-size:.7rem; text-transform:uppercase; letter-spacing:.1em; }
  .kpi-value { color:#c9d6ff; font-size:1.85rem; font-weight:600; margin-top:.2rem; }
  .kpi-sub   { color:#6b7280; font-size:.75rem; }

  .sec-title {
    font-family:'Playfair Display',serif; color:#e2e8f0; font-size:1.3rem;
    border-bottom:1px solid #2d2f42; padding-bottom:.35rem; margin:1.8rem 0 1rem;
  }
  .pred-box {
    background:linear-gradient(135deg,#1a1c2a,#222538);
    border:1px solid #4a4f7a; border-radius:18px;
    padding:2rem; text-align:center; margin-top:1rem;
  }
  .pred-price { font-family:'Playfair Display',serif; font-size:3.2rem; color:#c9d6ff; font-weight:700; }
  .pred-band  { color:#8892a4; font-size:1rem; margin-top:.4rem; }
  .pred-range { color:#c9d6ff; font-size:.9rem; margin-top:.3rem; }

  div[data-testid="stSidebar"] { background:#0f1018 !important; }
  .stTabs [data-baseweb="tab"] { color:#8892a4 !important; }
  .stTabs [aria-selected="true"] { color:#c9d6ff !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Data & Model  (cached so they only run once per session)
# ══════════════════════════════════════════════════════════════════════════════
DATA_CSV = os.path.join(os.path.dirname(__file__), "data", "diamonds.csv")

@st.cache_data(show_spinner="Generating dataset …")
def get_data():
    if os.path.isfile(DATA_CSV):
        df = pd.read_csv(DATA_CSV)
    else:
        df = generate_diamonds(n=5000)
        os.makedirs(os.path.dirname(DATA_CSV), exist_ok=True)
        df.to_csv(DATA_CSV, index=False)
    # encode ordinal columns
    df["cut_num"]     = df["cut"].map(CUT_MAP)
    df["color_num"]   = df["color"].map(COLOR_MAP)
    df["clarity_num"] = df["clarity"].map(CLARITY_MAP)
    return df

@st.cache_resource(show_spinner="Training model …")
def get_model(df):
    model, metrics, X_test, y_test, y_pred = train(df)
    return model, metrics, X_test, y_test, y_pred

df                                    = get_data()
model, metrics, X_test, y_test, y_pred = get_model(df)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 💎 Diamond Price Predictor")
    st.markdown("**Algorithm:** Multiple Linear Regression")
    st.markdown("**Dataset:** Kaggle Diamonds (synthetic mirror)")
    st.markdown("---")
    st.markdown("**Model Performance**")
    st.metric("R² (test)",   metrics["r2"])
    st.metric("R² (5-fold)", f"{metrics['cv_r2']} ± {metrics['cv_r2_std']}")
    st.metric("RMSE",        f"${metrics['rmse']:,}")
    st.metric("MAE",         f"${metrics['mae']:,}")
    st.markdown("---")
    st.markdown(f"""
**Regression Equation**
```
price =
  {metrics['intercept']}
  + β₁·carat
  + β₂·cut
  + β₃·color
  + β₄·clarity
  + β₅·depth
  + β₆·table
  + β₇·x  + β₈·y  + β₉·z
```
""")
    st.markdown("---")
    st.caption("Built with Streamlit · scikit-learn · pandas")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER  +  KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>💎 Diamond Price Predictor</h1>
  <p>Multiple Linear Regression &nbsp;·&nbsp; Kaggle Diamonds Dataset &nbsp;·&nbsp; Streamlit</p>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
kpi_data = [
    ("R² Score",     metrics["r2"],          "Test set accuracy"),
    ("RMSE",         f"${metrics['rmse']:,}", "Root Mean Sq. Error"),
    ("MAE",          f"${metrics['mae']:,}",  "Mean Abs. Error"),
    ("5-Fold R²",    metrics["cv_r2"],        "Cross-val accuracy"),
    ("Records",      f"{len(df):,}",          "Diamond samples"),
]
for col, (label, value, sub) in zip([k1, k2, k3, k4, k5], kpi_data):
    with col:
        st.markdown(f"""
        <div class="kpi">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮  Predict Price",
    "📊  EDA & Insights",
    "🧮  Model Details",
    "📋  Dataset",
])


# ─────────────────────────── TAB 1 · PREDICT ─────────────────────────────────
with tab1:
    st.markdown('<div class="sec-title">Configure Your Diamond</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        carat   = st.slider("⚖️ Carat Weight",      0.20, 4.00, 1.00, 0.01)
        cut     = st.selectbox("✂️ Cut Quality",     CUT_OPTIONS)
        color   = st.selectbox("🎨 Color Grade",     COLOR_OPTIONS,
                               help="D = Colorless (best)  →  J = Noticeable colour")
        clarity = st.selectbox("🔬 Clarity Grade",   CLARITY_OPTIONS,
                               help="IF = Internally Flawless  →  I1 = Included")
    with c2:
        depth = st.slider("📏 Depth %",   55.0, 70.0, 61.7, 0.1)
        table = st.slider("📐 Table %",   50.0, 70.0, 57.5, 0.1)
        x_dim = st.slider("↔️ Length x (mm)", 3.0, 10.5, 6.50, 0.01)
        y_dim = st.slider("↕️ Width  y (mm)", 3.0, 10.5, 6.50, 0.01)
        z_dim = st.slider("⬛ Depth  z (mm)", 2.0,  6.5, 4.00, 0.01)

    if st.button("✨  Predict Diamond Price", use_container_width=True, type="primary"):
        row  = build_input_row(carat, cut, color, clarity, depth, table, x_dim, y_dim, z_dim)
        pred = max(300.0, model.predict(row)[0])
        lo   = max(300.0, pred - metrics["rmse"])
        hi   = pred + metrics["rmse"]

        st.markdown(f"""
        <div class="pred-box">
          <div class="kpi-label">Estimated Market Price</div>
          <div class="pred-price">${pred:,.0f}</div>
          <div class="pred-band">{price_band(pred)}</div>
          <div class="pred-range">Confidence range: <b>${lo:,.0f}</b> – <b>${hi:,.0f}</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-title">Feature Impact Breakdown</div>', unsafe_allow_html=True)
        input_vals = [carat, CUT_MAP[cut], COLOR_MAP[color], CLARITY_MAP[clarity],
                      depth, table, x_dim, y_dim, z_dim]
        fig = fig_feature_impact(model, input_vals, FEATURE_COLS, FEATURE_LABELS)
        st.pyplot(fig)


# ─────────────────────────── TAB 2 · EDA ─────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("**Price Distribution**")
        st.pyplot(fig_price_distribution(df))
    with r1c2:
        st.markdown("**Carat vs Price  (coloured by Clarity)**")
        st.pyplot(fig_carat_vs_price(df))

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown("**Average Price by Cut Quality**")
        st.pyplot(fig_avg_price_by_cut(df))
    with r2c2:
        st.markdown("**Correlation Heatmap**")
        st.pyplot(fig_correlation_heatmap(df))

    st.markdown("**Price by Clarity Grade**")
    st.pyplot(fig_clarity_violin(df))

    st.markdown("**Price by Color Grade**")
    st.pyplot(fig_color_boxplot(df))


# ─────────────────────────── TAB 3 · MODEL ───────────────────────────────────
with tab3:
    st.markdown('<div class="sec-title">Model Evaluation</div>', unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Actual vs Predicted**")
        st.pyplot(fig_actual_vs_predicted(y_test, y_pred))
    with m2:
        st.markdown("**Residual Distribution**")
        st.pyplot(fig_residuals(y_test, y_pred))

    st.markdown('<div class="sec-title">Regression Coefficients</div>', unsafe_allow_html=True)
    st.pyplot(fig_coefficients(model, FEATURE_COLS, FEATURE_LABELS))

    coef_df = pd.DataFrame({
        "Feature":        FEATURE_COLS,
        "Label":          [FEATURE_LABELS[f] for f in FEATURE_COLS],
        "Coefficient (β)": model.coef_.round(4),
    }).sort_values("Coefficient (β)", ascending=False).reset_index(drop=True)

    st.dataframe(
        coef_df.style
            .format({"Coefficient (β)": "{:+.4f}"})
            .applymap(
                lambda v: "color:#6ee7b7" if isinstance(v, float) and v > 0
                          else ("color:#fca5a5" if isinstance(v, float) and v < 0 else ""),
                subset=["Coefficient (β)"]
            )
            .set_properties(**{
                "background-color": "#16171e",
                "color": "#c9d6ff",
                "border": "1px solid #2d2f42"
            }),
        use_container_width=True, height=340,
    )

    st.markdown(f"""
**Full OLS Equation:**
```
price = {metrics['intercept']}
      + {model.coef_[0]:+.4f} × carat
      + {model.coef_[1]:+.4f} × cut_grade
      + {model.coef_[2]:+.4f} × color_grade
      + {model.coef_[3]:+.4f} × clarity_grade
      + {model.coef_[4]:+.4f} × depth
      + {model.coef_[5]:+.4f} × table
      + {model.coef_[6]:+.4f} × x
      + {model.coef_[7]:+.4f} × y
      + {model.coef_[8]:+.4f} × z
```
    """)


# ─────────────────────────── TAB 4 · DATASET ─────────────────────────────────
with tab4:
    st.markdown('<div class="sec-title">Dataset Explorer</div>', unsafe_allow_html=True)

    fc1, fc2 = st.columns(2)
    with fc1:
        cut_filter = st.multiselect("Filter by Cut", list(CUT_MAP.keys()),
                                    default=list(CUT_MAP.keys()))
    with fc2:
        price_range = st.slider("Price Range ($)",
                                int(df.price.min()), int(df.price.max()),
                                (int(df.price.min()), int(df.price.max())))

    view_cols = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z", "price"]
    filtered  = df[
        df["cut"].isin(cut_filter) &
        df["price"].between(*price_range)
    ][view_cols]

    st.markdown(f"Showing **{len(filtered):,}** of **{len(df):,}** records")
    st.dataframe(filtered.reset_index(drop=True).head(300),
                 use_container_width=True, height=420)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️  Download Filtered Data (CSV)", csv,
                       "diamonds_filtered.csv", "text/csv",
                       use_container_width=True)
