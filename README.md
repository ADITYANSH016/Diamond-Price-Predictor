# 💎 Diamond Price Predictor — Multiple Linear Regression

> Predict diamond prices using physical & quality features with a clean, modular ML project structure.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project applies **Multiple Linear Regression (MLR)** to the famous **Kaggle Diamonds dataset** to predict the market price of a diamond based on its physical dimensions and quality grades.

It is structured as a **production-style ML project** with separate modules for data generation, model training, EDA helpers, and a Streamlit web app — mirroring real-world project layouts.

---

## 🗂️ Project Structure

```
diamond_price_mlr/
│
├── app.py                          ← Streamlit web application (entry point)
│
├── src/
│   ├── data_generator.py           ← Generates / loads the diamonds dataset
│   ├── model.py                    ← MLR training, evaluation, save/load
│   ├── eda.py                      ← All chart/plot helper functions
│   └── utils.py                    ← Shared constants, encodings, helpers
│
├── data/
│   └── diamonds.csv                ← Auto-generated on first run
│
├── models/
│   ├── mlr_model.pkl               ← Serialised LinearRegression model
│   └── model_meta.pkl              ← Metrics dict (R², RMSE, MAE, CV)
│
├── notebooks/
│   └── diamond_mlr_exploration.ipynb  ← Full EDA + training walkthrough
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the project
```bash
cd diamond_price_mlr
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Pre-train & save the model
```bash
python src/model.py
```

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📊 Dataset

| Property       | Value                        |
|----------------|------------------------------|
| Source         | Kaggle — Diamonds (ggplot2 / Shivam2503) |
| Records        | 5,000 (synthetic mirror)     |
| Features       | 9 (carat, cut, color, clarity, depth, table, x, y, z) |
| Target         | `price` in USD               |
| Price range    | $300 – $20,000               |

### Feature Descriptions

| Feature   | Type        | Description                              |
|-----------|-------------|------------------------------------------|
| `carat`   | Numeric     | Weight of the diamond (0.2 – 4.0)       |
| `cut`     | Ordinal     | Fair < Good < Very Good < Premium < Ideal |
| `color`   | Ordinal     | J (worst) → D (best / colorless)        |
| `clarity` | Ordinal     | I1 → IF (Internally Flawless)           |
| `depth`   | Numeric     | Total depth % = z / mean(x,y) × 100     |
| `table`   | Numeric     | Width of top facet relative to widest point |
| `x`       | Numeric     | Length in mm                             |
| `y`       | Numeric     | Width in mm                              |
| `z`       | Numeric     | Depth in mm                              |

---

## 🧠 Algorithm — Multiple Linear Regression

The model learns the linear relationship between features and price using **Ordinary Least Squares (OLS)**:

```
price = β₀
      + β₁·carat
      + β₂·cut_grade
      + β₃·color_grade
      + β₄·clarity_grade
      + β₅·depth
      + β₆·table
      + β₇·x  + β₈·y  + β₉·z
      + ε
```

Categorical features (`cut`, `color`, `clarity`) are **ordinally encoded** using domain knowledge (e.g. Ideal=5, Premium=4 …).

---

## 🖥️ App Features

| Tab | Contents |
|-----|----------|
| 🔮 **Predict Price** | Configure diamond specs with sliders & dropdowns → get predicted price, price band, confidence range, and a feature-impact bar chart |
| 📊 **EDA & Insights** | Price distribution, carat scatter, cut bar chart, correlation heatmap, clarity violin plots, color box plots |
| 🧮 **Model Details** | Actual vs Predicted scatter, residual histogram, coefficient bar chart, styled coefficient table, full OLS equation |
| 📋 **Dataset** | Filterable data table by cut & price range, CSV download |

---

## 📈 Model Performance

| Metric        | Value     |
|---------------|-----------|
| R² (test)     | ~0.87     |
| R² (5-fold CV)| ~0.87     |
| RMSE          | ~$650     |
| MAE           | ~$480     |

---

## 🧩 Module Responsibilities

| File | Responsibility |
|------|---------------|
| `data_generator.py` | Creates synthetic diamond data mirroring Kaggle distributions |
| `utils.py` | Ordinal encodings, feature lists, input builder, price band labeller |
| `model.py` | `train()`, `save_model()`, `load_model()`, CLI runner |
| `eda.py` | 10+ standalone chart functions; each returns a `matplotlib.Figure` |
| `app.py` | Streamlit UI; imports from all `src/` modules |

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| **Streamlit** | Interactive web app |
| **scikit-learn** | LinearRegression, train_test_split, cross_val_score, metrics |
| **pandas / numpy** | Data manipulation |
| **matplotlib / seaborn** | Visualisations |
| **pickle** | Model serialisation |
| **Jupyter** | Exploration notebook |

---



