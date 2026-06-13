---
title: AutoML Studio
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.45.1
app_file: main.py
pinned: false
---

# AutoML Studio

End-to-end machine learning pipeline. Upload a dataset, explore it, build and compare models, tune hyperparameters, evaluate with SHAP interpretability, and export the trained pipeline.

**Stack:** Streamlit · PyCaret · SHAP · Optuna · Plotly

## Pipeline Steps

1. **Define Problem** — describe the goal
2. **Load Dataset** — CSV, TSV, or XLSX
3. **Preview Data** — shape, types, statistics
4. **Exploratory Data Analysis** — distributions, correlations, boxplots, pairplot
5. **Data Cleaning** — missing values, duplicates, outliers (PyCaret)
6. **Preprocessing** — encoding, scaling, feature selection, train/test split (PyCaret)
7. **Model Training** — compare multiple models with cross-validation (PyCaret)
8. **Hyperparameter Tuning** — Bayesian (Optuna), Random, or Grid search
9. **Model Evaluation** — confusion matrix, ROC, classification report, residuals
10. **Model Comparison** — leaderboard table and chart
11. **Final Evaluation** — SHAP feature importance, holdout predictions
12. **Export** — download pipeline (.pkl), leaderboard, predictions
