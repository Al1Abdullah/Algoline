# Utility functions shared across routes

import json
import numpy as np
import pandas as pd


def safe_json(df, n=100):
    """Convert DataFrame rows to JSON-safe dicts. Handles NaN, inf, numpy scalars."""
    d = df.head(n).copy()
    cols = d.columns.tolist()
    records = []
    for _, row in d.iterrows():
        r = {}
        for c in cols:
            v = row[c]
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                r[c] = None
            elif pd.isna(v):
                r[c] = None
            elif isinstance(v, (np.integer,)):
                r[c] = int(v)
            elif isinstance(v, (np.floating,)):
                fv = float(v)
                r[c] = None if (np.isnan(fv) or np.isinf(fv)) else fv
            elif isinstance(v, np.bool_):
                r[c] = bool(v)
            elif isinstance(v, (str, int, float, bool)):
                r[c] = v
            else:
                r[c] = str(v)
        records.append(r)
    return records, cols


def infer_task(df, target):
    y = df[target].dropna()
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > max(12, int(len(y) * 0.05)):
        return "regression"
    return "classification"


def fig_json(fig, h=None):
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20), height=h,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        font=dict(family="Inter,sans-serif", size=12, color="#a3a3a3"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", zeroline=False)
    return json.loads(fig.to_json())


def safe_float(v, d=2):
    """Safe float for JSON — returns None for NaN/inf."""
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, d)
    except Exception:
        return None


def norm_lb(df):
    if "Model" not in df.columns:
        df = df.reset_index()
        if df.columns[0] != "Model":
            df = df.rename(columns={df.columns[0]: "Model"})
    return df


def get_exp(task):
    if task == "classification":
        from pycaret.classification import ClassificationExperiment
        return ClassificationExperiment()
    from pycaret.regression import RegressionExperiment
    return RegressionExperiment()
