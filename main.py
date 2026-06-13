"""AutoML Studio — FastAPI Backend"""

import os, glob, warnings, tempfile, base64, json, io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

warnings.filterwarnings("ignore")

app = FastAPI(title="AutoML Studio")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Global state (single-user free tier) ──
S = {}

COLORS = ["#10b981","#6366f1","#f59e0b","#ef4444","#8b5cf6",
          "#06b6d4","#84cc16","#ec4899","#f97316","#14b8a6"]
px.defaults.color_discrete_sequence = COLORS


# ────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────

def safe_json(df, n=100):
    d = df.head(n).copy()
    for c in d.columns:
        try:
            if d[c].dtype == "object" or d[c].apply(type).nunique() > 1:
                d[c] = d[c].astype(str)
        except Exception:
            d[c] = d[c].astype(str)
    # Replace NaN with None for JSON
    d = d.where(pd.notnull(d), None)
    return d.to_dict(orient="records"), d.columns.tolist()

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

def get_exp(task):
    if task == "classification":
        from pycaret.classification import ClassificationExperiment
        return ClassificationExperiment()
    from pycaret.regression import RegressionExperiment
    return RegressionExperiment()

def grab_img(fn):
    """Read a PNG file and return base64 data URI, then delete it."""
    if not fn or not os.path.exists(fn):
        return None
    with open(fn, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    try: os.remove(fn)
    except Exception: pass
    return f"data:image/png;base64,{b64}"

def grab_plot(exp, model, ptype):
    before = set(glob.glob("*.png"))
    try: exp.plot_model(model, plot=ptype, save=True)
    except Exception: return None
    new = set(glob.glob("*.png")) - before
    return grab_img(max(new, key=os.path.getmtime)) if new else None

def grab_shap(exp, model, ptype="summary"):
    before = set(glob.glob("*.png"))
    try: exp.interpret_model(model, plot=ptype, save=True)
    except Exception: return None
    new = set(glob.glob("*.png")) - before
    return grab_img(max(new, key=os.path.getmtime)) if new else None

def gen_all_plots(exp, model, task):
    plots = []
    pm = ([("confusion_matrix","Confusion Matrix"),("auc","ROC / AUC"),
           ("pr","Precision-Recall"),("feature","Feature Importance"),
           ("class_report","Classification Report")]
          if task == "classification"
          else [("residuals","Residuals"),("error","Predicted vs Actual"),
                ("feature","Feature Importance")])
    for pk, lb in pm:
        img = grab_plot(exp, model, pk)
        if img: plots.append({"label": lb, "image": img})
    for stype, lb in [("summary","SHAP Summary"),("correlation","SHAP Dependence")]:
        img = grab_shap(exp, model, stype)
        if img: plots.append({"label": lb, "image": img})
    return plots

def norm_lb(df):
    if "Model" not in df.columns:
        df = df.reset_index()
        if df.columns[0] != "Model":
            df = df.rename(columns={df.columns[0]: "Model"})
    return df


# ────────────────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    try:
        name = (file.filename or "").lower()
        buf = io.BytesIO(file.file.read())
        if name.endswith(".tsv"):
            df = pd.read_csv(buf, sep="\t")
        elif name.endswith(".xlsx"):
            df = pd.read_excel(buf)
        else:
            df = pd.read_csv(buf)
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as e:
        return JSONResponse({"error": f"Could not read file: {e}"}, 400)

    if len(df) < 10:
        return JSONResponse({"error": "Dataset needs at least 10 rows."}, 400)

    S.clear()
    S["df"] = df
    S["target"] = df.columns[-1]
    S["task"] = infer_task(df, S["target"])

    nc = df.select_dtypes(include=np.number).columns.tolist()
    rows_data, cols_data = safe_json(df, 50)

    # Types info
    types_rows = [{"Column": c, "Type": str(df[c].dtype),
                   "Non-Null": int(df[c].notna().sum()),
                   "Unique": int(df[c].nunique())} for c in df.columns]

    # Stats
    desc = df.describe(include="all").round(3)
    desc.insert(0, "Stat", desc.index)
    stats_rows, stats_cols = safe_json(desc.reset_index(drop=True), 30)

    total_cells = len(df) * df.shape[1]
    miss = int(df.isna().sum().sum())

    return {
        "rows": len(df), "n_columns": df.shape[1],
        "missing": miss, "duplicates": int(df.duplicated().sum()),
        "missing_pct": round(miss / total_cells * 100, 1) if total_cells else 0,
        "columns": df.columns.tolist(),
        "numeric_columns": nc,
        "target": S["target"], "task": S["task"],
        "preview": {"rows": rows_data, "columns": cols_data},
        "types": {"rows": types_rows, "columns": ["Column","Type","Non-Null","Unique"]},
        "stats": {"rows": stats_rows, "columns": stats_cols},
    }


@app.post("/api/target")
def set_target(target: str = Form(...)):
    if "df" not in S:
        return JSONResponse({"error": "No data loaded"}, 400)
    S["target"] = target
    S["task"] = infer_task(S["df"], target)
    return {"target": target, "task": S["task"]}


# ── Explore endpoints ──

@app.post("/api/explore/distribution")
def explore_dist(feature: str = Form(...)):
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    fig = px.histogram(S["df"], x=feature, nbins=40, marginal="box",
                       color_discrete_sequence=["#10b981"])
    fig.update_layout(title=f"Distribution: {feature}", title_font_size=14)
    return {"figure": fig_json(fig)}

@app.post("/api/explore/counts")
def explore_counts():
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    df, tgt = S["df"], S["target"]
    cats = df.select_dtypes(exclude=np.number).columns.tolist()
    col = cats[0] if cats else tgt
    cts = df[col].astype(str).value_counts().head(20).reset_index()
    cts.columns = [col, "Count"]
    fig = px.bar(cts, x=col, y="Count", color_discrete_sequence=["#6366f1"])
    fig.update_layout(title=f"Value Counts: {col}", title_font_size=14)
    return {"figure": fig_json(fig)}

@app.post("/api/explore/correlation")
def explore_corr():
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    nc = S["df"].select_dtypes(include=np.number).columns.tolist()
    if len(nc) < 2: return {"figure": None}
    corr = S["df"][nc].corr().round(2)
    fig = ff.create_annotated_heatmap(z=corr.values, x=corr.columns.tolist(),
          y=corr.index.tolist(), colorscale="RdBu", showscale=True, zmin=-1, zmax=1)
    fig.update_layout(title="Correlation Matrix", title_font_size=14)
    return {"figure": fig_json(fig, min(640, 200 + 28 * len(corr)))}

@app.post("/api/explore/boxplot")
def explore_box(feature: str = Form(...)):
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    df, tgt = S["df"], S["target"]
    if df[tgt].nunique() <= 15:
        bd = df[[feature, tgt]].copy(); bd[tgt] = bd[tgt].astype(str)
        fig = px.box(bd, x=tgt, y=feature, color=tgt)
    else:
        fig = px.box(df, y=feature)
    fig.update_layout(title=f"Boxplot: {feature}", title_font_size=14)
    return {"figure": fig_json(fig)}

@app.post("/api/explore/scatter")
def explore_scatter():
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    df, tgt = S["df"], S["target"]
    nc = df.select_dtypes(include=np.number).columns.tolist()
    fn = [c for c in nc if c != tgt]
    if len(fn) < 2: return {"figure": None}
    n = min(5, len(fn))
    top = (df[fn].corrwith(df[tgt]).abs().sort_values(ascending=False).head(n).index.tolist()
           if tgt in nc else fn[:n])
    sample = df[top + [tgt]].dropna()
    if len(sample) > 500: sample = sample.sample(500, random_state=42)
    clr = tgt if df[tgt].nunique() <= 10 else None
    if clr and pd.api.types.is_numeric_dtype(sample[clr]):
        sample[clr] = sample[clr].astype(str)
    fig = px.scatter_matrix(sample, dimensions=top, color=clr)
    fig.update_layout(title=f"Top {n} features by target correlation", title_font_size=14)
    return {"figure": fig_json(fig, 520)}

@app.post("/api/explore/quality")
def explore_quality():
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    df, tgt = S["df"], S["target"]
    num = df.select_dtypes(include=np.number).drop(columns=[tgt], errors="ignore")
    rows = []
    for c in num.columns:
        s = num[c].dropna()
        if s.empty: continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        out = int(((s < q1-1.5*iqr)|(s > q3+1.5*iqr)).sum()) if iqr else 0
        miss = int(df[c].isna().sum())
        rows.append({"Feature":c, "Missing":miss, "Missing%":round(miss/len(df)*100,1),
                      "Unique":int(df[c].nunique()), "Mean":round(float(s.mean()),2),
                      "Std":round(float(s.std()),2), "Skew":round(float(s.skew()),2),
                      "Outliers":out})
    return {"rows": rows, "columns": ["Feature","Missing","Missing%","Unique","Mean","Std","Skew","Outliers"]}


# ── Build endpoints ──

@app.post("/api/train")
def train_models(
    task: str = Form("classification"),
    drop_duplicates: bool = Form(True),
    remove_outliers: bool = Form(False),
    normalize: bool = Form(True),
    transform_skew: bool = Form(False),
    feature_selection: bool = Form(False),
    drop_multicollinear: bool = Form(True),
    polynomial_features: bool = Form(False),
    fix_imbalance: bool = Form(False),
    test_size: float = Form(0.2),
    cv_folds: int = Form(5),
):
    if "df" not in S:
        return JSONResponse({"error": "Upload data first."}, 400)
    try:
        import pycaret
    except ImportError:
        return JSONResponse({"error": "PyCaret not installed."}, 500)

    df, tgt = S["df"], S["target"]
    S["task"] = task
    work = df.copy()
    removed = 0
    if drop_duplicates:
        n0 = len(work); work = work.drop_duplicates(); removed = n0 - len(work)

    # Validate
    nuniq = work[tgt].nunique(dropna=True)
    if nuniq < 2:
        return JSONResponse({"error": f"Target '{tgt}' has only {nuniq} unique value(s)."}, 400)

    # Setup
    try:
        exp = get_exp(task)
        kw = dict(data=work, target=tgt, session_id=42,
                  train_size=1-test_size, fold=cv_folds,
                  normalize=normalize, transformation=transform_skew,
                  remove_multicollinearity=drop_multicollinear,
                  multicollinearity_threshold=0.9,
                  feature_selection=feature_selection,
                  remove_outliers=remove_outliers, outliers_threshold=0.05,
                  polynomial_features=polynomial_features,
                  html=False, verbose=False)
        if task == "classification":
            kw["fix_imbalance"] = fix_imbalance
        exp.setup(**kw)
        S["experiment"] = exp
    except Exception as e:
        return JSONResponse({"error": f"Setup failed: {e}"}, 500)

    # Compare
    try:
        best = exp.compare_models()
        cdf = norm_lb(exp.pull())
        S.update({"compare_df": cdf, "best_model": best, "active_model": best})
        S.pop("tuned_model", None); S.pop("tune_df", None)
    except Exception as e:
        return JSONResponse({"error": f"Training failed: {e}"}, 500)

    # Plots
    S["plots"] = gen_all_plots(exp, best, task)

    # Predictions
    try: S["predictions"] = exp.predict_model(best)
    except Exception: pass

    lb_rows, lb_cols = safe_json(cdf, 30)
    metrics = [c for c in cdf.columns if c not in ("Model", "TT (Sec)")]

    # Comparison chart
    comp_fig = None
    if metrics and "Model" in cdf.columns:
        fig = px.bar(cdf, x="Model", y=metrics[0], color="Model")
        fig.update_layout(title=f"Model Comparison: {metrics[0]}", title_font_size=14)
        comp_fig = fig_json(fig)

    return {
        "model_name": type(best).__name__,
        "duplicates_removed": removed,
        "leaderboard": {"rows": lb_rows, "columns": lb_cols},
        "metrics": metrics,
        "comparison_chart": comp_fig,
        "plots": S.get("plots", []),
        "n_predictions": len(S.get("predictions", [])),
    }


@app.post("/api/compare")
def compare_metric(metric: str = Form(...)):
    if "compare_df" not in S: return JSONResponse({"error": "No leaderboard"}, 400)
    cdf = S["compare_df"]
    if "Model" not in cdf.columns or metric not in cdf.columns:
        return {"figure": None}
    fig = px.bar(cdf, x="Model", y=metric, color="Model")
    fig.update_layout(title=f"Model Comparison: {metric}", title_font_size=14)
    return {"figure": fig_json(fig)}


@app.post("/api/tune")
def tune_model(
    method: str = Form("Bayesian (Optuna)"),
    iterations: int = Form(10),
):
    if "best_model" not in S:
        return JSONResponse({"error": "Train models first."}, 400)
    exp, best, task = S["experiment"], S["best_model"], S["task"]
    methods = {"Bayesian (Optuna)": ("optuna", None),
               "Random Search": ("scikit-learn", "random"),
               "Grid Search": ("scikit-learn", "grid")}
    slib, salg = methods.get(method, ("optuna", None))

    try:
        kw = {"estimator": best, "search_library": slib,
              "n_iter": iterations, "choose_better": True}
        if salg: kw["search_algorithm"] = salg
        tuned = exp.tune_model(**kw)
        S.update({"tuned_model": tuned, "active_model": tuned, "tune_df": exp.pull()})
    except Exception as e:
        return JSONResponse({"error": f"Tuning failed: {e}"}, 500)

    S["plots"] = gen_all_plots(exp, tuned, task)
    try: S["predictions"] = exp.predict_model(tuned)
    except Exception: pass

    tune_rows, tune_cols = safe_json(S["tune_df"], 30)
    return {
        "model_name": type(tuned).__name__,
        "improved": tuned is not best,
        "tune_results": {"rows": tune_rows, "columns": tune_cols},
        "plots": S.get("plots", []),
    }


# ── Export endpoints ──

@app.get("/api/export/pipeline")
def export_pipeline():
    if "active_model" not in S:
        return JSONResponse({"error": "No model"}, 400)
    path = os.path.join(tempfile.gettempdir(), "automl_pipeline")
    try:
        final = S["experiment"].finalize_model(S["active_model"])
        S["experiment"].save_model(final, path)
        return FileResponse(f"{path}.pkl", filename="automl_pipeline.pkl",
                            media_type="application/octet-stream")
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

@app.get("/api/export/leaderboard")
def export_leaderboard():
    if "compare_df" not in S:
        return JSONResponse({"error": "No leaderboard"}, 400)
    path = os.path.join(tempfile.gettempdir(), "leaderboard.csv")
    S["compare_df"].to_csv(path, index=False)
    return FileResponse(path, filename="leaderboard.csv")

@app.get("/api/export/predictions")
def export_predictions():
    if "predictions" not in S:
        return JSONResponse({"error": "No predictions"}, 400)
    path = os.path.join(tempfile.gettempdir(), "predictions.csv")
    S["predictions"].to_csv(path, index=False)
    return FileResponse(path, filename="predictions.csv")

@app.get("/api/summary")
def get_summary():
    if "active_model" not in S:
        return {"ready": False}
    return {
        "ready": True,
        "target": S.get("target", "-"),
        "task": S.get("task", "-"),
        "model": type(S["active_model"]).__name__,
        "tuned": "tuned_model" in S,
        "n_models": len(S.get("compare_df", [])),
        "n_plots": len(S.get("plots", [])),
        "n_predictions": len(S.get("predictions", [])),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
