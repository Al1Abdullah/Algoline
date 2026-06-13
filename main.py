"""AutoML Studio — FastAPI Backend"""

import os, glob, warnings, tempfile, base64, json, io, gc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
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
    """Convert DataFrame rows to JSON-safe dicts.  Handles NaN, inf, numpy scalars."""
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
    """Generate a PyCaret plot and return as base64 data URI."""
    plot_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    try:
        os.chdir(plot_dir)
        exp.plot_model(model, plot=ptype, save=True)
        pngs = glob.glob(os.path.join(plot_dir, "*.png"))
        if pngs:
            return grab_img(max(pngs, key=os.path.getmtime))
        return None
    except Exception:
        return None
    finally:
        os.chdir(old_cwd)

def grab_shap(exp, model, ptype="summary"):
    """Generate a SHAP plot and return as base64 data URI."""
    plot_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    try:
        os.chdir(plot_dir)
        exp.interpret_model(model, plot=ptype, save=True)
        pngs = glob.glob(os.path.join(plot_dir, "*.png"))
        if pngs:
            return grab_img(max(pngs, key=os.path.getmtime))
        return None
    except Exception:
        return None
    finally:
        os.chdir(old_cwd)

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
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(buf)
        else:
            df = pd.read_csv(buf)
        # Clean column names: strip whitespace, replace problematic characters
        df.columns = [str(c).strip().replace('[', '(').replace(']', ')').replace('<', 'lt').replace('>', 'gt') for c in df.columns]
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
    # Clear stale training state when target changes
    for k in ["experiment","compare_df","best_model","active_model","tuned_model","tune_df","plots"]:
        S.pop(k, None)
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

def _sf(v, d=2):
    """Safe float for JSON — returns None for NaN/inf."""
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, d)
    except Exception:
        return None

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
                      "Unique":int(df[c].nunique()), "Mean":_sf(s.mean()),
                      "Std":_sf(s.std()), "Skew":_sf(s.skew()),
                      "Outliers":out})
    return {"rows": rows, "columns": ["Feature","Missing","Missing%","Unique","Mean","Std","Skew","Outliers"]}


@app.post("/api/explore/target")
def explore_target():
    """Target variable distribution — class balance (classification) or histogram (regression)."""
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    df, tgt, task = S["df"], S["target"], S["task"]
    y = df[tgt].dropna()
    if task == "classification":
        cts = y.astype(str).value_counts().reset_index()
        cts.columns = [tgt, "Count"]
        fig = px.bar(cts, x=tgt, y="Count", color=tgt, text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(title=f"Target: {tgt} — Class Balance", title_font_size=14,
                          showlegend=False)
    else:
        fig = px.histogram(y, x=tgt, nbins=40, marginal="box",
                           color_discrete_sequence=["#6366f1"])
        fig.update_layout(title=f"Target: {tgt} — Distribution", title_font_size=14)
    return {"figure": fig_json(fig)}


@app.post("/api/explore/missing")
def explore_missing():
    """Missing values bar chart — shows % missing per column."""
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    df = S["df"]
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        return {"figure": None, "message": "No missing values found"}
    pct = (miss / len(df) * 100).round(1)
    mdf = pd.DataFrame({"Column": miss.index, "Missing": miss.values, "Percent": pct.values})
    fig = px.bar(mdf, x="Column", y="Percent", text="Missing",
                 color="Percent", color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"])
    fig.update_traces(textposition="outside")
    fig.update_layout(title="Missing Values per Column", title_font_size=14,
                      yaxis_title="Missing %", coloraxis_showscale=False)
    return {"figure": fig_json(fig)}


@app.post("/api/explore/feature_target")
def explore_feature_target(feature: str = Form(...)):
    """Feature vs Target — violin (classification) or scatter (regression)."""
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    df, tgt, task = S["df"], S["target"], S["task"]
    sub = df[[feature, tgt]].dropna()
    if len(sub) > 2000: sub = sub.sample(2000, random_state=42)

    if task == "classification":
        sub[tgt] = sub[tgt].astype(str)
        fig = px.violin(sub, x=tgt, y=feature, color=tgt, box=True, points="outliers")
        fig.update_layout(title=f"{feature} by {tgt}", title_font_size=14, showlegend=False)
    else:
        fig = px.scatter(sub, x=feature, y=tgt, trendline="ols",
                         color_discrete_sequence=["#8b5cf6"], opacity=0.6)
        fig.update_layout(title=f"{feature} vs {tgt}", title_font_size=14)
    return {"figure": fig_json(fig)}


@app.post("/api/explore/outliers")
def explore_outliers():
    """All numeric features boxplot — outlier overview in one view."""
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    df, tgt = S["df"], S["target"]
    nc = df.select_dtypes(include=np.number).drop(columns=[tgt], errors="ignore").columns.tolist()
    if not nc: return {"figure": None}
    # Normalize for comparison (z-score)
    normed = df[nc].apply(lambda x: (x - x.mean()) / x.std() if x.std() else x)
    melted = normed.melt(var_name="Feature", value_name="Z-Score").dropna()
    fig = px.box(melted, x="Feature", y="Z-Score", color="Feature")
    fig.update_layout(title="Outlier Overview (Z-Score Normalized)", title_font_size=14,
                      showlegend=False, xaxis_tickangle=-45)
    return {"figure": fig_json(fig, 420)}


# ── Additional Explore Endpoints (completing all 18 ML plots) ──

@app.post("/api/explore/kde")
def explore_kde(feature: str = Form(...)):
    """KDE (Kernel Density Estimation) plot."""
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    try:
        s = S["df"][feature].dropna()
        if not pd.api.types.is_numeric_dtype(s):
            # For categorical: show value counts as bar
            vc = s.astype(str).value_counts().head(20).reset_index()
            vc.columns = [feature, "Count"]
            fig = px.bar(vc, x=feature, y="Count", color_discrete_sequence=["#10b981"])
            fig.update_layout(title=f"Value Distribution: {feature}", title_font_size=14)
            return {"figure": fig_json(fig)}
        from scipy import stats as _st
        kde = _st.gaussian_kde(s.astype(float))
        xs = np.linspace(float(s.min()), float(s.max()), 300)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs.tolist(), y=kde(xs).tolist(), mode="lines",
                                 fill="tozeroy", line=dict(color="#10b981", width=2),
                                 fillcolor="rgba(16,185,129,0.15)"))
        fig.update_layout(title=f"KDE: {feature}", title_font_size=14,
                          xaxis_title=feature, yaxis_title="Density")
        return {"figure": fig_json(fig)}
    except Exception as e:
        return JSONResponse({"error": f"KDE failed: {str(e)}"}, 400)


@app.post("/api/explore/violin")
def explore_violin(feature: str = Form(...)):
    """Violin plot of feature, grouped by target if classification."""
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    try:
        df, tgt, task = S["df"], S["target"], S["task"]
        if not pd.api.types.is_numeric_dtype(df[feature]):
            return JSONResponse({"error": f"Violin plot requires a numeric feature. '{feature}' is categorical."}, 400)
        sub = df[[feature, tgt]].dropna()
        if len(sub) > 3000: sub = sub.sample(3000, random_state=42)
        if task == "classification" and df[tgt].nunique() <= 15:
            sub[tgt] = sub[tgt].astype(str)
            fig = px.violin(sub, x=tgt, y=feature, color=tgt, box=True, points="outliers")
            fig.update_layout(showlegend=False)
        else:
            fig = px.violin(sub, y=feature, box=True, points="outliers",
                            color_discrete_sequence=["#8b5cf6"])
        fig.update_layout(title=f"Violin: {feature}", title_font_size=14)
        return {"figure": fig_json(fig)}
    except Exception as e:
        return JSONResponse({"error": f"Violin failed: {str(e)}"}, 400)


@app.post("/api/explore/missing_heatmap")
def explore_missing_heatmap():
    """Missing value heatmap — shows pattern of nulls across rows/columns."""
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    try:
        df = S["df"]
        miss_cols = [c for c in df.columns if df[c].isna().any()]
        if not miss_cols:
            return {"figure": None, "message": "No missing values found in the dataset"}
        sample = df[miss_cols]
        if len(sample) > 200: sample = sample.head(200)
        mat = sample.isna().astype(int).values.tolist()
        fig = go.Figure(data=go.Heatmap(
            z=mat, x=miss_cols, y=list(range(len(sample))),
            colorscale=[[0, "#18181b"], [1, "#ef4444"]],
            showscale=True, colorbar=dict(tickvals=[0,1], ticktext=["Present","Missing"])
        ))
        fig.update_layout(title="Missing Value Heatmap (top 200 rows)", title_font_size=14,
                          yaxis_title="Row Index", xaxis_tickangle=-45)
        return {"figure": fig_json(fig, min(500, 200+20*len(miss_cols)))}
    except Exception as e:
        return JSONResponse({"error": f"Missing heatmap failed: {str(e)}"}, 400)


@app.post("/api/explore/pairplot")
def explore_pairplot():
    """Pair plot of top correlated numeric features."""
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    try:
        df, tgt = S["df"], S["target"]
        nc = df.select_dtypes(include=np.number).columns.tolist()
        fn = [c for c in nc if c != tgt]
        if len(fn) < 2: return {"figure": None, "message": "Need at least 2 numeric features for pair plot"}
        n = min(5, len(fn))
        if tgt in nc:
            top = df[fn].corrwith(df[tgt]).abs().sort_values(ascending=False).head(n).index.tolist()
        else:
            top = fn[:n]
        sample_cols = list(dict.fromkeys(top + [tgt]))  # deduplicate
        sample = df[sample_cols].dropna()
        if len(sample) > 400: sample = sample.sample(400, random_state=42)
        clr = tgt if df[tgt].nunique() <= 10 else None
        if clr and clr in sample.columns and pd.api.types.is_numeric_dtype(sample[clr]):
            sample[clr] = sample[clr].astype(str)
        fig = px.scatter_matrix(sample, dimensions=top, color=clr)
        fig.update_layout(title=f"Pair Plot — Top {n} Features", title_font_size=14)
        fig.update_traces(diagonal_visible=True)
        return {"figure": fig_json(fig, 560)}
    except Exception as e:
        return JSONResponse({"error": f"Pair plot failed: {str(e)}"}, 400)


@app.post("/api/explore/scatter_xy")
def explore_scatter_xy(feature: str = Form(...), feature2: str = Form(...)):
    """Scatter plot of feature vs feature2, colored by target."""
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    try:
        df, tgt = S["df"], S["target"]
        cols = list(dict.fromkeys([feature, feature2, tgt]))  # deduplicate
        sub = df[cols].dropna()
        if len(sub) > 2000: sub = sub.sample(2000, random_state=42)
        clr = tgt if df[tgt].nunique() <= 10 else None
        if clr and clr in sub.columns and pd.api.types.is_numeric_dtype(sub[clr]):
            sub[clr] = sub[clr].astype(str)
        fig = px.scatter(sub, x=feature, y=feature2, color=clr, opacity=0.7)
        fig.update_layout(title=f"Scatter: {feature} vs {feature2}", title_font_size=14)
        return {"figure": fig_json(fig)}
    except Exception as e:
        return JSONResponse({"error": f"Scatter failed: {str(e)}"}, 400)


@app.post("/api/explore/jointplot")
def explore_jointplot(feature: str = Form(...), feature2: str = Form(...)):
    """Joint plot — scatter with marginal histograms."""
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    try:
        df = S["df"]
        sub = df[[feature, feature2]].dropna()
        if len(sub) > 2000: sub = sub.sample(2000, random_state=42)
        fig = px.scatter(sub, x=feature, y=feature2, marginal_x="histogram",
                         marginal_y="histogram", color_discrete_sequence=["#6366f1"],
                         opacity=0.6)
        fig.update_layout(title=f"Joint: {feature} vs {feature2}", title_font_size=14)
        return {"figure": fig_json(fig)}
    except Exception as e:
        return JSONResponse({"error": f"Joint plot failed: {str(e)}"}, 400)


@app.post("/api/explore/countplot")
def explore_countplot(feature: str = Form(...)):
    """Count plot — value counts for a categorical/discrete feature."""
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    try:
        vc = S["df"][feature].astype(str).value_counts().head(25).reset_index()
        vc.columns = [feature, "Count"]
        fig = px.bar(vc, x=feature, y="Count", color=feature, text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(title=f"Count Plot: {feature}", title_font_size=14, showlegend=False)
        return {"figure": fig_json(fig)}
    except Exception as e:
        return JSONResponse({"error": f"Count plot failed: {str(e)}"}, 400)


@app.post("/api/explore/pie")
def explore_pie():
    """Pie chart of target variable distribution."""
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    try:
        tgt = S["target"]
        vc = S["df"][tgt].astype(str).value_counts().head(15)
        fig = px.pie(values=vc.values.tolist(), names=vc.index.tolist(), hole=0.4)
        fig.update_layout(title=f"Target Distribution: {tgt}", title_font_size=14)
        return {"figure": fig_json(fig)}
    except Exception as e:
        return JSONResponse({"error": f"Pie chart failed: {str(e)}"}, 400)


@app.post("/api/explore/scatter_index")
def explore_scatter_index(feature: str = Form(...)):
    """Scatter plot of feature values vs row index."""
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    try:
        df = S["df"]
        if not pd.api.types.is_numeric_dtype(df[feature]):
            return JSONResponse({"error": f"Scatter vs Index requires a numeric feature. '{feature}' is categorical."}, 400)
        sub = df[[feature]].dropna().reset_index(drop=True)
        if len(sub) > 3000: sub = sub.sample(3000, random_state=42).sort_index()
        fig = px.scatter(sub.reset_index(), x="index", y=feature,
                         color_discrete_sequence=["#10b981"], opacity=0.5)
        fig.update_layout(title=f"{feature} vs Row Index", title_font_size=14,
                          xaxis_title="Row Index")
        return {"figure": fig_json(fig)}
    except Exception as e:
        return JSONResponse({"error": f"Scatter index failed: {str(e)}"}, 400)


@app.post("/api/explore/mean_target")
def explore_mean_target(feature: str = Form(...)):
    """Bar plot of mean target value per category/bin of feature."""
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    try:
        df, tgt, task = S["df"], S["target"], S["task"]
        sub = df[[feature, tgt]].dropna()

        # If target is categorical (classification), show target rate per class
        if not pd.api.types.is_numeric_dtype(df[tgt]):
            # For classification: show most-frequent-class proportion per feature category
            if pd.api.types.is_numeric_dtype(sub[feature]):
                sub["_bin"] = pd.qcut(sub[feature], q=min(8, sub[feature].nunique()),
                                      duplicates="drop").astype(str)
                ct = pd.crosstab(sub["_bin"], sub[tgt], normalize="index").reset_index()
                ct = ct.melt(id_vars="_bin", var_name=tgt, value_name="Proportion")
                fig = px.bar(ct, x="_bin", y="Proportion", color=tgt, barmode="stack")
                fig.update_layout(xaxis_title=feature)
            else:
                ct = pd.crosstab(sub[feature], sub[tgt], normalize="index").reset_index()
                ct = ct.melt(id_vars=feature, var_name=tgt, value_name="Proportion")
                ct = ct.sort_values("Proportion", ascending=False)
                fig = px.bar(ct, x=feature, y="Proportion", color=tgt, barmode="stack")
            fig.update_layout(title=f"Target Rate per {feature}", title_font_size=14,
                              xaxis_tickangle=-45)
        else:
            # Numeric target — compute mean
            if pd.api.types.is_numeric_dtype(sub[feature]):
                sub["_bin"] = pd.qcut(sub[feature], q=min(10, sub[feature].nunique()),
                                      duplicates="drop").astype(str)
                grp = sub.groupby("_bin")[tgt].mean().reset_index()
                grp.columns = [feature, f"Mean {tgt}"]
            else:
                grp = sub.groupby(feature)[tgt].mean().sort_values(ascending=False).head(20).reset_index()
                grp.columns = [feature, f"Mean {tgt}"]
            fig = px.bar(grp, x=feature, y=f"Mean {tgt}", color_discrete_sequence=["#f59e0b"])
            fig.update_layout(title=f"Mean {tgt} per {feature}", title_font_size=14,
                              xaxis_tickangle=-45)
        return {"figure": fig_json(fig)}
    except Exception as e:
        return JSONResponse({"error": f"Mean target failed: {str(e)}"}, 400)


@app.post("/api/explore/grouped_box")
def explore_grouped_box(feature: str = Form(...)):
    """Grouped box plot — feature distribution per target class."""
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    try:
        df, tgt = S["df"], S["target"]
        if not pd.api.types.is_numeric_dtype(df[feature]):
            return JSONResponse({"error": f"Grouped box plot requires a numeric feature. '{feature}' is categorical."}, 400)
        sub = df[[feature, tgt]].dropna()
        if len(sub) > 3000: sub = sub.sample(3000, random_state=42)
        sub[tgt] = sub[tgt].astype(str)
        fig = px.box(sub, x=tgt, y=feature, color=tgt)
        fig.update_layout(title=f"Grouped Box: {feature} by {tgt}", title_font_size=14,
                          showlegend=False)
        return {"figure": fig_json(fig)}
    except Exception as e:
        return JSONResponse({"error": f"Grouped box failed: {str(e)}"}, 400)


@app.post("/api/explore/facetgrid")
def explore_facetgrid(feature: str = Form(...)):
    """FacetGrid / small multiples — histogram of feature faceted by target class."""
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    try:
        df, tgt = S["df"], S["target"]
        sub = df[[feature, tgt]].dropna()
        if len(sub) > 3000: sub = sub.sample(3000, random_state=42)
        n_classes = sub[tgt].nunique()
        sub[tgt] = sub[tgt].astype(str)
        if n_classes > 8:
            fig = px.histogram(sub, x=feature, color=tgt, barmode="overlay",
                               opacity=0.6, nbins=30)
        else:
            fig = px.histogram(sub, x=feature, facet_col=tgt, facet_col_wrap=min(4, n_classes),
                               color=tgt, nbins=30)
        fig.update_layout(title=f"FacetGrid: {feature} by {tgt}", title_font_size=14,
                          showlegend=False)
        return {"figure": fig_json(fig, 420)}
    except Exception as e:
        return JSONResponse({"error": f"FacetGrid failed: {str(e)}"}, 400)


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

    # Drop rows where the target has missing values (PyCaret requires this)
    tgt_missing = work[tgt].isna().sum()
    if tgt_missing > 0:
        work = work.dropna(subset=[tgt])

    # Validate
    nuniq = work[tgt].nunique(dropna=True)
    if nuniq < 2:
        return JSONResponse({"error": f"Target '{tgt}' has only {nuniq} unique value(s)."}, 400)

    # Smart sampling: for large datasets, sample to keep training fast on free CPU
    MAX_TRAIN_ROWS = 8000
    sampled = False
    original_rows = len(work)
    if len(work) > MAX_TRAIN_ROWS:
        sampled = True
        if task == "classification" and work[tgt].nunique() <= 50:
            # Stratified sampling to preserve class distribution
            work = work.groupby(tgt, group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(2, int(MAX_TRAIN_ROWS * len(x) / original_rows))),
                                   random_state=42)
            ).reset_index(drop=True)
        else:
            work = work.sample(MAX_TRAIN_ROWS, random_state=42).reset_index(drop=True)

    # Setup — always start completely fresh
    # Clear any previous experiment state fully
    for k in ["experiment","compare_df","best_model","active_model","tuned_model","tune_df","plots"]:
        S.pop(k, None)

    # Reset PyCaret internal globals and free memory
    try:
        if task == "classification":
            import pycaret.classification as _pcm
        else:
            import pycaret.regression as _pcm
        if hasattr(_pcm, '_current_experiment'):
            _pcm._current_experiment = None
    except Exception:
        pass
    gc.collect()

    # Prepare clean DataFrame
    train_df = work.reset_index(drop=True).copy()
    # Sanitize column names for scikit-learn compatibility
    train_df.columns = [str(c).strip().replace('[','(').replace(']',')').replace('<','lt').replace('>','gt') for c in train_df.columns]
    tgt_clean = str(tgt).strip().replace('[','(').replace(']',')').replace('<','lt').replace('>','gt')

    # Pre-encode ALL categorical features BEFORE PyCaret to bypass its buggy
    # ColumnTransformer that creates feature name mismatches (space vs underscore).
    # PyCaret will only see numeric features + the target column.
    cat_cols = train_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    # Keep target column as-is (PyCaret handles target encoding itself)
    encode_cols = [c for c in cat_cols if c != tgt_clean]
    if encode_cols:
        train_df = pd.get_dummies(train_df, columns=encode_cols, drop_first=False, dtype=int)
        # Sanitize the new one-hot column names (replace spaces with underscores)
        train_df.columns = [str(c).replace(' ', '_') for c in train_df.columns]

    # Fill remaining NaN in numeric columns with median (PyCaret imputation can also bug out)
    for col in train_df.select_dtypes(include=np.number).columns:
        if col != tgt_clean and train_df[col].isna().any():
            train_df[col] = train_df[col].fillna(train_df[col].median())

    try:
        exp = get_exp(task)
        # All preprocessing forced OFF — we handled encoding above manually.
        # This guarantees no feature name mismatches from PyCaret's internal pipeline.
        kw = dict(data=train_df, target=tgt_clean, session_id=42,
                  train_size=1-test_size, fold=cv_folds,
                  normalize=False, transformation=False,
                  remove_multicollinearity=False,
                  feature_selection=False,
                  remove_outliers=False,
                  polynomial_features=False,
                  html=False, verbose=False,
                  log_experiment=False, log_plots=False)
        if task == "classification":
            kw["fix_imbalance"] = fix_imbalance
        exp.setup(**kw)
        S["experiment"] = exp
    except Exception as e:
        return JSONResponse({"error": f"Setup failed: {e}"}, 500)

    # Compare — limit to fast models to keep training under ~2 min on free CPU
    try:
        best = exp.compare_models(n_select=1, budget_time=4.0)
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
        "sampled": sampled,
        "train_rows": len(train_df),
        "original_rows": original_rows,
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
