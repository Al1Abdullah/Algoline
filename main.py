"""
AutoML Studio — Production-Grade End-to-End ML Pipeline
Gradio + PyCaret + SHAP + Optuna + Plotly
"""

import os, glob, warnings, tempfile
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import gradio as gr

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ────────────────────────────────────────────────────────────

PALETTE = ["#10b981","#6366f1","#f59e0b","#ef4444","#8b5cf6",
           "#06b6d4","#84cc16","#ec4899","#f97316","#14b8a6"]
px.defaults.color_discrete_sequence = PALETTE

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.zinc,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill_dark="*neutral_950",
    block_background_fill="white",
    block_background_fill_dark="*neutral_900",
    block_border_width="1px",
    block_border_color="*neutral_200",
    block_border_color_dark="*neutral_800",
    block_shadow="0 1px 3px rgba(0,0,0,0.04)",
    block_shadow_dark="0 1px 3px rgba(0,0,0,0.2)",
    block_label_text_weight="500",
    block_label_text_size="*text_sm",
    block_label_text_color="*neutral_500",
    block_label_text_color_dark="*neutral_400",
    block_title_text_weight="600",
    block_title_text_size="*text_md",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_500",
    button_primary_text_color="white",
    button_primary_shadow="0 2px 6px rgba(16,185,129,0.25)",
    button_primary_shadow_hover="0 4px 14px rgba(16,185,129,0.35)",
    button_secondary_background_fill="*neutral_100",
    button_secondary_background_fill_dark="*neutral_800",
    button_secondary_background_fill_hover="*neutral_200",
    button_secondary_background_fill_hover_dark="*neutral_700",
    input_background_fill="*neutral_50",
    input_background_fill_dark="*neutral_800",
    input_border_color="*neutral_200",
    input_border_color_dark="*neutral_700",
    input_border_color_focus="*primary_500",
    input_border_color_focus_dark="*primary_400",
    input_shadow_focus="0 0 0 2px rgba(16,185,129,0.15)",
    checkbox_background_color_selected="*primary_600",
    checkbox_background_color_selected_dark="*primary_500",
    slider_color="*primary_500",
)

CSS = """
footer {display:none!important}
.gradio-container {max-width:1060px!important}

/* typography */
h1 {font-weight:700!important;letter-spacing:-.025em!important;font-size:1.5rem!important}
h2 {font-weight:600!important;font-size:1.05rem!important}
h3 {font-weight:600!important;font-size:.92rem!important}
.subtitle {font-size:.82rem;opacity:.55;margin-top:2px;font-weight:400}

/* transitions */
button {transition:all .15s ease!important}
button:active {transform:scale(.98)!important}
input,textarea,select,details {transition:all .15s ease!important}

/* tabs */
.tab-nav button {font-weight:500!important;font-size:.84rem!important;padding:10px 18px!important;
  border-bottom:2px solid transparent!important;transition:all .15s ease!important}
.tab-nav button.selected {font-weight:600!important;border-bottom-color:var(--primary-500)!important}

/* upload */
.file-upload {border:2px dashed var(--border-color-primary)!important;border-radius:14px!important;
  padding:28px!important;transition:border-color .2s ease!important;background:transparent!important}
.file-upload:hover {border-color:var(--primary-500)!important}

/* gallery */
.gallery {border-radius:10px!important;overflow:hidden}

/* status label */
.status-ok {color:var(--primary-500);font-weight:600;font-size:.82rem}
.status-wait {color:var(--neutral-400);font-weight:500;font-size:.82rem}

/* custom containers */
.hero {text-align:center;padding:24px 16px 16px}
.hero h1 {font-size:1.6rem!important;margin-bottom:0!important}
.hero .subtitle {font-size:.88rem;margin-top:6px}
"""


# ────────────────────────────────────────────────────────────
# UTILITIES
# ────────────────────────────────────────────────────────────

def read_file(path):
    if path is None: return None
    p = path.lower() if isinstance(path, str) else ""
    if p.endswith(".tsv"): return pd.read_csv(path, sep="\t")
    if p.endswith(".xlsx"): return pd.read_excel(path)
    return pd.read_csv(path)

def safe_df(df, n=100):
    d = df.head(n).copy()
    for c in d.columns:
        try:
            if d[c].dtype == "object" or d[c].apply(type).nunique() > 1:
                d[c] = d[c].astype(str)
        except Exception:
            d[c] = d[c].astype(str)
    return d

def infer_task(df, target):
    y = df[target].dropna()
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > max(12, int(len(y) * 0.05)):
        return "Regression"
    return "Classification"

def make_metric(label, value):
    return (f'<div style="text-align:center;padding:18px 10px">'
            f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.07em;opacity:.4;margin-bottom:6px">{label}</div>'
            f'<div style="font-size:28px;font-weight:700;letter-spacing:-.02em">{value}</div></div>')

def overview_html(df):
    r, c = len(df), df.shape[1]
    m, d = int(df.isna().sum().sum()), int(df.duplicated().sum())
    pct = f"{m/(r*c)*100:.1f}%" if r*c else "0%"
    cards = (make_metric("Rows", f"{r:,}") + make_metric("Columns", c)
             + make_metric("Missing", f"{m:,} ({pct})") + make_metric("Duplicates", f"{d:,}"))
    return (f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;'
            f'margin:12px 0;border:1px solid var(--border-color-primary);'
            f'border-radius:12px;overflow:hidden;background:var(--block-background-fill)">{cards}</div>')

def style_fig(fig, h=None):
    fig.update_layout(
        margin=dict(l=20, r=20, t=36, b=20), height=h,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="", font=dict(family="Inter,sans-serif", size=12, color="#71717a"),
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.08)", zeroline=False)
    return fig

def quality_table(df, target):
    num = df.select_dtypes(include=np.number).drop(columns=[target], errors="ignore")
    rows = []
    for c in num.columns:
        s = num[c].dropna()
        if s.empty: continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr else 0
        miss = int(df[c].isna().sum())
        rows.append({"Feature": c, "Missing": miss, "Missing %": round(miss/len(df)*100, 1),
                      "Unique": int(df[c].nunique()), "Mean": round(float(s.mean()), 2),
                      "Std": round(float(s.std()), 2), "Skew": round(float(s.skew()), 2),
                      "Outliers": out})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ────────────────────────────────────────────────────────────
# ML ENGINE
# ────────────────────────────────────────────────────────────

def get_exp(task):
    if task == "Classification":
        from pycaret.classification import ClassificationExperiment
        return ClassificationExperiment()
    from pycaret.regression import RegressionExperiment
    return RegressionExperiment()

def grab_plot(exp, model, ptype):
    before = set(glob.glob("*.png"))
    try: exp.plot_model(model, plot=ptype, save=True)
    except Exception: return None
    new = set(glob.glob("*.png")) - before
    return max(new, key=os.path.getmtime) if new else None

def grab_shap(exp, model, ptype="summary"):
    before = set(glob.glob("*.png"))
    try: exp.interpret_model(model, plot=ptype, save=True)
    except Exception: return None
    new = set(glob.glob("*.png")) - before
    return max(new, key=os.path.getmtime) if new else None

def gen_eval_plots(exp, model, task):
    images = []
    if task == "Classification":
        plot_list = [("confusion_matrix", "Confusion Matrix"), ("auc", "ROC / AUC"),
                     ("pr", "Precision-Recall"), ("feature", "Feature Importance"),
                     ("class_report", "Classification Report")]
    else:
        plot_list = [("residuals", "Residuals"), ("error", "Predicted vs Actual"),
                     ("feature", "Feature Importance")]
    for pk, lb in plot_list:
        p = grab_plot(exp, model, pk)
        if p: images.append((p, lb))
    for stype, lb in [("summary", "SHAP Summary"), ("correlation", "SHAP Dependence")]:
        p = grab_shap(exp, model, stype)
        if p: images.append((p, lb))
    return images

def norm_leaderboard(df):
    if "Model" not in df.columns:
        df = df.reset_index()
        if df.columns[0] != "Model":
            df = df.rename(columns={df.columns[0]: "Model"})
    return df


# ────────────────────────────────────────────────────────────
# CALLBACKS — DATA
# ────────────────────────────────────────────────────────────

def on_upload(filepath, state):
    if filepath is None:
        return (state, "", None, gr.update(choices=[], value=None), "",
                gr.update(choices=[], value=None), gr.update(choices=[], value=None),
                "", gr.update(visible=False))
    try:
        df = read_file(filepath)
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as e:
        raise gr.Error(f"Could not read file: {e}")

    if len(df) < 10:
        raise gr.Error("Dataset too small. Need at least 10 rows.")
    if df.shape[1] < 2:
        raise gr.Error("Dataset needs at least 2 columns.")

    state["df"] = df
    tgt = df.columns[-1]
    state["target"] = tgt
    state["task"] = infer_task(df, tgt)
    # clear pipeline
    for k in ["experiment", "setup_df", "compare_df", "best_model", "tuned_model",
              "active_model", "tune_df", "predictions", "plot_paths"]:
        state.pop(k, None)

    nc = df.select_dtypes(include=np.number).columns.tolist()
    fn = [c for c in nc if c != tgt]
    first = fn[0] if fn else None

    # data health summary
    miss_pct = df.isna().sum().sum() / (len(df) * df.shape[1]) * 100
    health = "Good" if miss_pct < 5 else ("Fair" if miss_pct < 20 else "Poor")
    health_md = (f"**Data quality:** {health}  |  "
                 f"**Numeric:** {len(nc)}  |  "
                 f"**Categorical:** {df.shape[1] - len(nc)}  |  "
                 f"**Missing:** {miss_pct:.1f}%")

    return (state, overview_html(df), safe_df(df, 50),
            gr.update(choices=df.columns.tolist(), value=tgt), state["task"],
            gr.update(choices=fn, value=first), gr.update(choices=fn, value=first),
            health_md, gr.update(visible=True))


def on_target(target, state):
    if not target or "df" not in state:
        return state, ""
    state["target"] = target
    state["task"] = infer_task(state["df"], target)
    return state, state["task"]


# ────────────────────────────────────────────────────────────
# CALLBACKS — EXPLORE
# ────────────────────────────────────────────────────────────

def plot_distribution(feature, state):
    if not feature or "df" not in state: return None
    fig = px.histogram(state["df"], x=feature, nbins=40, marginal="box",
                       color_discrete_sequence=["#10b981"])
    fig.update_layout(title=f"Distribution of {feature}", title_font_size=14)
    return style_fig(fig)

def plot_counts(state):
    if "df" not in state: return None
    df, tgt = state["df"], state.get("target", state["df"].columns[-1])
    cats = df.select_dtypes(exclude=np.number).columns.tolist()
    col = cats[0] if cats else tgt
    cts = df[col].astype(str).value_counts().head(20).reset_index()
    cts.columns = [col, "Count"]
    fig = px.bar(cts, x=col, y="Count", color_discrete_sequence=["#6366f1"])
    fig.update_layout(title=f"Value Counts: {col}", title_font_size=14)
    return style_fig(fig)

def plot_correlation(state):
    if "df" not in state: return None
    nc = state["df"].select_dtypes(include=np.number).columns.tolist()
    if len(nc) < 2: return None
    corr = state["df"][nc].corr().round(2)
    fig = ff.create_annotated_heatmap(z=corr.values, x=corr.columns.tolist(),
          y=corr.index.tolist(), colorscale="RdBu", showscale=True, zmin=-1, zmax=1)
    fig.update_layout(title="Feature Correlation Matrix", title_font_size=14)
    return style_fig(fig, min(640, 200 + 28 * len(corr)))

def plot_boxplot(feature, state):
    if not feature or "df" not in state: return None
    df, tgt = state["df"], state.get("target", state["df"].columns[-1])
    if df[tgt].nunique() <= 15:
        bd = df[[feature, tgt]].copy(); bd[tgt] = bd[tgt].astype(str)
        fig = px.box(bd, x=tgt, y=feature, color=tgt)
    else:
        fig = px.box(df, y=feature)
    fig.update_layout(title=f"Boxplot: {feature}", title_font_size=14)
    return style_fig(fig)

def plot_scatter_matrix(state):
    if "df" not in state: return None
    df, tgt = state["df"], state.get("target", state["df"].columns[-1])
    nc = df.select_dtypes(include=np.number).columns.tolist()
    fn = [c for c in nc if c != tgt]
    if len(fn) < 2: return None
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
    return style_fig(fig, 520)

def get_quality(state):
    if "df" not in state: return None
    qt = quality_table(state["df"], state.get("target", state["df"].columns[-1]))
    return safe_df(qt, 50) if not qt.empty else None


# ────────────────────────────────────────────────────────────
# CALLBACKS — BUILD
# ────────────────────────────────────────────────────────────

def train(state, task, dup, outliers, normalize, transform, feat_sel,
          multi, poly, imbalance, test_size, folds, progress=gr.Progress()):
    if "df" not in state:
        raise gr.Error("Upload a dataset on the Data tab first.")
    try:
        import pycaret
    except ImportError:
        raise gr.Error("PyCaret is not installed.")

    df, tgt = state["df"], state["target"]
    state["task"] = task

    # Validate
    nuniq = df[tgt].nunique(dropna=True)
    if nuniq < 2:
        raise gr.Error(f"Target '{tgt}' has only {nuniq} unique value(s). Need at least 2.")
    if task == "Classification" and nuniq > 50:
        raise gr.Error(f"Target '{tgt}' has {nuniq} classes. Too many for classification. "
                       "Consider regression or reduce classes.")

    work = df.copy()
    removed = 0
    if dup:
        n0 = len(work); work = work.drop_duplicates(); removed = n0 - len(work)

    # Setup
    progress(0.08, desc="Configuring experiment...")
    try:
        exp = get_exp(task)
        kw = dict(data=work, target=tgt, session_id=42,
                  train_size=1 - test_size, fold=int(folds),
                  normalize=normalize, transformation=transform,
                  remove_multicollinearity=multi, multicollinearity_threshold=0.9,
                  feature_selection=feat_sel, remove_outliers=outliers,
                  outliers_threshold=0.05, polynomial_features=poly,
                  html=False, verbose=False)
        if task == "Classification":
            kw["fix_imbalance"] = imbalance
        exp.setup(**kw)
        state["experiment"] = exp
        state["setup_df"] = exp.pull()
    except Exception as e:
        raise gr.Error(f"Setup failed: {e}")

    # Compare
    progress(0.3, desc="Training and comparing models (this may take a few minutes)...")
    try:
        best = exp.compare_models()
        cdf = norm_leaderboard(exp.pull())
        state.update({"compare_df": cdf, "best_model": best, "active_model": best})
        state.pop("tuned_model", None); state.pop("tune_df", None)
    except Exception as e:
        raise gr.Error(f"Model training failed: {e}")

    # Evaluate
    progress(0.7, desc="Generating evaluation plots...")
    state["plot_paths"] = gen_eval_plots(exp, best, task)

    progress(0.9, desc="Running holdout predictions...")
    try:
        state["predictions"] = exp.predict_model(best)
    except Exception:
        pass

    # Prepare outputs
    mc = [c for c in cdf.columns if c not in ("Model", "TT (Sec)")]
    comp_fig = style_fig(px.bar(cdf, x="Model", y=mc[0], color="Model")) if mc else None
    name = type(best).__name__

    dup_msg = f"  Removed {removed:,} duplicates." if removed else ""
    status = f"Training complete.{dup_msg} Best model: **{name}**"

    return (state,
            gr.update(value=safe_df(cdf, 30), visible=True),
            gr.update(value=comp_fig, visible=True),
            gr.update(value=state.get("plot_paths", []), visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            status,
            gr.update(choices=mc, value=mc[0] if mc else None, visible=True))


def update_comparison(metric, state):
    if "compare_df" not in state or not metric: return None
    cdf = state["compare_df"]
    if "Model" not in cdf.columns: return None
    fig = px.bar(cdf, x="Model", y=metric, color="Model")
    fig.update_layout(title=f"Model Comparison: {metric}", title_font_size=14)
    return style_fig(fig)


def tune(state, method, iters, progress=gr.Progress()):
    if "best_model" not in state:
        raise gr.Error("Train models first.")
    exp, best, task = state["experiment"], state["best_model"], state["task"]

    methods = {"Bayesian (Optuna)": ("optuna", None),
               "Random Search": ("scikit-learn", "random"),
               "Grid Search": ("scikit-learn", "grid")}
    slib, salg = methods.get(method, ("optuna", None))

    progress(0.15, desc="Tuning hyperparameters...")
    try:
        kw = {"estimator": best, "search_library": slib, "n_iter": int(iters), "choose_better": True}
        if salg: kw["search_algorithm"] = salg
        tuned = exp.tune_model(**kw)
        state.update({"tuned_model": tuned, "active_model": tuned, "tune_df": exp.pull()})
    except Exception as e:
        raise gr.Error(f"Tuning failed: {e}")

    progress(0.65, desc="Updating evaluation plots...")
    state["plot_paths"] = gen_eval_plots(exp, tuned, task)
    try:
        state["predictions"] = exp.predict_model(tuned)
    except Exception:
        pass

    name = type(tuned).__name__
    improved = tuned is not best
    msg = f"Tuning complete. Model: **{name}**" + (" (improved)" if improved else " (no improvement)")

    return (state, gr.update(value=safe_df(state["tune_df"], 30), visible=True),
            state.get("plot_paths", []), msg)


# ────────────────────────────────────────────────────────────
# CALLBACKS — EXPORT
# ────────────────────────────────────────────────────────────

def export_pipeline(state):
    if "active_model" not in state: raise gr.Error("Train a model first.")
    exp, model = state["experiment"], state["active_model"]
    path = os.path.join(tempfile.gettempdir(), "automl_pipeline")
    try:
        final = exp.finalize_model(model)
        exp.save_model(final, path)
        return gr.update(value=f"{path}.pkl", visible=True)
    except Exception as e:
        raise gr.Error(f"Export failed: {e}")

def export_csv(state, key, fname):
    if key not in state: raise gr.Error(f"No data available for {fname}.")
    path = os.path.join(tempfile.gettempdir(), fname)
    state[key].to_csv(path, index=False)
    return gr.update(value=path, visible=True)

def get_summary(state, goal):
    if "active_model" not in state:
        return "Train a model on the **Build** tab to see your export summary."
    name = type(state["active_model"]).__name__
    tgt, task = state.get("target", "-"), state.get("task", "-")
    n_models = len(state.get("compare_df", [])) if "compare_df" in state else "?"
    tuned = "Yes" if "tuned_model" in state else "No"
    n_plots = len(state.get("plot_paths", []))
    n_preds = len(state.get("predictions", [])) if "predictions" in state else 0

    md = f"""### Pipeline Summary

| Property | Value |
|:---|:---|
| **Target** | `{tgt}` |
| **Task** | {task} |
| **Models compared** | {n_models} |
| **Best model** | {name} |
| **Tuned** | {tuned} |
| **Eval plots** | {n_plots} |
| **Holdout predictions** | {n_preds:,} rows |
| **Timestamp** | {datetime.now().strftime('%Y-%m-%d %H:%M')} |"""
    if goal and goal.strip():
        md += f"\n| **Goal** | {goal.strip()} |"
    return md


# ────────────────────────────────────────────────────────────
# APPLICATION
# ────────────────────────────────────────────────────────────

with gr.Blocks(theme=theme, css=CSS, title="AutoML Studio") as app:
    state = gr.State({})

    # ─── Header ───
    gr.HTML('<div class="hero"><h1>AutoML Studio</h1>'
            '<div class="subtitle">Upload a dataset. Train models. Export results.</div></div>')

    with gr.Tabs():

        # ═══════════ DATA ═══════════
        with gr.Tab("Data"):
            gr.Markdown("### Upload Dataset")
            file_in = gr.File(label="CSV, TSV, or XLSX",
                              file_types=[".csv", ".tsv", ".xlsx"],
                              type="filepath", elem_classes=["file-upload"])
            overview = gr.HTML("")
            health_md = gr.Markdown("", visible=False)
            preview = gr.Dataframe(label="Preview", interactive=False, wrap=True)

            gr.Markdown("### Target Configuration")
            with gr.Row():
                target_dd = gr.Dropdown(label="Column to predict", interactive=True, scale=3)
                task_box = gr.Textbox(label="Detected task", interactive=False, scale=1)

            with gr.Accordion("Goal description (optional)", open=False):
                goal_txt = gr.Textbox(lines=2, show_label=False,
                                      placeholder="Describe what you want to predict and why.")

        # ═══════════ EXPLORE ═══════════
        with gr.Tab("Explore"):
            gr.Markdown("### Exploratory Data Analysis")
            with gr.Tabs():
                with gr.Tab("Distribution"):
                    dist_feat = gr.Dropdown(label="Numeric feature", interactive=True)
                    dist_plot = gr.Plot()
                with gr.Tab("Value Counts") as tab_cnt:
                    cnt_plot = gr.Plot()
                with gr.Tab("Correlation") as tab_corr:
                    corr_plot = gr.Plot()
                with gr.Tab("Boxplot"):
                    box_feat = gr.Dropdown(label="Numeric feature", interactive=True)
                    box_plot = gr.Plot()
                with gr.Tab("Scatter Matrix") as tab_scat:
                    scat_plot = gr.Plot()
                with gr.Tab("Quality") as tab_qual:
                    qual_df = gr.Dataframe(label="Feature statistics", interactive=False)

        # ═══════════ BUILD ═══════════
        with gr.Tab("Build"):
            gr.Markdown("### Model Training")
            build_task = gr.Radio(["Classification", "Regression"], value="Classification",
                                   label="Task type")
            with gr.Accordion("Preprocessing options", open=False):
                with gr.Row():
                    o_dup   = gr.Checkbox(label="Drop duplicates", value=True)
                    o_out   = gr.Checkbox(label="Remove outliers", value=False)
                    o_norm  = gr.Checkbox(label="Normalize", value=True)
                with gr.Row():
                    o_multi = gr.Checkbox(label="Drop multicollinear", value=True)
                    o_trans = gr.Checkbox(label="Transform skew", value=False)
                    o_fsel  = gr.Checkbox(label="Feature selection", value=False)
                with gr.Row():
                    o_poly  = gr.Checkbox(label="Polynomial features", value=False)
                    o_imb   = gr.Checkbox(label="Fix class imbalance", value=False)
                with gr.Row():
                    o_test  = gr.Slider(0.10, 0.40, 0.20, step=0.05, label="Test split")
                    o_folds = gr.Slider(2, 10, 5, step=1, label="Cross-validation folds")

            train_btn = gr.Button("Train All Models", variant="primary", size="lg")
            train_status = gr.Markdown("")

            gr.Markdown("### Results")
            lb_df = gr.Dataframe(label="Leaderboard", interactive=False, visible=False)
            metric_dd = gr.Dropdown(label="Metric", interactive=True, visible=False)
            comp_plot = gr.Plot(visible=False)

            with gr.Group(visible=False) as tune_grp:
                gr.Markdown("### Hyperparameter Tuning")
                with gr.Row():
                    t_method = gr.Dropdown(["Bayesian (Optuna)", "Random Search", "Grid Search"],
                                            value="Bayesian (Optuna)", label="Search method")
                    t_iters = gr.Slider(5, 50, 10, step=5, label="Iterations")
                tune_btn = gr.Button("Tune Best Model", variant="secondary")
                tune_status = gr.Markdown("")
                tune_df = gr.Dataframe(label="Tuning results", interactive=False, visible=False)

            with gr.Group(visible=False) as eval_grp:
                gr.Markdown("### Evaluation Plots")
                eval_gal = gr.Gallery(label="Diagnostics", columns=2,
                                       object_fit="contain", height=460,
                                       elem_classes=["gallery"])

        # ═══════════ EXPORT ═══════════
        with gr.Tab("Export") as tab_export:
            export_md = gr.Markdown("Train a model on the **Build** tab to see export options.")
            with gr.Row(equal_height=True):
                exp_pipe_btn = gr.Button("Download Pipeline (.pkl)", variant="secondary")
                exp_lb_btn   = gr.Button("Download Leaderboard (.csv)", variant="secondary")
                exp_pred_btn = gr.Button("Download Predictions (.csv)", variant="secondary")
            exp_file = gr.File(label="Download", visible=False)

    # ─── Event Wiring ───

    # Data
    file_in.change(on_upload, [file_in, state],
                   [state, overview, preview, target_dd, task_box,
                    dist_feat, box_feat, health_md, health_md])
    target_dd.change(on_target, [target_dd, state], [state, task_box])

    # Explore
    dist_feat.change(plot_distribution, [dist_feat, state], dist_plot)
    box_feat.change(plot_boxplot, [box_feat, state], box_plot)
    tab_cnt.select(plot_counts, [state], cnt_plot)
    tab_corr.select(plot_correlation, [state], corr_plot)
    tab_scat.select(plot_scatter_matrix, [state], scat_plot)
    tab_qual.select(get_quality, [state], qual_df)

    # Build
    train_btn.click(
        train, [state, build_task, o_dup, o_out, o_norm, o_trans, o_fsel,
                o_multi, o_poly, o_imb, o_test, o_folds],
        [state, lb_df, comp_plot, eval_gal, tune_grp, eval_grp, train_status, metric_dd])
    metric_dd.change(update_comparison, [metric_dd, state], comp_plot)
    tune_btn.click(tune, [state, t_method, t_iters],
                   [state, tune_df, eval_gal, tune_status])

    # Export
    tab_export.select(get_summary, [state, goal_txt], export_md)
    exp_pipe_btn.click(export_pipeline, [state], exp_file)
    exp_lb_btn.click(lambda s: export_csv(s, "compare_df", "leaderboard.csv"), [state], exp_file)
    exp_pred_btn.click(lambda s: export_csv(s, "predictions", "predictions.csv"), [state], exp_file)

app.launch()
