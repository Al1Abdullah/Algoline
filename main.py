"""AutoML Studio — End-to-End ML Pipeline (Gradio)"""

import os, glob, warnings, tempfile
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import gradio as gr

warnings.filterwarnings("ignore")

# ── Colors ──
COLORS = ["#10b981","#6366f1","#f59e0b","#ef4444","#8b5cf6",
          "#06b6d4","#84cc16","#ec4899","#f97316","#14b8a6"]
px.defaults.color_discrete_sequence = COLORS

# ── Theme ──
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.zinc,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    radius_size=gr.themes.sizes.radius_md,
)

CSS = """
footer {display:none!important}
.gradio-container {max-width:1080px!important}
.prose h1 {font-size:1.4rem!important;font-weight:700!important;margin-bottom:.2rem!important}
.prose h3 {font-size:.95rem!important;font-weight:600!important;margin-top:1.2rem!important}
.prose p {font-size:.85rem!important}
"""

# ────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────
def read_upload(path):
    if path is None: return None
    p = path.lower() if isinstance(path, str) else ""
    if p.endswith(".csv"):  return pd.read_csv(path)
    if p.endswith(".tsv"):  return pd.read_csv(path, sep="\t")
    if p.endswith(".xlsx"): return pd.read_excel(path)
    return pd.read_csv(path)

def safe_df(df, n=100):
    d = df.head(n).copy()
    for c in d.columns:
        try:
            if d[c].dtype == "object" or d[c].apply(type).nunique() > 1:
                d[c] = d[c].astype(str)
        except: d[c] = d[c].astype(str)
    return d

def infer_task(df, target):
    y = df[target].dropna()
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > max(12, int(len(y)*0.05)):
        return "Regression"
    return "Classification"

def overview_html(df):
    def card(label, value):
        return (f'<div style="text-align:center;padding:18px 10px;background:var(--block-background-fill);'
                f'border:1px solid var(--border-color-primary);border-radius:10px;">'
                f'<div style="font-size:10px;font-weight:600;text-transform:uppercase;'
                f'letter-spacing:.06em;opacity:.5;margin-bottom:6px">{label}</div>'
                f'<div style="font-size:26px;font-weight:700">{value}</div></div>')
    r, c = len(df), df.shape[1]
    m, d = int(df.isna().sum().sum()), int(df.duplicated().sum())
    return (f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:8px 0">'
            f'{card("Rows",f"{r:,}")}{card("Columns",c)}{card("Missing",f"{m:,}")}'
            f'{card("Duplicates",f"{d:,}")}</div>')

def style_fig(fig, h=None):
    fig.update_layout(
        margin=dict(l=20,r=20,t=36,b=20), height=h,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="", font=dict(family="Inter,sans-serif", size=12),
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.08)", zeroline=False)
    return fig

def get_exp(task):
    if task == "Classification":
        from pycaret.classification import ClassificationExperiment
        return ClassificationExperiment()
    from pycaret.regression import RegressionExperiment
    return RegressionExperiment()

def grab_plot(exp, model, ptype):
    before = set(glob.glob("*.png"))
    try: exp.plot_model(model, plot=ptype, save=True)
    except: return None
    new = set(glob.glob("*.png")) - before
    return max(new, key=os.path.getmtime) if new else None

def grab_shap(exp, model, ptype="summary"):
    before = set(glob.glob("*.png"))
    try: exp.interpret_model(model, plot=ptype, save=True)
    except: return None
    new = set(glob.glob("*.png")) - before
    return max(new, key=os.path.getmtime) if new else None

def gen_plots(exp, model, task):
    images = []
    pm = ([("confusion_matrix","Confusion Matrix"),("auc","ROC / AUC"),
           ("pr","Precision-Recall"),("feature","Feature Importance"),
           ("class_report","Classification Report")]
          if task=="Classification"
          else [("residuals","Residuals"),("error","Predicted vs Actual"),
                ("feature","Feature Importance")])
    for pk, lb in pm:
        p = grab_plot(exp, model, pk)
        if p: images.append((p, lb))
    for st, lb in [("summary","SHAP Summary"),("correlation","SHAP Correlation")]:
        p = grab_shap(exp, model, st)
        if p: images.append((p, lb))
    return images

def quality_table(df, target):
    num = df.select_dtypes(include=np.number).drop(columns=[target], errors="ignore")
    rows = []
    for c in num.columns:
        s = num[c].dropna()
        if s.empty: continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        out = int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum()) if iqr else 0
        rows.append({"Feature":c,"Missing":int(df[c].isna().sum()),
            "Unique":int(df[c].nunique()),"Mean":round(float(s.mean()),2),
            "Std":round(float(s.std()),2),"Skew":round(float(s.skew()),2),
            "Outliers":out})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ────────────────────────────────────────────────────────────
# CALLBACKS
# ────────────────────────────────────────────────────────────

def on_upload(filepath, state):
    if filepath is None:
        return (state, "", None, gr.update(choices=[], value=None), "",
                gr.update(choices=[], value=None), gr.update(choices=[], value=None))
    try:
        df = read_upload(filepath)
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as e:
        raise gr.Error(f"Could not read file: {e}")

    state["df"] = df
    tgt = df.columns[-1]
    state["target"] = tgt
    state["task"] = infer_task(df, tgt)
    for k in ["experiment","setup_df","compare_df","best_model","tuned_model",
              "active_model","tune_df","predictions","plot_paths"]:
        state.pop(k, None)

    nc = df.select_dtypes(include=np.number).columns.tolist()
    fn = [c for c in nc if c != tgt]
    first = fn[0] if fn else None

    return (state, overview_html(df), safe_df(df, 50),
            gr.update(choices=df.columns.tolist(), value=tgt), state["task"],
            gr.update(choices=fn, value=first), gr.update(choices=fn, value=first))


def on_target(target, state):
    if not target or "df" not in state: return state, ""
    state["target"] = target
    state["task"] = infer_task(state["df"], target)
    return state, state["task"]


# ── Explore callbacks ──

def plot_dist(feature, state):
    if not feature or "df" not in state: return None
    fig = px.histogram(state["df"], x=feature, nbins=40, marginal="box",
                       color_discrete_sequence=["#10b981"])
    return style_fig(fig)

def plot_counts(state):
    if "df" not in state: return None
    df = state["df"]; tgt = state.get("target", df.columns[-1])
    cats = df.select_dtypes(exclude=np.number).columns.tolist()
    col = cats[0] if cats else tgt
    cts = df[col].astype(str).value_counts().head(20).reset_index()
    cts.columns = [col, "Count"]
    return style_fig(px.bar(cts, x=col, y="Count", color_discrete_sequence=["#6366f1"]))

def plot_corr(state):
    if "df" not in state: return None
    nc = state["df"].select_dtypes(include=np.number).columns.tolist()
    if len(nc) < 2: return None
    corr = state["df"][nc].corr().round(2)
    fig = ff.create_annotated_heatmap(z=corr.values, x=corr.columns.tolist(),
          y=corr.index.tolist(), colorscale="RdBu", showscale=True, zmin=-1, zmax=1)
    return style_fig(fig, min(640, 200+28*len(corr)))

def plot_box(feature, state):
    if not feature or "df" not in state: return None
    df = state["df"]; tgt = state.get("target", df.columns[-1])
    if df[tgt].nunique() <= 15:
        bd = df[[feature, tgt]].copy(); bd[tgt] = bd[tgt].astype(str)
        fig = px.box(bd, x=tgt, y=feature, color=tgt)
    else:
        fig = px.box(df, y=feature)
    return style_fig(fig)

def plot_scatter(state):
    if "df" not in state: return None
    df = state["df"]; tgt = state.get("target", df.columns[-1])
    nc = df.select_dtypes(include=np.number).columns.tolist()
    fn = [c for c in nc if c != tgt]
    if len(fn) < 2: return None
    n = min(5, len(fn))
    top = (df[fn].corrwith(df[tgt]).abs().sort_values(ascending=False).head(n).index.tolist()
           if tgt in nc else fn[:n])
    sample = df[top+[tgt]].dropna()
    if len(sample) > 500: sample = sample.sample(500, random_state=42)
    clr = tgt if df[tgt].nunique() <= 10 else None
    if clr and pd.api.types.is_numeric_dtype(sample[clr]):
        sample[clr] = sample[clr].astype(str)
    return style_fig(px.scatter_matrix(sample, dimensions=top, color=clr), 520)

def get_quality(state):
    if "df" not in state: return None
    return safe_df(quality_table(state["df"], state.get("target", state["df"].columns[-1])), 50)


# ── Build callbacks ──

def train_models(state, task, dup, outliers, normalize, transform, feat_sel,
                 multi, poly, imbalance, test_size, folds, progress=gr.Progress()):
    if "df" not in state: raise gr.Error("Upload a dataset first.")
    try:
        import pycaret
    except ImportError:
        raise gr.Error("PyCaret is not installed. Run: pip install -r requirements.txt")

    df, tgt = state["df"], state["target"]
    state["task"] = task
    work = df.copy()
    if dup:
        work = work.drop_duplicates()

    progress(0.1, desc="Setting up experiment...")
    try:
        exp = get_exp(task)
        kw = {"data":work,"target":tgt,"session_id":42,
              "train_size":1-test_size,"fold":int(folds),
              "normalize":normalize,"transformation":transform,
              "remove_multicollinearity":multi,"multicollinearity_threshold":0.9,
              "feature_selection":feat_sel,"remove_outliers":outliers,
              "outliers_threshold":0.05,"polynomial_features":poly,
              "html":False,"verbose":False}
        if task == "Classification": kw["fix_imbalance"] = imbalance
        exp.setup(**kw)
        state["experiment"] = exp
        state["setup_df"] = exp.pull()
    except Exception as e:
        raise gr.Error(f"Setup failed: {e}")

    progress(0.35, desc="Training and comparing models...")
    try:
        best = exp.compare_models()
        cdf = exp.pull()
        if "Model" not in cdf.columns:
            cdf = cdf.reset_index()
            if cdf.columns[0] != "Model": cdf = cdf.rename(columns={cdf.columns[0]:"Model"})
        state["compare_df"] = cdf
        state["best_model"] = best
        state["active_model"] = best
        state.pop("tuned_model", None)
        state.pop("tune_df", None)
    except Exception as e:
        raise gr.Error(f"Training failed: {e}")

    progress(0.7, desc="Generating evaluation plots...")
    state["plot_paths"] = gen_plots(exp, best, task)

    progress(0.9, desc="Running holdout predictions...")
    try: state["predictions"] = exp.predict_model(best)
    except: pass

    mc = [c for c in cdf.columns if c not in ("Model","TT (Sec)")]
    comp_fig = style_fig(px.bar(cdf, x="Model", y=mc[0], color="Model")) if mc and "Model" in cdf.columns else None
    name = type(best).__name__

    return (
        state,
        gr.update(value=safe_df(cdf, 30), visible=True),
        gr.update(value=comp_fig, visible=True),
        gr.update(value=state.get("plot_paths",[]), visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        f"Best model: **{name}**",
        gr.update(choices=mc, value=mc[0] if mc else None, visible=True),
    )

def update_chart(metric, state):
    if "compare_df" not in state or not metric: return None
    cdf = state["compare_df"]
    if "Model" not in cdf.columns: return None
    return style_fig(px.bar(cdf, x="Model", y=metric, color="Model"))

def tune_model_fn(state, method, iters, progress=gr.Progress()):
    if "best_model" not in state: raise gr.Error("Train models first.")
    exp, best, task = state["experiment"], state["best_model"], state["task"]
    sm = {"Bayesian (Optuna)":("optuna",None),"Random Search":("scikit-learn","random"),
          "Grid Search":("scikit-learn","grid")}
    slib, salg = sm.get(method, ("optuna",None))

    progress(0.2, desc="Tuning hyperparameters...")
    try:
        kw = {"estimator":best,"search_library":slib,"n_iter":int(iters),"choose_better":True}
        if salg: kw["search_algorithm"] = salg
        tuned = exp.tune_model(**kw)
        state["tuned_model"] = tuned
        state["active_model"] = tuned
        state["tune_df"] = exp.pull()
    except Exception as e:
        raise gr.Error(f"Tuning failed: {e}")

    progress(0.7, desc="Updating plots...")
    state["plot_paths"] = gen_plots(exp, tuned, task)
    try: state["predictions"] = exp.predict_model(tuned)
    except: pass

    return (state, gr.update(value=safe_df(state["tune_df"],30), visible=True),
            state.get("plot_paths",[]), f"Tuned: **{type(tuned).__name__}**")


# ── Export callbacks ──

def export_pipeline(state):
    if "active_model" not in state: raise gr.Error("Train a model first.")
    exp, model = state["experiment"], state["active_model"]
    path = os.path.join(tempfile.gettempdir(), "automl_pipeline")
    try:
        final = exp.finalize_model(model)
        exp.save_model(final, path)
        return gr.update(value=f"{path}.pkl", visible=True)
    except Exception as e:
        raise gr.Error(f"Save failed: {e}")

def export_csv(state, key, fname):
    if key not in state: raise gr.Error(f"No {key} data available.")
    path = os.path.join(tempfile.gettempdir(), fname)
    state[key].to_csv(path, index=False)
    return gr.update(value=path, visible=True)

def get_summary(state):
    if "active_model" not in state:
        return "Train a model on the Build tab to see results."
    name = type(state["active_model"]).__name__
    return (f"| | |\n|---|---|\n| **Target** | `{state.get('target','-')}` |"
            f"\n| **Task** | {state.get('task','-')} |"
            f"\n| **Best Model** | {name} |"
            f"\n| **Exported** | {datetime.now().strftime('%Y-%m-%d %H:%M')} |")


# ────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────

with gr.Blocks(theme=theme, css=CSS, title="AutoML Studio") as app:
    state = gr.State({})
    gr.Markdown("# AutoML Studio\nEnd-to-end machine learning pipeline")

    with gr.Tabs():

        # ─── DATA ───
        with gr.Tab("Data"):
            file_in = gr.File(label="Upload dataset (CSV / TSV / XLSX)",
                              file_types=[".csv",".tsv",".xlsx"], type="filepath")
            overview = gr.HTML("")
            preview = gr.Dataframe(label="Preview", interactive=False, wrap=True)
            with gr.Row():
                target_dd = gr.Dropdown(label="Target column", interactive=True, scale=2)
                task_box = gr.Textbox(label="Detected task", interactive=False, scale=1)
            with gr.Accordion("Describe your goal (optional)", open=False):
                goal_txt = gr.Textbox(lines=2, show_label=False,
                                      placeholder="e.g. Predict customer churn from usage data.")

        # ─── EXPLORE ───
        with gr.Tab("Explore"):
            with gr.Tabs():
                with gr.Tab("Distribution") as tab_dist:
                    dist_feat = gr.Dropdown(label="Feature", interactive=True)
                    dist_plot = gr.Plot()
                with gr.Tab("Counts") as tab_cnt:
                    cnt_plot = gr.Plot()
                with gr.Tab("Correlation") as tab_corr:
                    corr_plt = gr.Plot()
                with gr.Tab("Boxplot"):
                    box_feat = gr.Dropdown(label="Feature", interactive=True)
                    box_plt = gr.Plot()
                with gr.Tab("Scatter Matrix") as tab_scat:
                    scat_plt = gr.Plot()
                with gr.Tab("Quality") as tab_qual:
                    qual_df = gr.Dataframe(label="Feature quality", interactive=False)

        # ─── BUILD ───
        with gr.Tab("Build"):
            build_task = gr.Radio(["Classification","Regression"], value="Classification",
                                   label="Task type")
            with gr.Accordion("Preprocessing options", open=False):
                with gr.Row():
                    o_dup  = gr.Checkbox(label="Drop duplicates", value=True)
                    o_out  = gr.Checkbox(label="Remove outliers", value=False)
                    o_norm = gr.Checkbox(label="Normalize", value=True)
                with gr.Row():
                    o_multi = gr.Checkbox(label="Drop multicollinear", value=True)
                    o_trans = gr.Checkbox(label="Transform skew", value=False)
                    o_fsel  = gr.Checkbox(label="Feature selection", value=False)
                with gr.Row():
                    o_poly = gr.Checkbox(label="Polynomial features", value=False)
                    o_imb  = gr.Checkbox(label="Fix class imbalance", value=False)
                with gr.Row():
                    o_test  = gr.Slider(0.10, 0.40, 0.20, step=0.05, label="Test size")
                    o_folds = gr.Slider(2, 10, 5, step=1, label="CV folds")

            train_btn = gr.Button("Train All Models", variant="primary", size="lg")
            train_status = gr.Markdown("")

            lb_df = gr.Dataframe(label="Leaderboard", interactive=False, visible=False)
            metric_dd = gr.Dropdown(label="Compare metric", interactive=True, visible=False)
            comp_plt = gr.Plot(visible=False)

            with gr.Group(visible=False) as tune_grp:
                gr.Markdown("### Hyperparameter Tuning")
                with gr.Row():
                    t_method = gr.Dropdown(["Bayesian (Optuna)","Random Search","Grid Search"],
                                            value="Bayesian (Optuna)", label="Method")
                    t_iters = gr.Slider(5, 50, 10, step=5, label="Iterations")
                tune_btn = gr.Button("Tune Best Model")
                tune_status = gr.Markdown("")
                tune_df = gr.Dataframe(label="Tuning results", interactive=False, visible=False)

            with gr.Group(visible=False) as eval_grp:
                gr.Markdown("### Evaluation")
                eval_gal = gr.Gallery(label="Diagnostic plots", columns=2,
                                       object_fit="contain", height=450)

        # ─── EXPORT ───
        with gr.Tab("Export") as tab_export:
            export_md = gr.Markdown("Train a model to see export options.")
            with gr.Row():
                exp_pipe_btn = gr.Button("Download Pipeline (.pkl)")
                exp_lb_btn   = gr.Button("Download Leaderboard (.csv)")
                exp_pred_btn = gr.Button("Download Predictions (.csv)")
            exp_file = gr.File(label="Download", visible=False)

    # ─── WIRING ───

    # Data
    file_in.change(on_upload, [file_in, state],
                   [state, overview, preview, target_dd, task_box, dist_feat, box_feat])
    target_dd.change(on_target, [target_dd, state], [state, task_box])

    # Explore
    dist_feat.change(plot_dist, [dist_feat, state], dist_plot)
    box_feat.change(plot_box, [box_feat, state], box_plt)
    tab_cnt.select(plot_counts, [state], cnt_plot)
    tab_corr.select(plot_corr, [state], corr_plt)
    tab_scat.select(plot_scatter, [state], scat_plt)
    tab_qual.select(get_quality, [state], qual_df)

    # Build
    train_btn.click(
        train_models,
        [state, build_task, o_dup, o_out, o_norm, o_trans, o_fsel,
         o_multi, o_poly, o_imb, o_test, o_folds],
        [state, lb_df, comp_plt, eval_gal, tune_grp, eval_grp, train_status, metric_dd],
    )
    metric_dd.change(update_chart, [metric_dd, state], comp_plt)
    tune_btn.click(tune_model_fn, [state, t_method, t_iters],
                   [state, tune_df, eval_gal, tune_status])

    # Export
    tab_export.select(get_summary, [state], export_md)
    exp_pipe_btn.click(export_pipeline, [state], exp_file)
    exp_lb_btn.click(lambda s: export_csv(s, "compare_df", "leaderboard.csv"), [state], exp_file)
    exp_pred_btn.click(lambda s: export_csv(s, "predictions", "predictions.csv"), [state], exp_file)

app.launch()
