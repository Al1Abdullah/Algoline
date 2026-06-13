"""
AutoML Studio — End-to-End Machine Learning Pipeline
Streamlit + PyCaret + SHAP + Optuna
"""

import io
import os
import glob
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AutoML Studio", page_icon="A", layout="wide")

PAGES = ["Data", "Explore", "Build", "Export"]
MODEL_FILE = "automl_pipeline"


# ═══════════════════════════════════════════════════════════════
# THEME
# ═══════════════════════════════════════════════════════════════

def apply_theme(dark):
    if dark:
        bg      = "#09090b"
        sf      = "#111113"
        cd      = "#18181b"
        tx      = "#d4d4d8"
        mt      = "#71717a"
        bd      = "#27272a"
        ac      = "#14b8a6"
        ac2     = "#2dd4bf"
        ib      = "#1c1c20"
    else:
        bg      = "#fafafa"
        sf      = "#ffffff"
        cd      = "#ffffff"
        tx      = "#18181b"
        mt      = "#71717a"
        bd      = "#e4e7eb"
        ac      = "#0d9488"
        ac2     = "#14b8a6"
        ib      = "#f4f4f5"

    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── base ── */
    html, body, .stApp, .stApp * {{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif!important;}}
    .stApp {{background:{bg};}}

    /* ── nuke streamlit chrome ── */
    #MainMenu,footer,header[data-testid="stHeader"],[data-testid="stToolbar"],
    [data-testid="stDecoration"],[data-testid="stSidebar"],
    div[data-testid="stStatusWidget"] {{display:none!important;visibility:hidden!important;}}

    .block-container {{max-width:1060px;padding:1.2rem 1rem 3rem;}}

    /* ── global text ── */
    h1,h2,h3,h4 {{color:{tx}!important;}}
    h1 {{font-size:1.3rem;font-weight:700;letter-spacing:-0.01em;margin-bottom:0.6rem;}}
    h2 {{font-size:1.05rem;font-weight:600;margin-top:1.4rem;margin-bottom:0.5rem;}}
    p,span,label,li,.stMarkdown,.stMarkdown p {{color:{tx}!important;}}
    .stCaption,.stCaption p,.stCaption span {{color:{mt}!important;font-size:0.78rem!important;}}

    /* ── metrics ── */
    [data-testid="stMetric"] {{
        background:{cd};border:1px solid {bd};border-radius:8px;padding:0.7rem 0.9rem;
    }}
    [data-testid="stMetricLabel"] p {{
        color:{mt}!important;font-size:0.7rem!important;font-weight:600;
        text-transform:uppercase;letter-spacing:0.04em;
    }}
    [data-testid="stMetricValue"] div {{font-size:1.3rem!important;font-weight:700;color:{tx}!important;}}

    /* ── dataframes ── */
    div[data-testid="stDataFrame"] {{border:1px solid {bd};border-radius:8px;overflow:hidden;}}

    /* ── buttons ── */
    .stButton>button {{
        background:{ac}!important;color:#fff!important;border:none!important;
        border-radius:6px;font-weight:600;font-size:0.82rem;padding:0.5rem 1.2rem;
        transition:all 0.12s ease;
    }}
    .stButton>button:hover {{background:{ac2}!important;color:#fff!important;}}
    .stButton>button:active {{transform:scale(0.98);}}
    .stDownloadButton>button {{
        background:{cd}!important;color:{tx}!important;
        border:1px solid {bd}!important;border-radius:6px;font-weight:500;font-size:0.82rem;
    }}
    .stDownloadButton>button:hover {{border-color:{ac}!important;color:{ac}!important;}}

    /* ── nav ── */
    div[data-testid="stSegmentedControl"] {{
        background:{sf};border:1px solid {bd};border-radius:8px;padding:0.2rem;
    }}
    div[data-testid="stSegmentedControl"] label {{
        border-radius:6px;font-weight:500;font-size:0.82rem;color:{mt}!important;
    }}
    div[data-testid="stSegmentedControl"] label[data-selected="true"],
    div[data-testid="stSegmentedControl"] label[aria-checked="true"] {{
        color:{tx}!important;font-weight:600;
    }}

    /* ── expander ── */
    details[data-testid="stExpander"] {{
        background:{cd};border:1px solid {bd}!important;border-radius:8px!important;
    }}
    details[data-testid="stExpander"] summary {{font-weight:600;font-size:0.85rem;color:{tx}!important;}}
    details[data-testid="stExpander"] summary span {{color:{tx}!important;}}
    details[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{color:{tx};}}

    /* ── tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap:0;background:{sf};border:1px solid {bd};border-radius:8px;padding:0.15rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius:6px;font-weight:500;font-size:0.82rem;color:{mt}!important;
    }}
    .stTabs [aria-selected="true"] {{color:{tx}!important;font-weight:600;}}
    .stTabs [data-baseweb="tab-panel"] {{color:{tx};}}

    /* ── file uploader ── */
    [data-testid="stFileUploader"] section {{
        border:1px dashed {bd};border-radius:8px;padding:1.5rem;background:{cd};
    }}
    [data-testid="stFileUploader"] section:hover {{border-color:{ac};}}
    [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] button {{color:{tx}!important;}}

    /* ── inputs ── */
    .stSelectbox label,.stTextArea label,.stTextInput label,
    .stSlider label,.stCheckbox label,.stRadio label,
    .stFileUploader label {{
        font-size:0.82rem!important;font-weight:500!important;color:{mt}!important;
    }}
    .stSelectbox [data-baseweb="select"]>div {{
        background:{ib}!important;border-color:{bd}!important;color:{tx}!important;
    }}
    .stSelectbox [data-baseweb="select"] span {{color:{tx}!important;}}
    .stSelectbox svg {{fill:{mt}!important;}}
    [data-baseweb="popover"],[data-baseweb="menu"] {{background:{cd}!important;border:1px solid {bd}!important;}}
    [data-baseweb="menu"] li {{color:{tx}!important;}}
    [data-baseweb="menu"] li:hover {{background:{ib}!important;}}
    .stTextArea textarea,.stTextInput input {{
        background:{ib}!important;border-color:{bd}!important;color:{tx}!important;
        border-radius:6px!important;font-size:0.85rem;
    }}
    .stTextArea textarea:focus,.stTextInput input:focus,
    .stSelectbox [data-baseweb="select"]>div:focus-within {{border-color:{ac}!important;}}

    .stCheckbox label span {{font-size:0.82rem!important;color:{tx}!important;}}
    .stRadio>div {{gap:0.4rem;}}
    .stRadio label span {{color:{tx}!important;font-size:0.82rem!important;}}

    /* ── slider ── */
    .stSlider [data-baseweb="slider"] div[role="slider"] {{background:{ac}!important;}}
    .stSlider [data-testid="stTickBarMin"],.stSlider [data-testid="stTickBarMax"] {{
        color:{mt}!important;font-size:0.75rem;
    }}
    .stSlider div[data-testid="stThumbValue"] {{color:{tx}!important;}}

    /* ── alerts ── */
    .stAlert {{border-radius:8px;}}
    .stAlert p,.stAlert span,.stAlert div {{color:{tx}!important;}}
    div[data-testid="stNotification"] p {{color:{tx}!important;}}

    /* ── radio horizontal ── */
    div[role="radiogroup"] label {{color:{tx}!important;}}
    div[role="radiogroup"] label span {{color:{tx}!important;}}

    /* ── spinner ── */
    .stSpinner>div {{color:{tx}!important;}}

    /* ── custom components ── */
    .hdr {{
        background:{sf};border:1px solid {bd};border-radius:8px;
        padding:0.65rem 1.1rem;margin-bottom:0.7rem;
    }}
    .hdr h1 {{font-size:0.9rem!important;margin:0!important;color:{tx}!important;}}
    .hdr p {{font-size:0.72rem;color:{mt}!important;margin:0;}}
    .tgl {{
        color:{mt}!important;background:{cd};border:1px solid {bd};border-radius:6px;
        padding:0.3rem 0.7rem;text-decoration:none!important;font-size:0.75rem;font-weight:500;
    }}
    .tgl:hover {{border-color:{ac};color:{tx}!important;}}
    </style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════

def read_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):   return pd.read_csv(uploaded)
    if name.endswith(".tsv"):   return pd.read_csv(uploaded, sep="\t")
    if name.endswith(".xlsx"):  return pd.read_excel(uploaded)
    raise ValueError("Use CSV, TSV, or XLSX.")


def show_df(df, n=50, **kw):
    """Always sanitize before display to prevent Arrow errors."""
    d = df.head(n).copy()
    for c in d.columns:
        if d[c].dtype == "object" or d[c].apply(type).nunique() > 1:
            d[c] = d[c].astype(str)
    st.dataframe(d, **kw)


def infer_task(df, target):
    y = df[target].dropna()
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > max(12, int(len(y) * 0.05)):
        return "Regression"
    return "Classification"


def pycaret_ok():
    try:
        import pycaret
        return True
    except ImportError:
        return False


def get_experiment(task):
    if task == "Classification":
        from pycaret.classification import ClassificationExperiment
        return ClassificationExperiment()
    from pycaret.regression import RegressionExperiment
    return RegressionExperiment()


def capture_plot(exp, model, plot_type):
    before = set(glob.glob("*.png"))
    try:
        exp.plot_model(model, plot=plot_type, save=True)
    except Exception:
        return None
    new = set(glob.glob("*.png")) - before
    return max(new, key=os.path.getmtime) if new else None


def capture_shap(exp, model, plot_type="summary"):
    before = set(glob.glob("*.png"))
    try:
        exp.interpret_model(model, plot=plot_type, save=True)
    except Exception:
        return None
    new = set(glob.glob("*.png")) - before
    return max(new, key=os.path.getmtime) if new else None


def fl(fig, h=None):
    dark = st.session_state.get("dark_mode", True)
    pbg = "#18181b" if dark else "#ffffff"
    fig.update_layout(
        margin=dict(l=16, r=16, t=36, b=16), height=h, legend_title_text="",
        paper_bgcolor=pbg, plot_bgcolor=pbg,
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def fq_table(df, target):
    num = df.select_dtypes(include=np.number).drop(columns=[target], errors="ignore")
    rows = []
    for c in num.columns:
        s = num[c].dropna()
        if s.empty: continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        out = int(((s < q1-1.5*iqr)|(s > q3+1.5*iqr)).sum()) if iqr else 0
        rows.append({"Feature":c, "Missing":int(df[c].isna().sum()),
                      "Unique":int(df[c].nunique()), "Mean":round(float(s.mean()),2),
                      "Std":round(float(s.std()),2), "Skew":round(float(s.skew()),2),
                      "Outliers":out})
    return pd.DataFrame(rows)


def norm_cmp(df):
    if "Model" not in df.columns:
        df = df.reset_index()
        if df.columns[0] != "Model":
            df = df.rename(columns={df.columns[0]: "Model"})
    return df


def reset_pipe():
    for k in ["experiment","setup_df","setup_done","compare_df","best_model",
              "tuned_model","active_model","tune_df","predictions","plot_paths"]:
        st.session_state.pop(k, None)


# ═══════════════════════════════════════════════════════════════
# PAGE: DATA
# ═══════════════════════════════════════════════════════════════

def page_data():
    st.header("Define Your Problem")
    st.text_area("Describe your prediction goal", value=st.session_state.get("problem",""),
                 key="problem", placeholder="e.g. Predict customer churn from usage data.", height=72)

    st.header("Upload Dataset")
    uploaded = st.file_uploader("CSV, TSV, or XLSX", type=["csv","tsv","xlsx"])

    if uploaded:
        try:
            df = read_file(uploaded)
            df.columns = [str(c).strip() for c in df.columns]
            st.session_state["raw_df"] = df
            reset_pipe()
        except Exception as e:
            st.error(f"Read error: {e}")

    if "raw_df" not in st.session_state:
        return

    df = st.session_state["raw_df"]

    st.header("Data Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Missing", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicates", f"{int(df.duplicated().sum()):,}")

    t1,t2,t3 = st.tabs(["Preview","Types","Statistics"])
    with t1:
        show_df(df, 50, use_container_width=True)
    with t2:
        td = pd.DataFrame({"Column":df.columns, "Type":[str(d) for d in df.dtypes],
                            "Non-Null":[int(df[c].notna().sum()) for c in df.columns],
                            "Unique":[int(df[c].nunique()) for c in df.columns]})
        show_df(td, 80, use_container_width=True)
    with t3:
        desc = df.describe(include="all").round(3).reset_index().rename(columns={"index":"Stat"})
        show_df(desc, 30, use_container_width=True)

    st.header("Target")
    cl,cr = st.columns(2)
    with cl:
        target = st.selectbox("Column to predict", df.columns, key="target_sel")
    with cr:
        task = infer_task(df, target)
        st.metric("Detected task", task)
    st.session_state["target"] = target
    st.session_state["task"] = task


# ═══════════════════════════════════════════════════════════════
# PAGE: EXPLORE
# ═══════════════════════════════════════════════════════════════

def page_explore():
    if "raw_df" not in st.session_state:
        st.info("Upload a dataset on the Data page first.")
        return

    df = st.session_state["raw_df"]
    target = st.session_state.get("target", df.columns[-1])
    nc = df.select_dtypes(include=np.number).columns.tolist()
    cc = df.select_dtypes(exclude=np.number).columns.tolist()
    fn = [c for c in nc if c != target]
    fc = [c for c in cc if c != target]

    st.header("Distributions")
    if fn:
        sel = st.selectbox("Feature", fn, key="d_f")
        st.plotly_chart(fl(px.histogram(df, x=sel, nbins=40, marginal="box")), use_container_width=True)
    else:
        st.info("No numeric features.")

    st.header("Value Counts")
    co = fc + ([target] if target not in fc else [])
    if co:
        sel = st.selectbox("Feature", co, key="c_f")
        cts = df[sel].astype(str).value_counts().head(25).reset_index()
        cts.columns = [sel, "Count"]
        st.plotly_chart(fl(px.bar(cts, x=sel, y="Count")), use_container_width=True)

    st.header("Correlation")
    if len(nc) >= 2:
        corr = df[nc].corr().round(2)
        fig = ff.create_annotated_heatmap(z=corr.values, x=corr.columns.tolist(),
              y=corr.index.tolist(), colorscale="RdBu", showscale=True, zmin=-1, zmax=1)
        fl(fig, min(660, 200+28*len(corr)))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need 2+ numeric columns.")

    st.header("Boxplots")
    if fn:
        sel = st.selectbox("Feature", fn, key="b_f")
        if df[target].nunique() <= 15:
            bd = df[[sel,target]].copy(); bd[target] = bd[target].astype(str)
            fig = px.box(bd, x=target, y=sel, color=target)
        else:
            fig = px.box(df, y=sel)
        st.plotly_chart(fl(fig), use_container_width=True)

    st.header("Scatter Matrix")
    if len(fn) >= 2:
        n = min(5, len(fn))
        top = (df[fn].corrwith(df[target]).abs().sort_values(ascending=False).head(n).index.tolist()
               if target in nc else fn[:n])
        sample = df[top+[target]].dropna()
        if len(sample) > 500: sample = sample.sample(500, random_state=42)
        clr = target if df[target].nunique() <= 10 else None
        if clr and pd.api.types.is_numeric_dtype(sample[clr]):
            sample[clr] = sample[clr].astype(str)
        fig = px.scatter_matrix(sample, dimensions=top, color=clr)
        fl(fig, 560)
        st.plotly_chart(fig, use_container_width=True)

    st.header("Feature Quality")
    qt = fq_table(df, target)
    if not qt.empty:
        show_df(qt, 50, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: BUILD
# ═══════════════════════════════════════════════════════════════

def page_build():
    if "raw_df" not in st.session_state:
        st.info("Upload a dataset on the Data page first.")
        return
    if not pycaret_ok():
        st.error("PyCaret is not installed. Run: pip install -r requirements.txt")
        return

    df = st.session_state["raw_df"]
    target = st.session_state.get("target")
    task = st.session_state.get("task")
    if not target:
        st.warning("Select a target on the Data page.")
        return

    task = st.radio("Task", ["Classification","Regression"],
                    index=0 if task=="Classification" else 1, horizontal=True)
    st.session_state["task"] = task

    st.header("Preprocessing")
    with st.expander("Cleaning", expanded=True):
        c1,c2,c3 = st.columns(3)
        o_dup = c1.checkbox("Remove duplicates", True, key="o1")
        o_miss = c2.checkbox("Handle missing values", True, key="o2")
        o_out = c3.checkbox("Remove outliers", False, key="o3")
        o_out_t = st.slider("Outlier threshold", 0.01, 0.10, 0.05, 0.01, key="o3t") if o_out else 0.05

    with st.expander("Feature engineering", expanded=True):
        c1,c2,c3 = st.columns(3)
        o_norm = c1.checkbox("Normalize", True, key="o4")
        o_trans = c2.checkbox("Transform skew", False, key="o5")
        o_fsel = c3.checkbox("Feature selection", False, key="o6")
        c4,c5,c6 = st.columns(3)
        o_multi = c4.checkbox("Drop multicollinear", True, key="o7")
        o_poly = c5.checkbox("Polynomial features", False, key="o8")
        o_imb = c6.checkbox("Fix class imbalance", False, disabled=task!="Classification", key="o9")
        c7,c8 = st.columns(2)
        o_test = c7.slider("Test size", 0.10, 0.40, 0.20, 0.05, key="o10")
        o_folds = c8.slider("CV folds", 2, 10, 5, 1, key="o11")

    st.header("Model Training")
    if st.button("Setup and Train All Models", type="primary", use_container_width=True):
        if df[target].nunique(dropna=True) < 2:
            st.error("Target needs 2+ unique values.")
        else:
            _train(df, target, task, dup=o_dup, out=o_out, out_t=o_out_t,
                   norm=o_norm, trans=o_trans, fsel=o_fsel, multi=o_multi,
                   poly=o_poly, imb=o_imb, test=o_test, folds=o_folds)

    if not st.session_state.get("setup_done"):
        return
    exp = st.session_state["experiment"]

    if "setup_df" in st.session_state:
        with st.expander("Setup summary"):
            show_df(st.session_state["setup_df"], 80, use_container_width=True)

    if "compare_df" in st.session_state:
        st.header("Model Comparison")
        cdf = st.session_state["compare_df"]
        show_df(cdf, 30, use_container_width=True)
        mc = [c for c in cdf.columns if c not in ("Model","TT (Sec)")]
        if mc and "Model" in cdf.columns:
            m = st.selectbox("Metric", mc, key="cm")
            st.plotly_chart(fl(px.bar(cdf, x="Model", y=m, color="Model")), use_container_width=True)

    if "best_model" in st.session_state:
        st.header("Hyperparameter Tuning")
        st.caption(f"Current: {type(st.session_state['active_model']).__name__}")
        t1,t2 = st.columns(2)
        sm = {"Bayesian (Optuna)":("optuna",None),
              "Random Search":("scikit-learn","random"),
              "Grid Search":("scikit-learn","grid")}
        sl = t1.selectbox("Method", list(sm.keys()), key="ts")
        ni = t2.slider("Iterations", 5, 50, 10, 5, key="tn")
        slib, salg = sm[sl]
        if st.button("Tune Best Model", use_container_width=True):
            _tune(exp, task, slib, salg, ni)
        if "tune_df" in st.session_state:
            with st.expander("Tuning results"):
                show_df(st.session_state["tune_df"], 30, use_container_width=True)

    if "active_model" in st.session_state:
        st.header("Evaluation")
        st.caption(f"Model: {type(st.session_state['active_model']).__name__}")
        plots = st.session_state.get("plot_paths", {})
        if plots:
            cols = st.columns(2)
            i = 0
            for lb, p in plots.items():
                if p and os.path.exists(p):
                    with cols[i%2]:
                        st.markdown(f"**{lb}**")
                        st.image(p, use_container_width=True)
                    i += 1
        if "predictions" in st.session_state:
            with st.expander(f"Holdout predictions ({len(st.session_state['predictions'])} rows)"):
                show_df(st.session_state["predictions"], 100, use_container_width=True)


def _train(df, target, task, **o):
    work = df.copy()
    if o.get("dup"):
        n0 = len(work); work = work.drop_duplicates()
        if n0-len(work): st.info(f"Removed {n0-len(work):,} duplicates.")

    with st.spinner("Setting up..."):
        try:
            exp = get_experiment(task)
            kw = {"data":work,"target":target,"session_id":42,
                  "train_size":1-o.get("test",0.2),"fold":o.get("folds",5),
                  "normalize":o.get("norm"),"transformation":o.get("trans"),
                  "remove_multicollinearity":o.get("multi"),
                  "multicollinearity_threshold":0.9,
                  "feature_selection":o.get("fsel"),
                  "remove_outliers":o.get("out"),
                  "outliers_threshold":o.get("out_t",0.05),
                  "polynomial_features":o.get("poly"),
                  "html":False,"verbose":False}
            if task=="Classification": kw["fix_imbalance"]=o.get("imb",False)
            exp.setup(**kw)
            st.session_state.update({"experiment":exp,"setup_df":exp.pull(),"setup_done":True})
        except Exception as e:
            st.error(f"Setup failed: {e}"); return

    with st.spinner("Training models..."):
        try:
            best = exp.compare_models()
            st.session_state.update({"compare_df":norm_cmp(exp.pull()),"best_model":best,"active_model":best})
            st.session_state.pop("tuned_model",None); st.session_state.pop("tune_df",None)
        except Exception as e:
            st.error(f"Training failed: {e}"); return

    with st.spinner("Generating plots..."):
        _plots(exp, best, task)
    with st.spinner("Predicting..."):
        try: st.session_state["predictions"] = exp.predict_model(best)
        except: pass
    st.rerun()


def _tune(exp, task, slib, salg, n_iter):
    best = st.session_state["best_model"]
    with st.spinner("Tuning..."):
        try:
            kw = {"estimator":best,"search_library":slib,"n_iter":n_iter,"choose_better":True}
            if salg: kw["search_algorithm"] = salg
            tuned = exp.tune_model(**kw)
            st.session_state.update({"tuned_model":tuned,"active_model":tuned,"tune_df":exp.pull()})
        except Exception as e:
            st.error(f"Tuning failed: {e}"); return
    with st.spinner("Updating plots..."):
        _plots(exp, tuned, task)
    try: st.session_state["predictions"] = exp.predict_model(tuned)
    except: pass
    st.rerun()


def _plots(exp, model, task):
    plots = {}
    pm = ([("confusion_matrix","Confusion Matrix"),("auc","ROC / AUC"),
           ("pr","Precision-Recall"),("feature","Feature Importance"),
           ("class_report","Classification Report")]
          if task=="Classification"
          else [("residuals","Residuals"),("error","Predicted vs Actual"),
                ("feature","Feature Importance")])
    for pk,lb in pm:
        p = capture_plot(exp, model, pk)
        if p: plots[lb] = p
    for st_t,lb in [("summary","SHAP Summary"),("correlation","SHAP Correlation")]:
        p = capture_shap(exp, model, st_t)
        if p: plots[lb] = p
    st.session_state["plot_paths"] = plots


# ═══════════════════════════════════════════════════════════════
# PAGE: EXPORT
# ═══════════════════════════════════════════════════════════════

def page_export():
    if "active_model" not in st.session_state:
        st.info("Train a model on the Build page first.")
        return

    exp = st.session_state["experiment"]
    model = st.session_state["active_model"]
    name = type(model).__name__

    st.header("Export")
    st.success(f"Model ready: **{name}**")

    try:
        final = exp.finalize_model(model)
        exp.save_model(final, MODEL_FILE)
        mp = f"{MODEL_FILE}.pkl"
        if os.path.exists(mp):
            with open(mp,"rb") as f:
                st.download_button("Download pipeline (.pkl)", f.read(),
                                   "automl_pipeline.pkl","application/octet-stream",
                                   use_container_width=True)
            st.caption("Full preprocessing + model. Load with pycaret or joblib.")
    except Exception as e:
        st.error(f"Save failed: {e}")

    if "compare_df" in st.session_state:
        st.download_button("Download leaderboard (.csv)",
                           st.session_state["compare_df"].to_csv(index=False).encode(),
                           "model_leaderboard.csv", use_container_width=True)
    if "predictions" in st.session_state:
        st.download_button("Download predictions (.csv)",
                           st.session_state["predictions"].to_csv(index=False).encode(),
                           "holdout_predictions.csv", use_container_width=True)

    prob = st.session_state.get("problem","")
    if prob:
        st.header("Summary")
        st.markdown(f"| | |\n|---|---|\n| **Goal** | {prob} |\n| **Target** | `{st.session_state.get('target','-')}` |\n| **Task** | {st.session_state.get('task','-')} |\n| **Model** | {name} |\n| **Exported** | {datetime.now().strftime('%Y-%m-%d %H:%M')} |")


# ═══════════════════════════════════════════════════════════════
# SHELL
# ═══════════════════════════════════════════════════════════════

if "dark_mode" not in st.session_state: st.session_state["dark_mode"] = True
if "active_page" not in st.session_state: st.session_state["active_page"] = PAGES[0]
if st.session_state.get("active_page") not in PAGES: st.session_state["active_page"] = PAGES[0]

tp = st.query_params.get("theme","dark")
if isinstance(tp,list): tp=tp[0]
st.session_state["dark_mode"] = tp!="light"
apply_theme(st.session_state["dark_mode"])

px.defaults.template = "plotly_dark" if st.session_state["dark_mode"] else "plotly_white"
px.defaults.color_discrete_sequence = [
    "#14b8a6","#6366f1","#f59e0b","#ef4444","#8b5cf6",
    "#06b6d4","#84cc16","#ec4899","#f97316","#10b981"]

st.markdown("""<div class="hdr"><h1>AutoML Studio</h1>
<p>End-to-end machine learning pipeline</p></div>""", unsafe_allow_html=True)

nc,tc = st.columns([0.85,0.15], vertical_alignment="center")
with nc:
    sel = st.segmented_control("n", PAGES, default=st.session_state["active_page"],
                               label_visibility="collapsed")
    if sel: st.session_state["active_page"] = sel
with tc:
    nt = "light" if st.session_state["dark_mode"] else "dark"
    tl = "Light" if st.session_state["dark_mode"] else "Dark"
    st.markdown(f'<a class="tgl" href="?theme={nt}" target="_self">{tl}</a>', unsafe_allow_html=True)

pg = st.session_state["active_page"]
if pg=="Data":     page_data()
elif pg=="Explore": page_explore()
elif pg=="Build":   page_build()
elif pg=="Export":  page_export()
