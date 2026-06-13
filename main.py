"""AutoML Studio — End-to-End Machine Learning Pipeline"""

import os, glob, warnings, io
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

warnings.filterwarnings("ignore")
st.set_page_config(page_title="AutoML Studio", page_icon="A", layout="wide")

PAGES = ["Data", "Explore", "Build", "Export"]
MODEL_PATH = "automl_pipeline"

# ────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ────────────────────────────────────────────────────────────
def apply_css(dark):
    if dark:
        bg="#0a0a0c"; sf="#111114"; cd="#16161a"; cdh="#1c1c21"
        tx="#ececef"; t2="#8b8b95"; tm="#56565e"
        bd="#232329"; bdh="#363640"; ac="#10b981"; ach="#34d399"
        acs="rgba(16,185,129,0.08)"; ib="#1a1a1f"
    else:
        bg="#f7f7f8"; sf="#ffffff"; cd="#ffffff"; cdh="#f9fafb"
        tx="#111114"; t2="#6b7280"; tm="#9ca3af"
        bd="#e5e7eb"; bdh="#d1d5db"; ac="#059669"; ach="#10b981"
        acs="rgba(5,150,105,0.05)"; ib="#f3f4f6"

    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    *{{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif!important;}}
    .stApp{{background:{bg};}}

    /* kill chrome */
    #MainMenu,footer,header[data-testid="stHeader"],[data-testid="stToolbar"],
    [data-testid="stDecoration"],[data-testid="stSidebar"],
    div[data-testid="stStatusWidget"]{{display:none!important;}}

    .block-container{{max-width:980px;padding:1rem 1.2rem 4rem;}}

    /* typography */
    h1{{color:{tx}!important;font-size:1.15rem!important;font-weight:700;letter-spacing:-.02em;margin:0 0 .4rem!important;}}
    h2{{color:{tx}!important;font-size:.95rem!important;font-weight:600;margin:1.8rem 0 .6rem!important;}}
    h3{{color:{tx}!important;font-size:.88rem!important;font-weight:600;}}
    p,span,label,li,.stMarkdown,.stMarkdown p{{color:{tx}!important;font-size:.84rem;}}
    .stCaption,.stCaption *{{color:{tm}!important;font-size:.74rem!important;}}

    /* metric cards */
    [data-testid="stMetric"]{{
        background:{cd};border:1px solid {bd};border-radius:10px;padding:.75rem 1rem;
    }}
    [data-testid="stMetricLabel"] p{{
        color:{t2}!important;font-size:.68rem!important;font-weight:600;
        text-transform:uppercase;letter-spacing:.05em;margin-bottom:.15rem!important;
    }}
    [data-testid="stMetricValue"] div{{font-size:1.35rem!important;font-weight:700;color:{tx}!important;}}

    /* dataframe */
    div[data-testid="stDataFrame"]{{border:1px solid {bd};border-radius:10px;overflow:hidden;}}

    /* primary btn */
    .stButton>button{{
        background:{ac}!important;color:#fff!important;border:none!important;
        border-radius:8px;font-weight:600;font-size:.82rem;padding:.55rem 1.4rem;
        transition:all .15s ease;letter-spacing:.01em;
    }}
    .stButton>button:hover{{background:{ach}!important;transform:translateY(-1px);}}
    .stButton>button:active{{transform:translateY(0);}}

    /* download btn */
    .stDownloadButton>button{{
        background:{cd}!important;color:{tx}!important;
        border:1px solid {bd}!important;border-radius:8px;font-weight:500;font-size:.82rem;
        transition:all .15s ease;
    }}
    .stDownloadButton>button:hover{{border-color:{ac}!important;color:{ac}!important;}}

    /* nav */
    div[data-testid="stSegmentedControl"]{{
        background:{sf};border:1px solid {bd};border-radius:10px;padding:.2rem;
    }}
    div[data-testid="stSegmentedControl"] label{{
        border-radius:8px;font-weight:500;font-size:.8rem;color:{t2}!important;
        transition:all .12s ease;
    }}
    div[data-testid="stSegmentedControl"] label[data-selected="true"],
    div[data-testid="stSegmentedControl"] label[aria-checked="true"]{{
        color:{tx}!important;font-weight:600;
    }}

    /* expander */
    details[data-testid="stExpander"]{{
        background:{cd};border:1px solid {bd}!important;border-radius:10px!important;
    }}
    details[data-testid="stExpander"] summary{{font-weight:600;font-size:.84rem;color:{tx}!important;}}
    details[data-testid="stExpander"] summary span{{color:{tx}!important;}}
    details[data-testid="stExpander"] div{{color:{tx}!important;}}

    /* tabs */
    .stTabs [data-baseweb="tab-list"]{{
        gap:0;background:{sf};border:1px solid {bd};border-radius:10px;padding:.15rem;
    }}
    .stTabs [data-baseweb="tab"]{{
        border-radius:8px;font-weight:500;font-size:.8rem;color:{t2}!important;
    }}
    .stTabs [aria-selected="true"]{{color:{tx}!important;font-weight:600;}}
    .stTabs [data-baseweb="tab-panel"]{{padding-top:.8rem;}}

    /* file uploader */
    [data-testid="stFileUploader"] section{{
        border:2px dashed {bd};border-radius:12px;padding:2rem 1.5rem;background:{cd};
        transition:border-color .15s ease;
    }}
    [data-testid="stFileUploader"] section:hover{{border-color:{ac};}}
    [data-testid="stFileUploader"] *{{color:{tx}!important;}}

    /* inputs */
    .stSelectbox label,.stTextArea label,.stTextInput label,
    .stSlider label,.stCheckbox label,.stRadio label,
    .stFileUploader label{{
        font-size:.78rem!important;font-weight:500!important;color:{t2}!important;margin-bottom:.2rem!important;
    }}
    .stSelectbox [data-baseweb="select"]>div{{
        background:{ib}!important;border-color:{bd}!important;border-radius:8px!important;color:{tx}!important;
        min-height:38px;font-size:.84rem;
    }}
    .stSelectbox [data-baseweb="select"] span{{color:{tx}!important;}}
    .stSelectbox svg{{fill:{t2}!important;}}
    [data-baseweb="popover"],[data-baseweb="menu"]{{background:{cd}!important;border:1px solid {bd}!important;border-radius:10px!important;}}
    [data-baseweb="menu"] li{{color:{tx}!important;font-size:.84rem;}}
    [data-baseweb="menu"] li:hover{{background:{ib}!important;}}
    .stTextArea textarea,.stTextInput input{{
        background:{ib}!important;border-color:{bd}!important;border-radius:8px!important;
        color:{tx}!important;font-size:.84rem;min-height:38px;
    }}
    .stTextArea textarea:focus,.stTextInput input:focus,
    .stSelectbox [data-baseweb="select"]>div:focus-within{{border-color:{ac}!important;}}
    .stCheckbox label span{{font-size:.82rem!important;color:{tx}!important;}}
    .stRadio>div{{gap:.3rem;}}
    .stRadio label span,.stRadio div[role="radiogroup"] label{{color:{tx}!important;font-size:.82rem!important;}}

    /* slider */
    .stSlider [data-baseweb="slider"] div[role="slider"]{{background:{ac}!important;}}
    .stSlider [data-testid="stTickBarMin"],.stSlider [data-testid="stTickBarMax"]{{color:{tm}!important;}}
    .stSlider div[data-testid="stThumbValue"]{{color:{tx}!important;}}

    /* alerts */
    .stAlert{{border-radius:10px;}}
    .stAlert *{{color:{tx}!important;}}
    div[data-testid="stNotification"] *{{color:{tx}!important;}}

    /* spinner */
    .stSpinner *{{color:{tx}!important;}}

    /* custom */
    .brand{{
        background:linear-gradient(135deg,{acs},{sf});
        border:1px solid {bd};border-radius:12px;
        padding:.8rem 1.2rem;margin-bottom:.8rem;
    }}
    .brand h1{{font-size:.88rem!important;margin:0!important;}}
    .brand p{{font-size:.7rem;color:{t2}!important;margin:0;}}
    .tgl{{
        color:{t2}!important;background:{cd};border:1px solid {bd};border-radius:8px;
        padding:.32rem .7rem;text-decoration:none!important;font-size:.74rem;font-weight:500;
        transition:all .15s ease;
    }}
    .tgl:hover{{border-color:{ac};color:{tx}!important;}}
    .sec{{margin-top:1.6rem;margin-bottom:.5rem;}}
    .badge{{
        display:inline-block;background:{acs};color:{ac}!important;
        font-size:.65rem;font-weight:600;padding:.15rem .5rem;
        border-radius:20px;text-transform:uppercase;letter-spacing:.04em;
        margin-bottom:.3rem;
    }}
    </style>""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# UTILITIES
# ────────────────────────────────────────────────────────────
def read_file(f):
    n = f.name.lower()
    if n.endswith(".csv"):  return pd.read_csv(f)
    if n.endswith(".tsv"):  return pd.read_csv(f, sep="\t")
    if n.endswith(".xlsx"): return pd.read_excel(f)
    raise ValueError("Use CSV, TSV, or XLSX.")

def show_df(df, n=50, **kw):
    d = df.head(n).copy()
    for c in d.columns:
        try:
            if d[c].dtype == "object" or d[c].apply(type).nunique() > 1:
                d[c] = d[c].astype(str)
        except Exception:
            d[c] = d[c].astype(str)
    st.dataframe(d, **kw)

def infer_task(df, target):
    y = df[target].dropna()
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > max(12, int(len(y)*0.05)):
        return "Regression"
    return "Classification"

def pycaret_available():
    try: import pycaret; return True
    except ImportError: return False

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

def chart_layout(fig, h=None):
    dark = st.session_state.get("dark_mode", True)
    pbg = "#16161a" if dark else "#ffffff"
    gtx = "#8b8b95" if dark else "#9ca3af"
    fig.update_layout(
        margin=dict(l=20, r=20, t=36, b=20), height=h,
        paper_bgcolor=pbg, plot_bgcolor=pbg, legend_title_text="",
        font=dict(family="Inter, sans-serif", size=12, color=gtx),
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
        out = int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum()) if iqr else 0
        rows.append({"Feature":c,"Missing":int(df[c].isna().sum()),
            "Unique":int(df[c].nunique()),"Mean":round(float(s.mean()),2),
            "Std":round(float(s.std()),2),"Skew":round(float(s.skew()),2),"Outliers":out})
    return pd.DataFrame(rows)

def norm_cmp(df):
    if "Model" not in df.columns:
        df = df.reset_index()
        if df.columns[0] != "Model": df = df.rename(columns={df.columns[0]:"Model"})
    return df

def reset_state():
    for k in ["experiment","setup_df","setup_done","compare_df","best_model",
              "tuned_model","active_model","tune_df","predictions","plot_paths"]:
        st.session_state.pop(k, None)

def badge(text):
    st.markdown(f'<div class="badge">{text}</div>', unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# PAGE: DATA
# ────────────────────────────────────────────────────────────
def page_data():
    badge("Step 1 of 4")
    st.header("Upload and Configure")
    st.caption("Load your dataset, preview it, and select the target column.")

    uploaded = st.file_uploader("Drop your file here", type=["csv","tsv","xlsx"])
    if uploaded:
        try:
            df = read_file(uploaded)
            df.columns = [str(c).strip() for c in df.columns]
            st.session_state["raw_df"] = df
            reset_state()
        except Exception as e:
            st.error(f"Read error: {e}")

    if "raw_df" not in st.session_state:
        return

    df = st.session_state["raw_df"]

    # overview
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Missing", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicates", f"{int(df.duplicated().sum()):,}")

    # preview
    t1,t2,t3 = st.tabs(["Preview","Types","Statistics"])
    with t1: show_df(df, 50, use_container_width=True)
    with t2:
        td = pd.DataFrame({"Column":df.columns,"Type":[str(d) for d in df.dtypes],
            "Non-Null":[int(df[c].notna().sum()) for c in df.columns],
            "Unique":[int(df[c].nunique()) for c in df.columns]})
        show_df(td, 80, use_container_width=True)
    with t3:
        desc = df.describe(include="all").round(3).reset_index().rename(columns={"index":"Stat"})
        show_df(desc, 30, use_container_width=True)

    # target
    st.header("Target Configuration")
    cl,cr = st.columns(2)
    with cl: target = st.selectbox("Column to predict", df.columns, key="tgt")
    with cr:
        task = infer_task(df, target)
        st.metric("Detected task", task)
    st.session_state["target"] = target
    st.session_state["task"] = task

    # optional goal
    with st.expander("Define your goal (optional)"):
        st.text_area("Describe what you want to predict or classify",
                     key="problem", placeholder="e.g. Predict customer churn from usage data.",
                     height=68, label_visibility="collapsed")


# ────────────────────────────────────────────────────────────
# PAGE: EXPLORE
# ────────────────────────────────────────────────────────────
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

    badge("Step 2 of 4")
    st.header("Exploratory Analysis")
    st.caption("Understand your data through interactive visualizations.")

    t_dist, t_corr, t_box, t_scatter, t_quality = st.tabs(
        ["Distributions", "Correlation", "Boxplots", "Scatter Matrix", "Quality"])

    with t_dist:
        if fn:
            sel = st.selectbox("Numeric feature", fn, key="d_f")
            st.plotly_chart(chart_layout(px.histogram(df, x=sel, nbins=40, marginal="box")),
                            use_container_width=True)
        # counts
        co = fc + ([target] if target not in fc else [])
        if co:
            sel2 = st.selectbox("Categorical feature", co, key="c_f")
            cts = df[sel2].astype(str).value_counts().head(20).reset_index()
            cts.columns = [sel2, "Count"]
            st.plotly_chart(chart_layout(px.bar(cts, x=sel2, y="Count")), use_container_width=True)

    with t_corr:
        if len(nc) >= 2:
            corr = df[nc].corr().round(2)
            fig = ff.create_annotated_heatmap(z=corr.values, x=corr.columns.tolist(),
                  y=corr.index.tolist(), colorscale="RdBu", showscale=True, zmin=-1, zmax=1)
            chart_layout(fig, min(640, 200+28*len(corr)))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need 2+ numeric columns.")

    with t_box:
        if fn:
            sel = st.selectbox("Feature", fn, key="b_f")
            if df[target].nunique() <= 15:
                bd = df[[sel,target]].copy(); bd[target] = bd[target].astype(str)
                fig = px.box(bd, x=target, y=sel, color=target)
            else:
                fig = px.box(df, y=sel)
            st.plotly_chart(chart_layout(fig), use_container_width=True)

    with t_scatter:
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
            chart_layout(fig, 540)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Top {n} features by target correlation. Sampled to 500 rows.")
        else:
            st.info("Need 2+ numeric features.")

    with t_quality:
        qt = quality_table(df, target)
        if not qt.empty: show_df(qt, 50, use_container_width=True)
        else: st.info("No numeric features.")


# ────────────────────────────────────────────────────────────
# PAGE: BUILD
# ────────────────────────────────────────────────────────────
def page_build():
    if "raw_df" not in st.session_state:
        st.info("Upload a dataset on the Data page first.")
        return
    if not pycaret_available():
        st.error("PyCaret is not installed. Run: pip install -r requirements.txt")
        return

    df = st.session_state["raw_df"]
    target = st.session_state.get("target")
    task = st.session_state.get("task")
    if not target:
        st.warning("Select a target on the Data page.")
        return

    badge("Step 3 of 4")
    st.header("Build Models")
    st.caption("Configure preprocessing, train multiple models, tune the best one.")

    task = st.radio("Task type", ["Classification","Regression"],
                    index=0 if task=="Classification" else 1, horizontal=True)
    st.session_state["task"] = task

    # options
    with st.expander("Preprocessing options"):
        c1,c2,c3 = st.columns(3)
        o_dup  = c1.checkbox("Drop duplicates", True, key="o1")
        o_out  = c2.checkbox("Remove outliers", False, key="o3")
        o_norm = c3.checkbox("Normalize", True, key="o4")
        c4,c5,c6 = st.columns(3)
        o_multi = c4.checkbox("Drop multicollinear", True, key="o7")
        o_trans = c5.checkbox("Transform skew", False, key="o5")
        o_fsel  = c6.checkbox("Feature selection", False, key="o6")
        c7,c8,c9 = st.columns(3)
        o_poly = c7.checkbox("Polynomial features", False, key="o8")
        o_imb  = c8.checkbox("Fix class imbalance", False, disabled=task!="Classification", key="o9")
        _ = c9.empty()
        c10,c11 = st.columns(2)
        o_test  = c10.slider("Test size", 0.10, 0.40, 0.20, 0.05, key="o10")
        o_folds = c11.slider("CV folds", 2, 10, 5, 1, key="o11")

    # train
    if st.button("Train All Models", type="primary", use_container_width=True):
        if df[target].nunique(dropna=True) < 2:
            st.error("Target needs 2+ unique values.")
        else:
            _train(df, target, task, dup=o_dup, out=o_out,
                   norm=o_norm, trans=o_trans, fsel=o_fsel, multi=o_multi,
                   poly=o_poly, imb=o_imb, test=o_test, folds=o_folds)

    if not st.session_state.get("setup_done"):
        return

    exp = st.session_state["experiment"]

    # results
    if "compare_df" in st.session_state:
        st.header("Leaderboard")
        cdf = st.session_state["compare_df"]
        show_df(cdf, 30, use_container_width=True)
        mc = [c for c in cdf.columns if c not in ("Model","TT (Sec)")]
        if mc and "Model" in cdf.columns:
            m = st.selectbox("Compare metric", mc, key="cm")
            st.plotly_chart(chart_layout(px.bar(cdf, x="Model", y=m, color="Model")),
                            use_container_width=True)

    # tuning
    if "best_model" in st.session_state:
        st.header("Hyperparameter Tuning")
        st.caption(f"Best model: {type(st.session_state['active_model']).__name__}")
        t1,t2 = st.columns(2)
        sm = {"Bayesian (Optuna)":("optuna",None),
              "Random Search":("scikit-learn","random"),
              "Grid Search":("scikit-learn","grid")}
        sl = t1.selectbox("Search method", list(sm.keys()), key="ts")
        ni = t2.slider("Iterations", 5, 50, 10, 5, key="tn")
        slib, salg = sm[sl]
        if st.button("Tune Model", use_container_width=True):
            _tune(exp, task, slib, salg, ni)
        if "tune_df" in st.session_state:
            with st.expander("Tuning results"):
                show_df(st.session_state["tune_df"], 30, use_container_width=True)

    # evaluation
    if "active_model" in st.session_state:
        st.header("Evaluation")
        plots = st.session_state.get("plot_paths", {})
        if plots:
            items = [(lb,p) for lb,p in plots.items() if p and os.path.exists(p)]
            cols = st.columns(2)
            for i,(lb,p) in enumerate(items):
                with cols[i%2]:
                    st.markdown(f"**{lb}**")
                    st.image(p, use_container_width=True)
        if "predictions" in st.session_state:
            with st.expander(f"Holdout predictions ({len(st.session_state['predictions'])} rows)"):
                show_df(st.session_state["predictions"], 100, use_container_width=True)

    if "setup_df" in st.session_state:
        with st.expander("Setup details"):
            show_df(st.session_state["setup_df"], 80, use_container_width=True)


def _train(df, target, task, **o):
    work = df.copy()
    if o.get("dup"):
        n0 = len(work); work = work.drop_duplicates()
        if n0-len(work): st.info(f"Removed {n0-len(work):,} duplicates.")

    with st.spinner("Setting up experiment..."):
        try:
            exp = get_exp(task)
            kw = {"data":work,"target":target,"session_id":42,
                  "train_size":1-o.get("test",0.2),"fold":o.get("folds",5),
                  "normalize":o.get("norm"),"transformation":o.get("trans"),
                  "remove_multicollinearity":o.get("multi"),
                  "multicollinearity_threshold":0.9,
                  "feature_selection":o.get("fsel"),
                  "remove_outliers":o.get("out"),
                  "outliers_threshold":0.05,
                  "polynomial_features":o.get("poly"),
                  "html":False,"verbose":False}
            if task=="Classification": kw["fix_imbalance"]=o.get("imb",False)
            exp.setup(**kw)
            st.session_state.update({"experiment":exp,"setup_df":exp.pull(),"setup_done":True})
        except Exception as e:
            st.error(f"Setup failed: {e}"); return

    with st.spinner("Training and comparing models..."):
        try:
            best = exp.compare_models()
            st.session_state.update({"compare_df":norm_cmp(exp.pull()),"best_model":best,"active_model":best})
            st.session_state.pop("tuned_model",None); st.session_state.pop("tune_df",None)
        except Exception as e:
            st.error(f"Training failed: {e}"); return

    with st.spinner("Generating evaluation plots..."):
        _gen_plots(exp, best, task)
    with st.spinner("Running holdout predictions..."):
        try: st.session_state["predictions"] = exp.predict_model(best)
        except: pass
    st.rerun()


def _tune(exp, task, slib, salg, n_iter):
    best = st.session_state["best_model"]
    with st.spinner("Tuning hyperparameters..."):
        try:
            kw = {"estimator":best,"search_library":slib,"n_iter":n_iter,"choose_better":True}
            if salg: kw["search_algorithm"] = salg
            tuned = exp.tune_model(**kw)
            st.session_state.update({"tuned_model":tuned,"active_model":tuned,"tune_df":exp.pull()})
        except Exception as e:
            st.error(f"Tuning failed: {e}"); return
    with st.spinner("Updating evaluation..."):
        _gen_plots(exp, tuned, task)
    try: st.session_state["predictions"] = exp.predict_model(tuned)
    except: pass
    st.rerun()


def _gen_plots(exp, model, task):
    plots = {}
    pm = ([("confusion_matrix","Confusion Matrix"),("auc","ROC / AUC"),
           ("pr","Precision-Recall"),("feature","Feature Importance"),
           ("class_report","Classification Report")]
          if task=="Classification"
          else [("residuals","Residuals"),("error","Predicted vs Actual"),
                ("feature","Feature Importance")])
    for pk,lb in pm:
        p = grab_plot(exp, model, pk)
        if p: plots[lb] = p
    for st_t,lb in [("summary","SHAP Summary"),("correlation","SHAP Correlation")]:
        p = grab_shap(exp, model, st_t)
        if p: plots[lb] = p
    st.session_state["plot_paths"] = plots


# ────────────────────────────────────────────────────────────
# PAGE: EXPORT
# ────────────────────────────────────────────────────────────
def page_export():
    if "active_model" not in st.session_state:
        st.info("Train a model on the Build page first.")
        return

    exp = st.session_state["experiment"]
    model = st.session_state["active_model"]
    name = type(model).__name__

    badge("Step 4 of 4")
    st.header("Export Results")
    st.caption(f"Your best model: **{name}**")

    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            final = exp.finalize_model(model)
            exp.save_model(final, MODEL_PATH)
            mp = f"{MODEL_PATH}.pkl"
            if os.path.exists(mp):
                with open(mp,"rb") as f:
                    st.download_button("Pipeline (.pkl)", f.read(),
                        "automl_pipeline.pkl","application/octet-stream", use_container_width=True)
        except Exception as e:
            st.error(f"Save failed: {e}")

    with col2:
        if "compare_df" in st.session_state:
            st.download_button("Leaderboard (.csv)",
                st.session_state["compare_df"].to_csv(index=False).encode(),
                "leaderboard.csv", use_container_width=True)

    with col3:
        if "predictions" in st.session_state:
            st.download_button("Predictions (.csv)",
                st.session_state["predictions"].to_csv(index=False).encode(),
                "predictions.csv", use_container_width=True)

    # summary
    prob = st.session_state.get("problem","")
    st.header("Summary")
    st.markdown(f"""
| | |
|---|---|
| **Target** | `{st.session_state.get('target','-')}` |
| **Task** | {st.session_state.get('task','-')} |
| **Best Model** | {name} |
| **Exported** | {datetime.now().strftime('%Y-%m-%d %H:%M')} |
""" + (f"| **Goal** | {prob} |" if prob else ""))


# ────────────────────────────────────────────────────────────
# APP SHELL
# ────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state: st.session_state["dark_mode"] = True
if "active_page" not in st.session_state: st.session_state["active_page"] = PAGES[0]
if st.session_state.get("active_page") not in PAGES: st.session_state["active_page"] = PAGES[0]

tp = st.query_params.get("theme","dark")
if isinstance(tp,list): tp=tp[0]
st.session_state["dark_mode"] = tp!="light"
apply_css(st.session_state["dark_mode"])

px.defaults.template = "plotly_dark" if st.session_state["dark_mode"] else "plotly_white"
px.defaults.color_discrete_sequence = [
    "#10b981","#6366f1","#f59e0b","#ef4444","#8b5cf6",
    "#06b6d4","#84cc16","#ec4899","#f97316","#14b8a6"]

# header
st.markdown("""<div class="brand"><h1>AutoML Studio</h1>
<p>End-to-end machine learning pipeline</p></div>""", unsafe_allow_html=True)

nc,tc = st.columns([0.86,0.14], vertical_alignment="center")
with nc:
    sel = st.segmented_control("n", PAGES, default=st.session_state["active_page"],
                               label_visibility="collapsed")
    if sel: st.session_state["active_page"] = sel
with tc:
    nt = "light" if st.session_state["dark_mode"] else "dark"
    tl = "Light" if st.session_state["dark_mode"] else "Dark"
    st.markdown(f'<a class="tgl" href="?theme={nt}" target="_self">{tl}</a>', unsafe_allow_html=True)

pg = st.session_state["active_page"]
{"Data":page_data,"Explore":page_explore,"Build":page_build,"Export":page_export}[pg]()
