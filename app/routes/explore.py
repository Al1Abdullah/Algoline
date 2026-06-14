from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, plotly.figure_factory as ff
from app.state import S
from app.helpers import fig_json, safe_float

router = APIRouter()

# ── Explore endpoints ──

@router.post("/api/explore/distribution")
def explore_dist(feature: str = Form(...)):
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    fig = px.histogram(S["df"], x=feature, nbins=40, marginal="box",
                       color_discrete_sequence=["#6366f1"])
    fig.update_layout(title=f"Distribution: {feature}", title_font_size=14)
    return {"figure": fig_json(fig)}

@router.post("/api/explore/counts")
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

@router.post("/api/explore/correlation")
def explore_corr():
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    nc = S["df"].select_dtypes(include=np.number).columns.tolist()
    if len(nc) < 2: return {"figure": None}
    corr = S["df"][nc].corr().round(2)
    fig = ff.create_annotated_heatmap(z=corr.values, x=corr.columns.tolist(),
          y=corr.index.tolist(), colorscale="RdBu", showscale=True, zmin=-1, zmax=1)
    fig.update_layout(title="Correlation Matrix", title_font_size=14)
    return {"figure": fig_json(fig, min(640, 200 + 28 * len(corr)))}

@router.post("/api/explore/boxplot")
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

@router.post("/api/explore/scatter")
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

@router.post("/api/explore/quality")
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
                      "Unique":int(df[c].nunique()), "Mean":safe_float(s.mean()),
                      "Std":safe_float(s.std()), "Skew":safe_float(s.skew()),
                      "Outliers":out})
    return {"rows": rows, "columns": ["Feature","Missing","Missing%","Unique","Mean","Std","Skew","Outliers"]}


@router.post("/api/explore/target")
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


@router.post("/api/explore/missing")
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


@router.post("/api/explore/feature_target")
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
                         color_discrete_sequence=["#22d3ee"], opacity=0.6)
        fig.update_layout(title=f"{feature} vs {tgt}", title_font_size=14)
    return {"figure": fig_json(fig)}


@router.post("/api/explore/outliers")
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

@router.post("/api/explore/kde")
def explore_kde(feature: str = Form(...)):
    """KDE (Kernel Density Estimation) plot."""
    if "df" not in S: return JSONResponse({"error": "No data"}, 400)
    try:
        s = S["df"][feature].dropna()
        if not pd.api.types.is_numeric_dtype(s):
            # For categorical: show value counts as bar
            vc = s.astype(str).value_counts().head(20).reset_index()
            vc.columns = [feature, "Count"]
            fig = px.bar(vc, x=feature, y="Count", color_discrete_sequence=["#6366f1"])
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


@router.post("/api/explore/violin")
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
                            color_discrete_sequence=["#22d3ee"])
        fig.update_layout(title=f"Violin: {feature}", title_font_size=14)
        return {"figure": fig_json(fig)}
    except Exception as e:
        return JSONResponse({"error": f"Violin failed: {str(e)}"}, 400)


@router.post("/api/explore/missing_heatmap")
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


@router.post("/api/explore/pairplot")
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


@router.post("/api/explore/scatter_xy")
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


@router.post("/api/explore/jointplot")
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


@router.post("/api/explore/countplot")
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


@router.post("/api/explore/pie")
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


@router.post("/api/explore/scatter_index")
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
                         color_discrete_sequence=["#6366f1"], opacity=0.5)
        fig.update_layout(title=f"{feature} vs Row Index", title_font_size=14,
                          xaxis_title="Row Index")
        return {"figure": fig_json(fig)}
    except Exception as e:
        return JSONResponse({"error": f"Scatter index failed: {str(e)}"}, 400)


@router.post("/api/explore/mean_target")
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
            fig = px.bar(grp, x=feature, y=f"Mean {tgt}", color_discrete_sequence=["#a78bfa"])
            fig.update_layout(title=f"Mean {tgt} per {feature}", title_font_size=14,
                              xaxis_tickangle=-45)
        return {"figure": fig_json(fig)}
    except Exception as e:
        return JSONResponse({"error": f"Mean target failed: {str(e)}"}, 400)


@router.post("/api/explore/grouped_box")
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


@router.post("/api/explore/facetgrid")
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
