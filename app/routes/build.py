from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
import os, gc, numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go

from app.state import S
from app.helpers import safe_json, fig_json, norm_lb, get_exp

router = APIRouter()


def gen_all_plots(exp, model, task):
    """Generate REAL evaluation plots from actual model predictions using Plotly."""
    plots = []
    try:
        # Get actual predictions from the model on the test set
        preds = exp.predict_model(model)
    except Exception:
        return plots

    target_col = exp.get_config('target_param') if hasattr(exp, 'get_config') else None

    if task == "classification":
        # --- CONFUSION MATRIX (real data) ---
        try:
            from sklearn.metrics import confusion_matrix
            y_true = preds[target_col] if target_col else preds.iloc[:, -2]
            y_pred = preds['prediction_label'] if 'prediction_label' in preds.columns else preds['Label']
            labels = sorted(y_true.unique())
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            labels_str = [str(l) for l in labels]
            # Annotate with counts
            annotations = []
            for i, row in enumerate(cm):
                for j, val in enumerate(row):
                    annotations.append(dict(
                        x=labels_str[j], y=labels_str[i],
                        text=str(val), showarrow=False,
                        font=dict(color='white' if val > cm.max()/2 else '#a1a1aa', size=14)
                    ))
            fig = go.Figure(data=go.Heatmap(
                z=cm, x=labels_str, y=labels_str,
                colorscale=[[0,'#01243a'],[0.5,'#1e1b4b'],[1,'#6366f1']],
                showscale=True, hoverongaps=False
            ))
            fig.update_layout(
                title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual',
                yaxis=dict(autorange='reversed'), annotations=annotations,
                height=420
            )
            plots.append({"label": "Confusion Matrix", "figure": fig_json(fig)})
        except Exception:
            pass

        # --- ROC CURVE (real data) ---
        try:
            from sklearn.metrics import roc_curve, auc
            y_true = preds[target_col] if target_col else preds.iloc[:, -2]
            # Try to get prediction scores
            score_cols = [c for c in preds.columns if c.startswith('prediction_score')]
            if len(score_cols) >= 2 and y_true.nunique() == 2:
                labels = sorted(y_true.unique())
                y_binary = (y_true == labels[1]).astype(int)
                scores = preds[score_cols[-1]]
                fpr, tpr, _ = roc_curve(y_binary, scores)
                roc_auc = auc(fpr, tpr)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                    name=f'ROC (AUC={roc_auc:.3f})', line=dict(color='#6366f1', width=2)))
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                    name='Random', line=dict(color='#71717a', dash='dash', width=1)))
                fig.update_layout(title=f'ROC Curve (AUC = {roc_auc:.3f})',
                    xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                    height=420)
                plots.append({"label": "ROC Curve", "figure": fig_json(fig)})
        except Exception:
            pass

        # --- PRECISION-RECALL (real data) ---
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            y_true = preds[target_col] if target_col else preds.iloc[:, -2]
            score_cols = [c for c in preds.columns if c.startswith('prediction_score')]
            if len(score_cols) >= 2 and y_true.nunique() == 2:
                labels = sorted(y_true.unique())
                y_binary = (y_true == labels[1]).astype(int)
                scores = preds[score_cols[-1]]
                prec, rec, _ = precision_recall_curve(y_binary, scores)
                ap = average_precision_score(y_binary, scores)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rec, y=prec, mode='lines',
                    name=f'PR (AP={ap:.3f})', line=dict(color='#818cf8', width=2), fill='tozeroy',
                    fillcolor='rgba(139,92,246,0.1)'))
                fig.update_layout(title=f'Precision-Recall Curve (AP = {ap:.3f})',
                    xaxis_title='Recall', yaxis_title='Precision', height=420)
                plots.append({"label": "Precision-Recall", "figure": fig_json(fig)})
        except Exception:
            pass

        # --- CLASS DISTRIBUTION OF PREDICTIONS ---
        try:
            y_pred = preds['prediction_label'] if 'prediction_label' in preds.columns else preds['Label']
            counts = y_pred.value_counts()
            fig = px.bar(x=counts.index.astype(str), y=counts.values,
                         color=counts.index.astype(str),
                         color_discrete_sequence=['#6366f1','#818cf8','#f59e0b','#ef4444','#3b82f6'])
            fig.update_layout(title='Prediction Distribution',
                xaxis_title='Class', yaxis_title='Count',
                showlegend=False, height=380)
            plots.append({"label": "Prediction Distribution", "figure": fig_json(fig)})
        except Exception:
            pass

    else:  # regression
        # --- RESIDUALS PLOT ---
        try:
            y_true = preds[target_col] if target_col else preds.iloc[:, -2]
            y_pred = preds['prediction_label'] if 'prediction_label' in preds.columns else preds['Label']
            residuals = y_true - y_pred
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                marker=dict(color='#6366f1', size=5, opacity=0.6), name='Residuals'))
            fig.add_hline(y=0, line_dash="dash", line_color="#71717a")
            fig.update_layout(title='Residuals Plot',
                xaxis_title='Predicted', yaxis_title='Residual', height=420)
            plots.append({"label": "Residuals", "figure": fig_json(fig)})
        except Exception:
            pass

        # --- PREDICTED vs ACTUAL ---
        try:
            y_true = preds[target_col] if target_col else preds.iloc[:, -2]
            y_pred = preds['prediction_label'] if 'prediction_label' in preds.columns else preds['Label']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers',
                marker=dict(color='#818cf8', size=5, opacity=0.6), name='Predictions'))
            mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines',
                name='Perfect', line=dict(color='#6366f1', dash='dash', width=2)))
            fig.update_layout(title='Predicted vs Actual',
                xaxis_title='Actual', yaxis_title='Predicted', height=420)
            plots.append({"label": "Predicted vs Actual", "figure": fig_json(fig)})
        except Exception:
            pass

        # --- RESIDUAL DISTRIBUTION ---
        try:
            y_true = preds[target_col] if target_col else preds.iloc[:, -2]
            y_pred = preds['prediction_label'] if 'prediction_label' in preds.columns else preds['Label']
            residuals = y_true - y_pred
            fig = go.Figure(data=go.Histogram(x=residuals, nbinsx=40,
                marker_color='#6366f1', opacity=0.8,
                marker_line=dict(width=1, color='rgba(255,255,255,0.15)')))
            fig.update_layout(title='Residual Distribution',
                xaxis_title='Residual', yaxis_title='Count', height=380)
            plots.append({"label": "Residual Distribution", "figure": fig_json(fig)})
        except Exception:
            pass

    # --- FEATURE IMPORTANCE (works for both tasks) ---
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = exp.get_config('X_train').columns.tolist()
            imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            imp_df = imp_df.sort_values('Importance', ascending=True).tail(15)
            fig = go.Figure(go.Bar(x=imp_df['Importance'], y=imp_df['Feature'],
                orientation='h', marker_color='#6366f1',
                marker_line=dict(width=1, color='rgba(255,255,255,0.12)')))
            fig.update_layout(title='Feature Importance (Top 15)',
                xaxis_title='Importance', height=max(350, len(imp_df)*28 + 80),
                margin=dict(l=160))
            plots.append({"label": "Feature Importance", "figure": fig_json(fig)})
        elif hasattr(model, 'coef_'):
            coefs = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
            feature_names = exp.get_config('X_train').columns.tolist()
            if len(coefs) == len(feature_names):
                imp_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
                imp_df['AbsCoef'] = imp_df['Coefficient'].abs()
                imp_df = imp_df.sort_values('AbsCoef', ascending=True).tail(15)
                colors = ['#6366f1' if v >= 0 else '#ef4444' for v in imp_df['Coefficient']]
                fig = go.Figure(go.Bar(x=imp_df['Coefficient'], y=imp_df['Feature'],
                    orientation='h', marker_color=colors,
                    marker_line=dict(width=1, color='rgba(255,255,255,0.12)')))
                fig.update_layout(title='Feature Coefficients (Top 15)',
                    xaxis_title='Coefficient', height=max(350, len(imp_df)*28 + 80),
                    margin=dict(l=160))
                plots.append({"label": "Feature Coefficients", "figure": fig_json(fig)})
    except Exception:
        pass

    return plots


# ── Build endpoints ──

@router.post("/api/train")
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

    # ── Training Summary ──
    model_name = type(best).__name__
    n_models = len(cdf)
    key_metric_name = metrics[0] if metrics else ""
    key_metric_val = float(cdf[key_metric_name].iloc[0]) if key_metric_name and len(cdf) > 0 else 0

    # Top features
    top_features = []
    try:
        if hasattr(best, 'feature_importances_'):
            feat_names = exp.get_config('X_train').columns.tolist()
            imp = best.feature_importances_
            top_idx = np.argsort(imp)[::-1][:3]
            top_features = [feat_names[i] for i in top_idx if i < len(feat_names)]
        elif hasattr(best, 'coef_'):
            feat_names = exp.get_config('X_train').columns.tolist()
            coefs = np.abs(best.coef_.flatten() if best.coef_.ndim > 1 else best.coef_)
            top_idx = np.argsort(coefs)[::-1][:3]
            top_features = [feat_names[i] for i in top_idx if i < len(feat_names)]
    except Exception:
        pass

    # Natural language summary
    summary_text = f"{model_name} achieved {key_metric_val:.4f} {key_metric_name}"
    if top_features:
        summary_text += f". Most predictive features: {', '.join(top_features)}"
    summary_text += f". Compared {n_models} models on {len(train_df):,} training rows."

    summary = {
        "model": model_name,
        "n_models": n_models,
        "key_metric": key_metric_name,
        "key_metric_value": round(key_metric_val, 4),
        "top_features": top_features,
        "text": summary_text,
    }

    return {
        "model_name": model_name,
        "duplicates_removed": removed,
        "sampled": sampled,
        "train_rows": len(train_df),
        "original_rows": original_rows,
        "leaderboard": {"rows": lb_rows, "columns": lb_cols},
        "metrics": metrics,
        "comparison_chart": comp_fig,
        "plots": S.get("plots", []),
        "n_predictions": len(S.get("predictions", [])),
        "summary": summary,
    }


@router.post("/api/compare")
def compare_metric(metric: str = Form(...)):
    if "compare_df" not in S: return JSONResponse({"error": "No leaderboard"}, 400)
    cdf = S["compare_df"]
    if "Model" not in cdf.columns or metric not in cdf.columns:
        return {"figure": None}
    fig = px.bar(cdf, x="Model", y=metric, color="Model")
    fig.update_layout(title=f"Model Comparison: {metric}", title_font_size=14)
    return {"figure": fig_json(fig)}


@router.post("/api/tune")
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
