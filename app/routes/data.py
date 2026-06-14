import io
import numpy as np
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse

from app.state import S
from app.helpers import safe_json, infer_task

router = APIRouter()


@router.get("/")
def index():
    return FileResponse("static/index.html")


@router.post("/api/upload")
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

    # ── Auto Insights ──
    insights = []
    target = S["target"]
    task = S["task"]

    # 1. Class imbalance check (classification only)
    if task == "classification":
        vc = df[target].value_counts(normalize=True)
        if len(vc) >= 2 and vc.iloc[0] > 0.8:
            insights.append({"type": "warning", "title": "Class Imbalance Detected",
                "detail": f"'{target}' is {vc.iloc[0]*100:.0f}% class '{vc.index[0]}'. Consider enabling 'Fix Class Imbalance' during training."})
        elif len(vc) >= 2:
            insights.append({"type": "success", "title": "Balanced Classes",
                "detail": f"Target '{target}' has {len(vc)} classes with reasonable balance."})

    # 2. High missing columns
    miss_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    high_miss = miss_pct[miss_pct > 30]
    if len(high_miss) > 0:
        cols_str = ", ".join([f"{c} ({v:.0f}%)" for c, v in high_miss.head(3).items()])
        insights.append({"type": "warning", "title": f"{len(high_miss)} Feature(s) with High Missing Values",
            "detail": f"{cols_str}. These may reduce model performance."})
    elif miss == 0:
        insights.append({"type": "success", "title": "No Missing Values",
            "detail": "Dataset is complete with zero missing entries."})

    # 3. High cardinality categoricals
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    high_card = [(c, df[c].nunique()) for c in cat_cols if df[c].nunique() > 50]
    if high_card:
        cols_str = ", ".join([f"{c} ({n} unique)" for c, n in high_card[:3]])
        insights.append({"type": "warning", "title": "High Cardinality Features",
            "detail": f"{cols_str}. Consider dropping ID-like columns before training."})

    # 4. Highly correlated features
    if len(nc) >= 2:
        try:
            corr = df[nc].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            high_corr = [(upper.columns[j], upper.index[i], upper.iloc[i, j])
                         for i in range(len(upper)) for j in range(len(upper.columns))
                         if upper.iloc[i, j] > 0.9]
            if high_corr:
                pair = high_corr[0]
                insights.append({"type": "info", "title": f"{len(high_corr)} Highly Correlated Pair(s)",
                    "detail": f"e.g. '{pair[0]}' and '{pair[1]}' ({pair[2]:.2f}). Enable 'Drop Multicollinear' to handle this."})
        except Exception:
            pass

    # 5. Constant columns
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if const_cols:
        insights.append({"type": "warning", "title": f"{len(const_cols)} Constant Column(s)",
            "detail": f"{', '.join(const_cols[:3])}. These provide no predictive value."})

    # 6. Dataset size summary
    insights.append({"type": "info", "title": "Dataset Overview",
        "detail": f"{len(df):,} rows, {len(nc)} numeric and {len(cat_cols)} categorical features. Task: {task}."})

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
        "insights": insights,
    }


@router.post("/api/target")
def set_target(target: str = Form(...)):
    if "df" not in S:
        return JSONResponse({"error": "No data loaded"}, 400)
    S["target"] = target
    S["task"] = infer_task(S["df"], target)
    # Clear stale training state when target changes
    for k in ["experiment","compare_df","best_model","active_model","tuned_model","tune_df","plots"]:
        S.pop(k, None)
    return {"target": target, "task": S["task"]}
