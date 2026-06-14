import os
import tempfile

from fastapi import APIRouter
from fastapi.responses import JSONResponse, FileResponse

from app.state import S

router = APIRouter()

# ── Export endpoints ──

@router.get("/api/export/pipeline")
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

@router.get("/api/export/leaderboard")
def export_leaderboard():
    if "compare_df" not in S:
        return JSONResponse({"error": "No leaderboard"}, 400)
    path = os.path.join(tempfile.gettempdir(), "leaderboard.csv")
    S["compare_df"].to_csv(path, index=False)
    return FileResponse(path, filename="leaderboard.csv")

@router.get("/api/export/predictions")
def export_predictions():
    if "predictions" not in S:
        return JSONResponse({"error": "No predictions"}, 400)
    path = os.path.join(tempfile.gettempdir(), "predictions.csv")
    S["predictions"].to_csv(path, index=False)
    return FileResponse(path, filename="predictions.csv")

@router.get("/api/summary")
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
