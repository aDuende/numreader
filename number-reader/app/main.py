from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import sys
import os

from app.routes.predict import router as predict_router
from app.routes.collect import router as collect_router
from app.routes.models import router as models_router
from app.services import model_service

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        os.environ.get("FRONTEND_URL", ""),
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(collect_router)
app.include_router(models_router)

_retrain_status = {"running": False, "last_result": None}


def _run_retrain():
    _retrain_status["running"] = True
    _retrain_status["last_result"] = None
    try:
        result = subprocess.run(
            [sys.executable, "app/model/train_test_model.py"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        if result.returncode == 0:
            try:
                model_service.reload_model()
            except Exception as e:
                print(f"[retrain] reload failed: {e}")
        _retrain_status["last_result"] = {
            "success": result.returncode == 0,
            "output": result.stdout[-2000:] if result.stdout else "",
            "error": result.stderr[-1000:] if result.stderr else "",
        }
    except Exception as e:
        _retrain_status["last_result"] = {"success": False, "error": str(e)}
    finally:
        _retrain_status["running"] = False


@app.post("/api/retrain")
def retrain(background_tasks: BackgroundTasks):
    if _retrain_status["running"]:
        return {"message": "retrain already running"}
    background_tasks.add_task(_run_retrain)
    return {"message": "retrain started"}


@app.get("/api/retrain/status")
def retrain_status():
    return _retrain_status


@app.get("/")
def root():
    return {"message": "AI Number Reader Backend Running"}