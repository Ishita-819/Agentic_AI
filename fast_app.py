from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import logging

# import your orchestration functions
from AI_Orchestrator_2 import (
    run_analysis_for_web,
    run_selected_next_step,
    load_complete_results
)

app = FastAPI(title="Agentic scRNA-seq UI")

# Allow frontend JS to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend `index1.html` at the root and mount static files
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/")
def read_index():
    index_path = Path(__file__).resolve().parent / "index1.html"
    if not index_path.exists():
        logging.error(f"index1.html not found at {index_path}")
        return {"error": "index1.html not found on server"}
    return FileResponse(str(index_path))

# ---------------- GLOBAL CONTEXT (single-user prototype) ----------------
GLOBAL_CONTEXT = {}

# Simple job store for background analyses (single-node prototype)
JOBS: dict = {}

def _run_analysis_job(job_id: str):
    """Background wrapper that runs the analysis and stores results in JOBS."""
    try:
        JOBS[job_id] = {"status": "running", "result": None}

        # Run pipeline and load results
        result = run_analysis_for_web()
        bio_text, deg_df, traj_df = load_complete_results()

        # Store context so run-next-step can work later
        GLOBAL_CONTEXT["bio_text"] = bio_text
        GLOBAL_CONTEXT["deg_df"] = deg_df
        GLOBAL_CONTEXT["traj_df"] = traj_df

        JOBS[job_id]["status"] = "finished"
        JOBS[job_id]["result"] = {
         "result":result,
         "bio_text_len": len(bio_text) if isinstance(bio_text, str) else None
        }
    except Exception as e:
        logging.exception("Analysis job failed")
        JOBS[job_id] = {"status": "failed", "error": str(e)}


class NextStepRequest(BaseModel):
    suggestion: str


@app.post("/run-analysis")
def run_analysis(background_tasks: BackgroundTasks):
    """Start the full pipeline in the background and return a job id immediately.

    Use GET /status/{job_id} and GET /result/{job_id} to poll for progress and results.
    """
    import uuid

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "result": None}
    background_tasks.add_task(_run_analysis_job, job_id)
    return {"job_id": job_id, "status": "queued"}


@app.post("/run-next-step")
def run_next_step(req: NextStepRequest):
    """
    Executes selected next step
    """
    if not GLOBAL_CONTEXT:
        return {"error": "No active analysis context. Run analysis first."}

    return run_selected_next_step(req.suggestion, GLOBAL_CONTEXT)


@app.get("/status/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return {"error": "job not found"}
    return {"job_id": job_id, "status": job.get("status")}


@app.get("/result/{job_id}")
def job_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return {"error": "job not found"}
    if job.get("status") != "finished":
        return {"job_id": job_id, "status": job.get("status"), "result": None}
    return {"job_id": job_id, "status": "finished", "result": job.get("result")}


@app.get("/health")
def health():
    return {"status": "ok"}
