"""
FastAPI web interface — streaming SSE endpoint for analysis,
REST endpoints for session management.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import get_settings
from ..engine import AnalystAgent
from ..engine.session import AnalysisSession
from ..ingestion import DataLoader, DataProfiler, QualityScorer
from ..logging_config import get_logger, configure_logging
from ..memory import MemoryStore

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    configure_logging(get_settings().log_level)
    cfg = get_settings()

    app = FastAPI(
        title="Analyst Agent API",
        description="Autonomous Data Analyst — Claude-powered",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    memory = MemoryStore(persist_dir=cfg.chroma_path)
    agent = AnalystAgent(memory=memory)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    class AnalyseRequest(BaseModel):
        source: str
        question: str
        resume_session_id: str | None = None

    class ChatRequest(BaseModel):
        message: str

    class ProfileRequest(BaseModel):
        source: str

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return """
        <h1>Analyst Agent API</h1>
        <p>See <a href="/docs">/docs</a> for the API reference.</p>
        """

    @app.post("/analyse/stream")
    async def analyse_stream(req: AnalyseRequest) -> StreamingResponse:
        """Stream analysis progress as Server-Sent Events."""
        async def event_stream() -> AsyncGenerator[str, None]:
            loop = asyncio.get_event_loop()
            # Run synchronous generator in thread pool to avoid blocking
            for event in agent.analyse(req.source, req.question, req.resume_session_id):
                data = json.dumps({"event": event})
                yield f"data: {data}\n\n"
                await asyncio.sleep(0)  # yield control
            yield "data: {\"event\": \"[DONE]\"}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/chat")
    async def chat_endpoint(req: ChatRequest) -> dict[str, str]:
        """Send a follow-up to the active session. Returns full response."""
        session = agent.last_session()
        if session is None:
            raise HTTPException(400, "No active session. Call /analyse/stream first.")
        responses = list(agent.chat(req.message))
        return {"response": "\n".join(responses)}

    @app.post("/profile")
    async def profile_endpoint(req: ProfileRequest) -> dict:
        """Profile a dataset and return quality metadata."""
        cfg = get_settings()
        loader = DataLoader(database_url=cfg.database_url)
        profiler = DataProfiler()
        scorer = QualityScorer()
        df, source_label = loader.load(req.source)
        schema = scorer.score(profiler.profile(df, source_label))
        return schema.to_dict()

    @app.post("/upload")
    async def upload_file(file: UploadFile) -> dict[str, str]:
        """Upload a CSV/Excel/Parquet file for analysis."""
        cfg = get_settings()
        dest = cfg.workspace_dir / "uploads" / (file.filename or "upload")
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        dest.write_bytes(content)
        return {"path": str(dest), "filename": file.filename, "size_bytes": len(content)}

    @app.get("/sessions")
    async def list_sessions() -> list[dict]:
        cfg = get_settings()
        session_files = sorted(
            Path(cfg.session_dir).glob("session_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        results = []
        for sf in session_files[:50]:
            try:
                s = AnalysisSession.load(sf)
                results.append({
                    "session_id": s.session_id,
                    "question": s.business_question[:80],
                    "status": s.status,
                    "quality_score": s.quality_score,
                    "findings_count": len(s.findings),
                    "created_at": s.created_at,
                })
            except Exception:
                pass
        return results

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> dict:
        cfg = get_settings()
        matches = list(Path(cfg.session_dir).glob(f"session_{session_id}.json"))
        if not matches:
            raise HTTPException(404, f"Session {session_id} not found")
        from dataclasses import asdict
        s = AnalysisSession.load(matches[0])
        return asdict(s)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": get_settings().analyst_model}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("analyst_agent.interface.api:app", host="0.0.0.0", port=8000, reload=True)
