#!/usr/bin/env python3
"""
web/app.py
FastAPI web application for Dawah Dialectics V2 (yt-spx):
Speaker-Diarized Claim-and-Proof Dossiers, Full Scenario Transcript Studio & Smart Assistant.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from db.database import (
    init_db,
    search_qa,
    count_qa,
    get_categories_breakdown,
    get_interlocutor_breakdown,
    get_speakers_breakdown,
    list_videos,
    get_video_scenario,
    search_scenario_turns,
    get_stats,
)
from core.smart_bot import smart_bot

# Ensure V2 database schema and Dual-FTS5 indices are synchronized on startup
init_db()

app = FastAPI(
    title="Dawah Dialectics V2 — Diarized Scenario & Argument Archive",
    version="2.0.0",
)

static_dir = BASE_DIR / "web" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
def index_page():
    template_path = BASE_DIR / "web" / "templates" / "index.html"
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Template not found")
    return template_path.read_text(encoding="utf-8")


@app.get("/api/search")
def api_search(
    q: str = Query("", description="Search term in English or Arabic"),
    category: str = Query("all", description="Topic category filter"),
    interlocutor: str = Query("all", description="Interlocutor archetype filter"),
    speaker: str = Query("all", description="Da'ee speaker profile or display name filter"),
    video_id: str = Query("all", description="Specific YouTube video ID filter"),
    sort: str = Query("default", description="Sort order: default, timestamp_asc, timestamp_desc, category"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Bilingual hybrid FTS5 & multi-facet dossier search endpoint."""
    total = count_qa(
        query=q,
        category=category,
        interlocutor=interlocutor,
        speaker=speaker,
        video_id=video_id,
    )
    results = search_qa(
        query=q,
        category=category,
        interlocutor=interlocutor,
        speaker=speaker,
        video_id=video_id,
        limit=limit,
        offset=offset,
        sort=sort,
    )
    return {
        "results": results,
        "count": len(results),
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@app.get("/api/chat")
def api_chat(q: str = Query(..., description="User query for the smart dialectical assistant")):
    """Zero-LLM instant dialectical assistant endpoint with scenario turn fallback."""
    return smart_bot.answer_query(q)


@app.get("/api/categories")
def api_categories():
    """Returns topic pillar breakdown with counts."""
    return {"categories": get_categories_breakdown()}


@app.get("/api/interlocutors")
def api_interlocutors():
    """Returns interlocutor archetype breakdown with counts."""
    return {"interlocutors": get_interlocutor_breakdown()}


@app.get("/api/speakers")
def api_speakers():
    """Returns identified Da'ee voiceprint speaker breakdown across dossiers and diarized videos."""
    return {"speakers": get_speakers_breakdown()}


@app.get("/api/videos")
def api_videos(
    q: str = Query("", description="Filter cataloged videos by Arabic title or YouTube ID"),
    speaker: str = Query("all", description="Filter by Da'ee profile ID or name"),
    only_diarized: bool = Query(False, description="Return only videos with completed Stage 1 scenarios"),
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Lists cataloged and Stage-1-diarized videos with turn, overlap, and pedestrian metrics."""
    return list_videos(
        query=q,
        speaker=speaker,
        only_diarized=only_diarized,
        limit=limit,
        offset=offset,
    )


@app.get("/api/videos/{video_id}/scenario")
def api_video_scenario(
    video_id: str,
    include_pedestrians: bool = Query(True, description="Include or filter out pedestrian interruption turns"),
):
    """Returns full multi-speaker Scenario Transcript, speaker roster, and extracted Q&As for a video."""
    payload = get_video_scenario(video_id=video_id, include_pedestrians=include_pedestrians)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found in catalog")
    return payload


@app.get("/api/transcript-search")
def api_transcript_search(
    q: str = Query("", description="Full-text search across diarized scenario turns"),
    role: str = Query("all", description="Optional speaker role filter: MAIN_DAEE, INTERLOCUTOR, PEDESTRIAN"),
    limit: int = Query(40, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Searches directly inside Stage 1 speaker-diarized scenario turns via FTS5."""
    return search_scenario_turns(query=q, role=role, limit=limit, offset=offset)


@app.get("/api/stats")
def api_stats():
    """Returns V2 acoustic diarization & extraction corpus telemetry."""
    return get_stats()


if __name__ == "__main__":
    import uvicorn

    print("[+] Starting Dawah Dialectics V2 Web Portal on http://127.0.0.1:8000")
    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=True)
