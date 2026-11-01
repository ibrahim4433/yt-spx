#!/usr/bin/env python3
"""
web/app.py
FastAPI web application for the Dawah Dialectics Q&A Knowledge Base & Smart Assistant.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Dict, Any

from db.database import init_db, search_qa, count_qa, get_categories_breakdown, get_interlocutor_breakdown, get_stats, get_db_connection
from core.smart_bot import smart_bot

# Ensure database schema and FTS index are synchronized on startup
init_db()

app = FastAPI(title="Dawah Dialectics Q&A Knowledge Base", version="1.0.0")

# Mount static files
static_dir = BASE_DIR / "web" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
def index_page():
    template_path = BASE_DIR / "web" / "templates" / "index.html"
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Template not found")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/search")
def api_search(
    q: str = Query("", description="Search term in English or Arabic"),
    category: str = Query("all", description="Topic category filter"),
    interlocutor: str = Query("all", description="Interlocutor type filter"),
    sort: str = Query("default", description="Sort order: default, timestamp_asc, timestamp_desc, category"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
):
    """Bilingual hybrid full-text & metadata search endpoint."""
    total = count_qa(query=q, category=category, interlocutor=interlocutor)
    results = search_qa(
        query=q,
        category=category,
        interlocutor=interlocutor,
        limit=limit,
        offset=offset,
        sort=sort
    )
    return {"results": results, "count": len(results), "total": total, "offset": offset, "limit": limit}


@app.get("/api/chat")
def api_chat(q: str = Query(..., description="User query for the smart assistant")):
    """Zero-LLM instant smart assistant endpoint."""
    response = smart_bot.answer_query(q)
    return response


@app.get("/api/categories")
def api_categories():
    """Returns category breakdown with counts."""
    categories = get_categories_breakdown()
    return {"categories": categories}


@app.get("/api/interlocutors")
def api_interlocutors():
    """Returns interlocutor breakdown with counts."""
    interlocutors = get_interlocutor_breakdown()
    return {"interlocutors": interlocutors}


@app.get("/api/stats")
def api_stats():
    """Returns library coverage stats."""
    return get_stats()


if __name__ == "__main__":
    import uvicorn
    print("[+] Starting Dawah Dialectics Web App on http://127.0.0.1:8000")
    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=True)
