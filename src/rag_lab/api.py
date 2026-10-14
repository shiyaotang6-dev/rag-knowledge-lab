from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .core import KnowledgeBase

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.kb = KnowledgeBase.from_env()
    yield


app = FastAPI(title="RAG Knowledge Lab", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=8)
    mode: Literal["offline", "deepseek"] = "offline"


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    kb = app.state.kb
    return {"status": "ok", "documents": len({c.source for c in kb.chunks}), "chunks": len(kb.chunks), "embedding": kb.embedding_name, "reranker": kb.reranker_name, "api_available": bool(os.getenv("RAG_API_KEY") and os.getenv("RAG_MODEL"))}


@app.get("/api/documents")
def documents():
    return [{"id": c.id, "source": c.source, "title": c.title, "text": c.text} for c in app.state.kb.chunks]


@app.post("/api/ask")
def ask(request: AskRequest):
    question = request.question.strip()
    if len(question) < 2:
        raise HTTPException(422, "请输入至少两个非空白字符。")
    if request.mode == "deepseek" and not (os.getenv("RAG_API_KEY") and os.getenv("RAG_MODEL")):
        raise HTTPException(409, "当前服务器未配置模型 API。")
    try:
        return app.state.kb.ask(question, top_k=request.top_k, mode=request.mode)
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
        # Return a fixed message: upstream errors can contain provider details.
        raise HTTPException(502, "模型调用失败或未返回有效证据引用，请重试或使用本地证据模式。") from exc
