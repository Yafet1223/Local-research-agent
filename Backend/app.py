"""
Sage — FastAPI backend.

This file handles HTTP: routing, request validation, and calling into the
agent. All agent logic lives in agent/yafet.py.

Requires:
    pip install fastapi uvicorn langgraph langchain-google-genai python-dotenv

Run:
    uvicorn app:app --reload --port 5000
Then open:
    http://localhost:5000        (the chat UI)
    http://localhost:5000/docs   (auto-generated API docs)
"""

import sys
import os
import uuid
import logging
from pathlib import Path
from typing import Optional

# Add the parent directory to sys.path so we can import the agent package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import AIMessage, ToolMessage

# Import the compiled graph (named 'app' in agent/yafet.py) as 'app_graph'
from agent.yafet import app as app_graph

# ---------------------------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    user_id: str = "web-user"

class ToolTraceItem(BaseModel):
    tool: str
    args: dict
    result: str

class ChatResponse(BaseModel):
    response: str
    trace: list[ToolTraceItem]
    thread_id: str


app = FastAPI(title="Sage")
logger = logging.getLogger("sage")


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id, "user_id": req.user_id}}

    before_count = 0
    try:
        existing = app_graph.get_state(config)
        if existing and existing.values.get("messages"):
            before_count = len(existing.values["messages"])
    except Exception:
        pass

    try:
        result = app_graph.invoke({"messages": [("user", message)]}, config=config)
    except Exception as e:
        logger.exception("Agent invocation failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    new_messages = result["messages"][before_count:]

    trace = []
    pending_calls = {}
    for m in new_messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                pending_calls[tc["id"]] = {"name": tc["name"], "args": tc["args"]}
        if isinstance(m, ToolMessage):
            call = pending_calls.get(m.tool_call_id, {"name": "unknown", "args": {}})
            trace.append(ToolTraceItem(tool=call["name"], args=call["args"], result=m.content))

    return ChatResponse(response=result["messages"][-1].content, trace=trace, thread_id=thread_id)


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve yaf.html at the root route "/"
@app.get("/")
def read_root():
    frontend_path = Path(__file__).parent.parent / "Frontend" / "yaf.html"
    if not frontend_path.exists():
        raise HTTPException(status_code=404, detail="Frontend file yaf.html not found")
    return FileResponse(frontend_path)


# Mount the Frontend folder last for static assets
frontend_dir = Path(__file__).parent.parent / "Frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
