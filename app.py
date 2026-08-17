"""
FastAPI wrapper around the LLM-tool-calling LangGraph cancellation agent.

REWRITTEN (see app.py.pre_rewrite_backup for the old version). The old
file had to reconstruct main.py's two pre-graph CLI questions and
distinguish "graph paused on wait_for_otp/selection/confirmation" from
"graph not started yet" via graph.get_state(...).next, because the old
graph had a fixed sequence of named interrupt points. None of that
exists anymore: every turn is now identical - append the user's message,
invoke the graph, return the LLM's reply - handled by main.py's
send_message(), completely unmodified from what main.py's own CLI calls.
This file is now a thin, literal HTTP wrapper with no logic of its own
beyond request/response shaping and error handling.
"""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import config
import main as agent  # unmodified from this point of view: send_message()

config.configure_logging()
logger = logging.getLogger("app")

# Surface a misconfigured complaint transport at startup rather than
# only when a patient has already typed out their whole complaint.
config.check_complaint_delivery_config()

app = FastAPI(
    title="Guest Booking Cancellation Agent API",
    description="HTTP wrapper around the LLM-tool-calling LangGraph cancellation agent, for n8n/Messenger integration.",
    version="2.0.0",
)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Stable per-conversation id (e.g. Messenger sender id)")
    client_id: str = Field(..., min_length=1, description="Which client_config.csv row to use (clinic/tenant)")
    message: str = Field(..., min_length=1, description="The user's message text")
    channel_phone: str | None = Field(
        None, description="Optional verified channel identity phone (e.g. WhatsApp sender number)"
    )


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    logger.info(
        "Incoming /chat session_id=%s client_id=%s message=%r channel_phone=%r",
        req.session_id, req.client_id, req.message, req.channel_phone,
    )

    try:
        reply = agent.send_message(req.client_id, req.session_id, req.message, channel_phone=req.channel_phone)
    except Exception:
        logger.exception(
            "Graph invocation failed for session_id=%s client_id=%s", req.session_id, req.client_id
        )
        raise HTTPException(status_code=500, detail="internal_error: failed to process message")

    logger.info("session_id=%s reply=%r", req.session_id, reply)

    return ChatResponse(reply=reply)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"status": "error", "detail": "internal_error"})
