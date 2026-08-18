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
    client_id: str = Field(..., min_length=1, description="Which client_config.csv row to use (clinic/tenant) - only used as a fallback when client_config isn't sent")
    message: str = Field(..., min_length=1, description="The user's message text")
    channel_phone: str | None = Field(
        None, description="Optional verified channel identity phone (e.g. WhatsApp sender number)"
    )
    client_config: dict | None = Field(
        None,
        description=(
            "The client's full config row, straight from n8n's own client_config "
            "Data Table (n8n's single source of truth). When given, this is used "
            "INSTEAD of looking client_id up in this project's bundled "
            "client_config.csv - so n8n's Data Table no longer needs a matching "
            "CSV row to work. Omit to keep the old CSV-lookup-by-client_id behavior."
        ),
    )


class ChatResponse(BaseModel):
    reply: str
    # Read by n8n's "IF - Escalate?" node - true the moment
    # tools.request_human_handoff was called this turn (patient asked
    # for staff, or a tool failure is being handed off).
    escalate: bool = False
    # Read by n8n to decide whether to look branch_name up in its own
    # client_config data table (for lat/lng) and send a map pin.
    location: bool = False
    branch_name: str | None = None


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
        result = agent.send_message_with_signals(
            req.client_id, req.session_id, req.message,
            channel_phone=req.channel_phone, client_config=req.client_config,
        )
    except Exception:
        logger.exception(
            "Graph invocation failed for session_id=%s client_id=%s", req.session_id, req.client_id
        )
        raise HTTPException(status_code=500, detail="internal_error: failed to process message")

    logger.info("session_id=%s reply=%r escalate=%s location=%s branch_name=%r",
                req.session_id, result["reply"], result["escalate"], result["location"], result["branch_name"])

    return ChatResponse(
        reply=result["reply"],
        escalate=result["escalate"],
        location=result["location"],
        branch_name=result["branch_name"],
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"status": "error", "detail": "internal_error"})
