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

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from langgraph.errors import GraphRecursionError

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


# Request fields that are the CALL's own parameters, never part of the
# client's config row - used to separate the two when a caller sends the
# config flattened into the top level (see ChatRequest.resolved_client_config).
_CALL_FIELDS = frozenset({"session_id", "client_id", "message", "channel_phone", "client_config"})


class ChatRequest(BaseModel):
    # extra="allow" on purpose: n8n's Data Table row is commonly spread
    # straight into the request body at the TOP LEVEL (clinic_name,
    # msg_unknown_fallback, base_url, ... all as sibling fields of
    # session_id/message), rather than nested under a "client_config"
    # key. Rejecting or silently dropping those extras is exactly what
    # made a fully-populated n8n config look completely empty on this
    # side - confirmed directly from a real request body. Accepting both
    # shapes means neither end has to be reshaped to match the other.
    model_config = ConfigDict(extra="allow")

    session_id: str = Field(..., min_length=1, description="Stable per-conversation id (e.g. Messenger sender id)")
    client_id: str = Field(..., min_length=1, description="Which client to use (clinic/tenant) - only used as a fallback when no config is sent")
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
            "CSV row to work. May ALSO be sent flattened into the top level of "
            "the request body instead of nested here; both are accepted. Omit "
            "entirely to keep the old CSV-lookup-by-client_id behavior."
        ),
    )

    def resolved_client_config(self) -> dict | None:
        """The client's config row for this request, from whichever shape
        the caller used: the nested `client_config` object if present,
        otherwise any extra top-level fields (n8n's Data Table row spread
        flat into the body). Returns None when neither is present, which
        is what makes config.get_messages() fall back to the CSV."""

        if self.client_config:
            return self.client_config

        extras = {k: v for k, v in (self.model_extra or {}).items() if k not in _CALL_FIELDS}
        return extras or None


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

    resolved_config = req.resolved_client_config()

    # Diagnostic ONLY - never logs the actual values (tokens, phone
    # numbers, etc. can live in this dict), just whether a client config
    # arrived this turn (in either shape) and whether the handful of
    # fields most symptoms trace back to are actually present in it. This
    # turns "is the config even arriving?" from an inference from side
    # effects (an empty greeting, a doctor list falling back to
    # not_configured) into a direct one-line answer in the logs.
    if resolved_config:
        logger.info(
            "session_id=%s: client config received (%s), %d key(s): %s | "
            "has msg_unknown_fallback=%s base_url=%s doctors_base_url=%s",
            req.session_id,
            "nested" if req.client_config else "flattened top-level",
            len(resolved_config), sorted(resolved_config.keys()),
            bool(resolved_config.get("msg_unknown_fallback")),
            bool(resolved_config.get("base_url")),
            bool(resolved_config.get("doctors_base_url")),
        )
    else:
        logger.info(
            "session_id=%s: NO client config in this request - falling back to "
            "client_config.csv lookup for client_id=%s",
            req.session_id, req.client_id,
        )

    try:
        result = agent.send_message_with_signals(
            req.client_id, req.session_id, req.message,
            channel_phone=req.channel_phone, client_config=resolved_config,
        )
    except GraphRecursionError:
        # The turn hit the step ceiling - something looped instead of
        # answering. The patient must still get a message: a 500 here
        # means silence on their phone, which is the worst outcome and
        # the one confirmed in production twice.
        logger.exception(
            "Graph hit the recursion limit for session_id=%s client_id=%s - something is "
            "looping. Returning the configured failure message so the patient is not left "
            "with nothing.", req.session_id, req.client_id,
        )
        templates = config.get_messages(req.client_id, client_row_override=resolved_config)
        return ChatResponse(
            reply=(
                templates.get("msg_On_failure")
                or "حدث خطأ تقني 😕. تحب تحاول مرة ثانية؟"
            ),
            escalate=False,
            location=False,
            branch_name=None,
        )
    except Exception:
        # ANY failure still owes the patient a message.
        #
        # A 500 here is silence on their phone: they wrote in, nothing
        # came back, and they have no idea whether to wait or retry.
        # CONFIRMED REAL PRODUCTION FAILURE: a malformed conversation
        # (an assistant tool_call left without its tool response by an
        # earlier aborted turn) returned 500 on EVERY subsequent
        # message, and the session was silently dead from the patient's
        # side. The underlying cause is fixed separately - this makes
        # sure the next one of its kind is visible as a reply rather
        # than as nothing at all.
        logger.exception(
            "Graph invocation failed for session_id=%s client_id=%s", req.session_id, req.client_id
        )
        templates = config.get_messages(req.client_id, client_row_override=resolved_config)
        return ChatResponse(
            reply=(
                templates.get("msg_On_failure")
                or "حدث خطأ تقني 😕. تحب تحاول مرة ثانية؟"
            ),
            escalate=False,
            location=False,
            branch_name=None,
        )

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
