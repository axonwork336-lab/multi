"""
Agent entrypoint / orchestration layer.

REWRITTEN (see main.py.pre_rewrite_backup for the old version). The old
functions (start_cancellation_by_reference, start_cancellation_by_phone,
resume_with_value, pending_interrupt, etc.) existed to drive a fixed
sequence of graph interrupts. There is no such fixed sequence anymore -
the LLM decides everything - so this file collapses to the one thing
that's actually still needed: append a HumanMessage to a thread's chat
history, invoke the graph, and return whatever the LLM's final message
of that turn is. That's it. This is intentionally much simpler than
before, which is a direct consequence of moving all conversation logic
into the LLM.

The CLI (`python main.py`) still behaves the same way from the user's
point of view - free-text back and forth - even though its internal
implementation necessarily changed along with the graph shape.
"""

import logging
import threading
import time
from typing import Dict

from langchain_core.messages import HumanMessage

from config import POST_SUCCESS_TIMEOUT_SECONDS, SESSION_TIMEOUT_SECONDS, THREAD_ID_PREFIX, configure_logging
from graph import graph

import progress

configure_logging()
logger = logging.getLogger(__name__)


# ==========================================================
# Per-session concurrency guard
# ==========================================================
#
# Serializes processing for a given session_id, so two near-simultaneous
# calls (e.g. a duplicate/retried webhook delivery from Messenger/
# WhatsApp for the same underlying message) can't both read "no prior
# state yet" before either has finished writing - which previously
# caused BOTH calls to independently treat themselves as turn 1 and each
# send the full deterministic greeting (observed directly: two full
# greeting messages back to back for a single incoming message). This
# does not deduplicate the underlying webhook event itself (that's best
# done in n8n, keyed by the Messenger message's own "mid" field) - it
# only prevents the race condition on OUR side from producing two
# separate "first turn" responses for what should be one.

_session_locks: Dict[str, threading.Lock] = {}
_session_locks_guard = threading.Lock()


def _lock_for(session_id: str) -> threading.Lock:
    with _session_locks_guard:
        if session_id not in _session_locks:
            _session_locks[session_id] = threading.Lock()
        return _session_locks[session_id]


# ==========================================================
# Inactivity-based reset
# ==========================================================
#
# Tracks, per session_id, when it last sent a message and which
# "generation" it's currently on. Two independent triggers bump the
# generation counter - which changes the actual thread_id passed to the
# graph/checkpointer, so MemorySaver treats it as a brand new, empty
# conversation:
#
#   1. General inactivity: more than SESSION_TIMEOUT_SECONDS since the
#      last message of any kind.
#   2. Post-success inactivity: more than POST_SUCCESS_TIMEOUT_SECONDS
#      since a cancellation completed successfully, with no further
#      message in between. A quick follow-up question right after a
#      successful cancellation does NOT reset anything and does NOT
#      trigger the opening greeting again - it's still the same natural
#      conversation. Only silence AFTER the success, past this shorter
#      window, starts a fresh one.
#
# The caller's own session_id never changes; this is entirely internal
# bookkeeping. Consistent with MemorySaver itself, this is in-process
# only (resets on server restart too - see README).

_last_active: Dict[str, float] = {}
_success_at: Dict[str, float] = {}
_generation: Dict[str, int] = {}


_now = time.time  # dedicated reference so tests can patch main._now in
                  # isolation, without affecting time.time() globally
                  # (which LangGraph's own internals also call)


def _config_for(session_id: str) -> dict:
    now = _now()
    last = _last_active.get(session_id)
    success = _success_at.get(session_id)

    reset = False

    if success is not None and (now - success) > POST_SUCCESS_TIMEOUT_SECONDS:
        logger.info(
            "session_id=%s: %.0fs since last successful cancellation (> %ss) with no follow-up - starting a fresh conversation",
            session_id, now - success, POST_SUCCESS_TIMEOUT_SECONDS,
        )
        reset = True
    elif last is not None and (now - last) > SESSION_TIMEOUT_SECONDS:
        logger.info(
            "session_id=%s: %.0fs since last message (> %ss timeout) - starting a fresh conversation",
            session_id, now - last, SESSION_TIMEOUT_SECONDS,
        )
        reset = True

    if reset:
        _generation[session_id] = _generation.get(session_id, 0) + 1
    elif success is not None:
        # A follow-up message arrived within the post-success grace
        # window - this is an ordinary continuation of the same
        # conversation, not a fresh one. Clear the marker; from here on
        # only the general inactivity timeout applies again.
        _success_at.pop(session_id, None)

    _last_active[session_id] = now

    generation = _generation.get(session_id, 0)
    thread_id = f"{THREAD_ID_PREFIX}:{session_id}:{generation}" if generation else f"{THREAD_ID_PREFIX}:{session_id}"

    return {"configurable": {"thread_id": thread_id}}


def _cancellation_just_succeeded(messages: list) -> bool:
    """Detect whether this turn's messages include a successful
    cancel_appointment tool call. Used to reset the session's memory
    right after a cancellation completes - see send_message()."""

    for msg in messages:
        if getattr(msg, "name", None) == "cancel_appointment":
            content = str(getattr(msg, "content", ""))
            if '"status": "success"' in content or "'status': 'success'" in content:
                return True

    return False


def send_message(client_id: str, session_id: str, message: str, channel_phone: str = None) -> str:
    """
    Send one user message for `session_id` and return the agent's reply
    text for this turn.

    This is the ONLY public function this file needs now: whether it's
    the first message of a brand new conversation or a follow-up to an
    earlier question, the call is identical - the checkpointer
    (MemorySaver, preserved) already knows the full prior chat history
    for this thread_id, so client_id/channel_phone only need to be
    supplied again in case they weren't set yet (load_config is a no-op
    once templates are already cached for this thread).

    RESET BEHAVIOR: memory resets automatically in two cases -
      1. After SESSION_TIMEOUT_SECONDS of general inactivity.
      2. After POST_SUCCESS_TIMEOUT_SECONDS of no follow-up message
         following a successful cancellation specifically (below) - NOT
         immediately after success. A quick follow-up question right
         after cancelling stays in the same conversation (no repeated
         greeting); only silence past that shorter window starts fresh.
    See _config_for() for exactly how these two triggers are evaluated.
    """

    logger.info("session_id=%s: sending message", session_id)

    # Bracket the whole turn for progress.py. begin_turn arms nothing by
    # itself - it just marks that a turn is now in flight, so the timer
    # armed later (by graph.py, when a tool call is decided) can be
    # cancelled the moment the turn finishes. The try/finally is what
    # guarantees that cancellation happens even if the turn raises, which
    # is what stops a "please wait" message arriving AFTER the answer -
    # or worse, after an error.
    progress.begin_turn(session_id)

    try:
        with _lock_for(session_id):
            thread_config = _config_for(session_id)

            # Snapshot the message count BEFORE this turn, so we can isolate
            # exactly which messages this turn added afterward - result["messages"]
            # is the FULL accumulated history for the thread (via add_messages),
            # not just this turn's new messages. Without this, a cancellation
            # that succeeded several turns ago would keep being "detected" again
            # on every subsequent turn in the same thread, endlessly re-arming
            # the post-success reset timer.
            existing_snapshot = graph.get_state(thread_config)
            is_new_thread = not existing_snapshot.values
            previous_count = len(existing_snapshot.values.get("messages", [])) if existing_snapshot.values else 0

            state = {
                "client_id": client_id,
                "session_id": session_id,
                "channel_phone": channel_phone,
                "messages": [HumanMessage(content=message)],
            }

            if is_new_thread:
                # Seed "greeted" only on a genuinely brand-new thread. Providing
                # it on EVERY turn would reset it back to False each time (state
                # channels use "last write wins" by default) - undoing the
                # persistence this field exists for. Omitting it on the very
                # first turn, on the other hand, left it completely absent from
                # state until agent() got around to setting it - which broke
                # InjectedState's strict validation for any TOOL CALL made
                # before that point (e.g. a first message that goes straight to
                # check_booking_status/cancel_appointment without a preceding
                # plain-text reply).
                state["greeted"] = False
                # Same reason as above: seed the language field so it exists
                # in state from the very first turn, including for a tool
                # call made before agent() writes the detected value.
                state["target_language"] = None

            result = graph.invoke(state, config=thread_config)

            reply = result["messages"][-1].content
            logger.info("session_id=%s: reply=%r", session_id, reply)

            new_messages_this_turn = result["messages"][previous_count:]

            if _cancellation_just_succeeded(new_messages_this_turn):
                _success_at[session_id] = _now()
                logger.info(
                    "session_id=%s: cancellation succeeded this turn - will reset after %ss of no follow-up",
                    session_id, POST_SUCCESS_TIMEOUT_SECONDS,
                )

    finally:
        # Whatever happened above, the turn is over: cancel any interim
        # "please wait" timer that hasn't fired yet.
        progress.end_turn(session_id)

    return reply


# ==========================================================
# CLI
# ==========================================================

def _run_cli() -> None:
    print("=== Guest Booking Cancellation Agent (CLI) ===")

    client_id = input("Client id [Dar El Oyoun-demo]: ").strip() or "Dar El Oyoun-demo"
    session_id = input("Session id [demo-session]: ").strip() or "demo-session"
    channel_phone = input("Channel/WhatsApp sender number (optional, press Enter to skip): ").strip() or None

    print("\nType your message below (e.g. 'I want to cancel my appointment'). Ctrl+C to quit.\n")

    message = input("You: ").strip()

    while True:
        try:
            reply = send_message(client_id, session_id, message, channel_phone=channel_phone)
        except Exception as exc:  # pragma: no cover - CLI convenience only
            print(f"\n[error] {exc}\n")
            break

        print(f"\nAssistant: {reply}\n")
        message = input("You: ").strip()


if __name__ == "__main__":
    _run_cli()
