"""
Shared LangGraph state for the Guest Booking Cancellation Agent.

REWRITTEN for the LLM-tool-calling architecture (previously a 38-field
hand-written state machine for a deterministic router - see
state.py.pre_rewrite_backup for the old version). The LLM now owns all
conversation logic, so state only needs to carry: tenant identity, the
cached per-tenant config/system-prompt (computed once per thread, not
re-derived every turn), and the chat history itself.

IMPORTANT: `messages` uses LangGraph's `add_messages` reducer. Each graph
invocation only needs to supply the NEW message(s) (e.g. one HumanMessage
per turn) - the checkpointer (MemorySaver, unchanged) automatically
appends to and persists the full history per thread_id. This replaces
ALL of the old manual retry-counter/interrupt-payload bookkeeping: the
LLM's own latest message simply either contains tool_calls (loop
continues) or doesn't (turn ends, that message IS the reply to the user).
"""

from typing import Annotated, NotRequired, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):

    # ==========================================================
    # Identity / tenancy
    # ==========================================================

    client_id: str
    session_id: str
    channel_phone: Optional[str]  # verified channel identity (e.g. WhatsApp sender), used by compare_phone

    # The client's full config row, straight from n8n's own client_config
    # Data Table (sent fresh in every /chat request body, since n8n
    # already fetches it before calling us) - this is n8n's single
    # source of truth for client config, taking over from this project's
    # bundled client_config.csv when present. NotRequired/optional: the
    # CLI and anything invoking the graph directly (no n8n in front)
    # legitimately never sends this, and load_config falls back to the
    # CSV lookup by client_id in that case - see config.get_messages()'s
    # client_row_override parameter.
    raw_client_config: NotRequired[Optional[dict]]

    # ==========================================================
    # Cached per-tenant config (loaded once per thread by graph.py's
    # load_config node, not re-derived every turn)
    # ==========================================================

    # Merged client_config.csv + dialect_templates.csv row (renamed from
    # the old project's "messages" field, which clashed with the
    # conventional chat-history key name this architecture needs).
    templates: dict

    # The system prompt built from `templates` (prompts.build_system_prompt) -
    # cached so config.get_messages()/CSV lookups only happen once per
    # thread rather than on every single turn.
    system_prompt: Optional[str]

    # ==========================================================
    # Chat history (the entire conversation - replaces every one of the
    # old state machine's step-specific fields)
    # ==========================================================

    messages: Annotated[list, add_messages]

    # Which language THIS conversation is being conducted in ("ar"/"en"),
    # computed deterministically each turn by graph.py's
    # _detect_target_language and written here BEFORE any tool runs.
    # Tools read it (via tools.conversation_language) so that every
    # human-readable field they return - times, weekday names, doctor/
    # branch/specialty names - is already in the user's own language,
    # rather than being emitted in English and copied verbatim into an
    # otherwise-Arabic reply.
    # NotRequired on purpose: InjectedState validates the state dict
    # strictly for every tool call, so a field that isn't present yet
    # would hard-fail any tool invoked before something wrote it. It is
    # written on every agent() turn anyway - this just makes its absence
    # harmless rather than fatal.
    target_language: NotRequired[Optional[str]]

    # True once the exact opening greeting has been deterministically
    # prepended for this thread (see graph.py's agent() node). This
    # exists because relying on the LLM to reproduce the clinic's
    # greeting text verbatim, every single time, turned out to be
    # unreliable in practice (observed directly: the same clinic's
    # greeting came out differently worded/structured across separate
    # conversations despite explicit prompt instructions to reuse it
    # exactly). Guaranteeing it in code removes that source of
    # inconsistency entirely, without touching how the LLM handles
    # anything else in the conversation.
    greeted: bool

    # ==========================================================
    # Multi-agent routing
    # ==========================================================

    # Which specialist owns this conversation right now - one of
    # agents.registry.AGENT_NAMES ("cancel", "booking", "medical", ...).
    # Written by graph.py's router node at the start of every turn and
    # persisted by the checkpointer, which is what makes routing STICKY:
    # a mid-flow message carrying no intent words of its own ("نعم",
    # "١", an OTP, a phone number, a weekday) stays with whoever was
    # already handling the flow instead of being re-classified from
    # scratch and scattering one conversation across several
    # specialists.
    #
    # NotRequired for the same reason as target_language above: tools
    # validate the state dict strictly on every call, and threads
    # checkpointed before this field existed must keep resuming
    # cleanly rather than hard-failing.
    active_agent: NotRequired[Optional[str]]

    # Why the router made that choice, for the logs only - never shown
    # to the patient, who must never learn there is more than one agent.
    routing_reason: NotRequired[Optional[str]]

    # Which specialist owned the PREVIOUS turn. Lets a specialist tell
    # "I have been running this flow for several turns" apart from "I
    # have just taken this conversation over", which is the difference
    # between a patient detouring mid-booking and a patient abandoning
    # one - see graph.py's _build_abandoned_booking_directive.
    #
    # NotRequired for the same reason as the fields above: threads
    # checkpointed before it existed must keep resuming cleanly.
    previous_agent: NotRequired[Optional[str]]
