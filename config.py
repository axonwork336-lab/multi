"""
Central configuration for the Guest Booking Cancellation Agent.

All environment-dependent values live here so that services/nodes never
hardcode URLs, credentials, tunable settings, or CSV paths directly.

This module is also responsible for loading and merging the two
multi-tenant reference-data files exported alongside the n8n workflow:

  - data/client_config.csv       (per-clinic branding, routing, and a
                                   PARTIAL set of message-template overrides)
  - data/dialect_templates.csv   (per-Arabic-dialect DEFAULT message
                                   templates and conversational behavior
                                   strings)

In the original n8n workflow these were read by two nodes referenced from
the agent's system prompt ("Get Client Config" / "Get Dialect Templates")
that were not included in the exported workflow JSON. We reproduce their
effect here: client_config values win when present, otherwise the
client's dialect row in dialect_templates.csv supplies the default. This
was confirmed by inspecting both files - client_config.csv only defines 8
of the ~27 msg_*/behavior keys; everything else (msg_cancellation_confirmation,
msg_cancel_success, msg_phone_number_ask, msg_booking_refrence, ...) only
exists in dialect_templates.csv.
"""

import csv
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# ==========================================================
# .env loading (Part 4 - OpenAI key / dotenv best practice)
# ==========================================================
#
# Resolved by absolute path (PROJECT_ROOT), NOT by relying on the current
# working directory - this must work identically whether launched via
# `python main.py` from anywhere, `pytest`, `langgraph dev`, or on
# LangGraph Platform (which may invoke the module from a different cwd
# than the repo root).

_PROJECT_ROOT_FOR_ENV = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT_FOR_ENV / ".env")

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT: Path = _PROJECT_ROOT_FOR_ENV

DATA_DIR: Path = Path(os.getenv("AGENT_DATA_DIR", str(PROJECT_ROOT / "data")))

# Candidate directories searched, in order, for each CSV filename below.
# This is the actual fix for the "Config CSV not found" warning: the
# loader no longer assumes the CSVs live in exactly one place. It checks
# AGENT_DATA_DIR/data/ first (the documented, recommended location), then
# falls back to the project root itself (where these two files currently
# are, per the uploaded project). Both are resolved from __file__, never
# from os.getcwd(), so this is unaffected by which directory a process is
# launched from - locally, under `langgraph dev`, or on LangGraph
# Platform.
_CANDIDATE_DATA_DIRS: List[Path] = [DATA_DIR, PROJECT_ROOT]

# ==========================================================
# Booking API
# ==========================================================
#
# Precedence (highest wins), per the Railway/n8n deployment requirements:
#   1. BOOKING_API_BASE_URL environment variable, IF explicitly set
#   2. client_config.csv's "base_url" column for the resolved client_id
#   3. _DEFAULT_BASE_URL below (hardcoded fallback)
#
# _ENV_BASE_URL_OVERRIDE is None when the env var isn't set at all
# (as opposed to BASE_URL below, which always has a value via its
# os.getenv(..., default) - we need to distinguish "not set" from "set
# to the default" to know whether the env var should override the CSV).

_DEFAULT_BASE_URL: str = "https://demo.catalystsystems.io:1102"

_ENV_BASE_URL_OVERRIDE: Optional[str] = os.getenv("BOOKING_API_BASE_URL") or None

BASE_URL: str = _ENV_BASE_URL_OVERRIDE or _DEFAULT_BASE_URL

CLIENT_ID_HEADER: str = "ClientId"

REQUEST_TIMEOUT_SECONDS: float = float(
    os.getenv("BOOKING_API_TIMEOUT_SECONDS", "15")
)

# How many times to retry a Doctors/Specialties API call that failed with
# a 5xx or a timeout, before giving up. Confirmed real production
# behaviour: this endpoint has been measured returning the SAME query in
# 0.5s and then, later, either timing out at 29s or answering with a bare
# 500 and an empty body - a transient blip, not a real outage - yet
# _post_json previously gave up on the FIRST 500/timeout. That one failed
# call then propagated all the way to the model, which (with no real
# branch/doctor data to work from) started inventing branch names, which
# the reply-validation guard correctly caught and replaced with the
# generic "حدث خطأ تقني" fallback - so a single transient 500 upstream
# was surfacing as a hard failure to the patient. Kept LOW and env-
# overridable: this is a shield against a known-flaky endpoint's short
# blips, not a way to paper over a genuinely down service - if it's
# really down, retrying more just delays the (still correct) failure
# response to the patient.
DOCTORS_API_MAX_RETRIES: int = int(
    os.getenv("DOCTORS_API_MAX_RETRIES", "2")
)

# Base delay between retries, in seconds. Doubles each attempt
# (0.5s, 1s, 2s, ...) so a genuinely struggling endpoint isn't hammered
# harder than a transient one.
DOCTORS_API_RETRY_BACKOFF_SECONDS: float = float(
    os.getenv("DOCTORS_API_RETRY_BACKOFF_SECONDS", "0.5")
)


# ==========================================================
# Doctors / Specialties API (Medical Concierge feature)
# ==========================================================
#
# Confirmed by the user directly: this is a SEPARATE service from
# GuestBookings, on port 1102 (GuestBookings is on 1101), AND each
# clinic has its OWN different Doctors/Specialties API - confirmed
# directly (tanasuq-saudi's is not the same as Dar El Oyoun-demo's).
#
# CRITICAL: there is deliberately NO hardcoded fallback URL here anymore.
# An earlier version defaulted to tanasuq's confirmed URL for every
# client that didn't have its own configured - which meant any OTHER
# client (e.g. Dar El Oyoun-demo, before its own URL was known) would
# silently query TANASUQ's API and could have ended up suggesting
# Tanasuq's doctors/specialties to a Dar El Oyoun conversation. That is
# a real cross-tenant data leak risk, not just a cosmetic bug - so the
# resolution order is now:
#   1. DOCTORS_API_BASE_URL env var, IF explicitly set (an intentional,
#      explicit override for ALL clients - e.g. a single-tenant deploy).
#   2. client_config.csv's "doctors_base_url" column for the resolved
#      client_id.
#   3. None - tools.py's list_specialties/find_available_doctors treat
#      this as "not_configured" and say so honestly, rather than ever
#      falling back to some other client's URL.

_ENV_DOCTORS_BASE_URL_OVERRIDE: Optional[str] = os.getenv("DOCTORS_API_BASE_URL") or None

# How many days ahead to search for doctor availability by default, when
# the user doesn't specify a particular day - see
# tools.find_available_doctors().
# How long a fetched doctor list stays reusable. Short on purpose: this
# is a latency shield, not a data store - the roster does change. It
# exists because the tenant's Doctors/GetList endpoint was measured
# returning the SAME query in 0.5s and then, an hour later, timing out at
# 29s; a patient who lists doctors, picks one, then changes their mind
# should not pay that cost three times in one conversation.
DOCTOR_LIST_CACHE_SECONDS: float = float(
    os.getenv("DOCTOR_LIST_CACHE_SECONDS", "90")
)

# How far ahead to look for a bookable slot.
#
# RAISED FROM 14 DAYS. Clinics publish the next season's rota in
# advance, and patients are expected to book into it - confirmed in
# production, a doctor's Monday rota was valid from 01/10 while the
# search window ended two weeks out, so those Mondays could not be found
# no matter what the patient asked. With the schedule lookup no longer
# hiding future rotas (see api.get_doctor_schedule's `include_future`),
# a short window here would just move the same blind spot one step
# later: the schedule would show Monday and the day search would find
# nothing.
#
# Cost is one request either way - the sweep is a single paged call, not
# a call per day - so the change is more rows returned, not more round
# trips. Lower it if a client's booking horizon is genuinely shorter.
DOCTOR_AVAILABILITY_WINDOW_DAYS: int = int(
    os.getenv("DOCTOR_AVAILABILITY_WINDOW_DAYS", "60")
)



# ==========================================================
# Booking Status
# ==========================================================
#
# f_cancel_appointment.json checks statusName == "Cancelled" before
# cancelling (idempotency guard). f_lookup_appointment.json's phone-path
# filter excludes numeric status == 6 (Cancelled) when computing "active"
# bookings. Both are preserved as named constants rather than magic
# literals scattered through node.py.

CANCELLED_STATUS_NAME: str = "Cancelled"

# Official numeric status codes, confirmed directly from the Booking
# API's own documentation - these replace the earlier fragile approach
# of matching statusName strings (which had to handle both English AND
# Arabic spellings depending on the accept-language header, and broke at
# least once in practice). Numeric codes are language-independent.
#
# Only the codes something actually reads are defined. The full
# enumeration (ARRIVED/NO_SHOW/COMPLETED/CANCELLED as separate
# constants, plus CANCELLABLE_STATUSES as a parallel list of NAMES) used
# to be spelled out here and none of it was referenced anywhere - a
# second, string-based cancellability list sitting next to the numeric
# one is exactly how the two drift apart and how the string-matching bug
# above comes back.
STATUS_NEW: int = 1
STATUS_CONFIRMED: int = 2

# Only these two are cancellable - confirmed directly from the
# dashboard's own status dropdown (جديد/تم التأكيد were the only ones NOT
# excluded; وصل/لم يحضر/مكتمل/ملغي were all excluded).
CANCELLABLE_STATUS_CODES = (STATUS_NEW, STATUS_CONFIRMED)


# ==========================================================
# Timezone conversion (per-client, NOT a single hardcoded offset)
# ==========================================================
#
# client_config.csv already has a real "timezone" column per client
# (e.g. "Africa/Cairo" for Dar El Oyoun, "Asia/Riyadh" for tanasuq) -
# this used to be ignored in favor of a single hardcoded "+3 hours"
# applied to every clinic regardless of its actual timezone, which would
# have silently produced wrong times for any clinic outside Saudi
# Arabia. tools.py's to_riyadh() (kept its historical name, but now
# genuinely per-client) reads this value from state["templates"]
# instead. This constant is ONLY the fallback for the rare client row
# missing the column entirely.

DEFAULT_TIMEZONE: str = os.getenv("DEFAULT_TIMEZONE", "Asia/Riyadh")


# ==========================================================
# OTP Settings
# ==========================================================
#
# Two providers are supported, selected by OTP_PROVIDER:
#   "dummy"       -> mirrors OTP_Dummy_send.json / OTP_Dummy_verify.json
#                    (always succeeds; TEST_OTP is accepted as correct)
#   "authentica"  -> mirrors send_otp5 / verify_otp5 in
#                    langchain_cancellation.json (api.authentica.sa)
# Defaults to "dummy" so the project runs end-to-end with no external
# OTP credentials, exactly like the n8n dev setup that ships both a real
# and a dummy OTP sub-workflow side by side.

OTP_PROVIDER: str = os.getenv("OTP_PROVIDER", "dummy")

TEST_OTP: str = os.getenv("TEST_OTP", "123456")
OTP_TTL_SECONDS: int = 5 * 60  # 5 minutes

AUTHENTICA_BASE_URL: str = os.getenv(
    "AUTHENTICA_BASE_URL", "http://api.authentica.sa/api/v2"
)
AUTHENTICA_API_KEY: str = os.getenv("AUTHENTICA_API_KEY", "")
AUTHENTICA_TEMPLATE_ID: str = os.getenv("AUTHENTICA_TEMPLATE_ID", "31")
AUTHENTICA_FALLBACK_EMAIL: str = os.getenv("AUTHENTICA_FALLBACK_EMAIL", "")

# SMTP config for the Complaint Agent's send_complaint_email tool.
# Per-clinic recipient list comes from client_config.csv's own
# "complaint_email_to" column (see get_messages) - the SMTP server
# credentials themselves are shared infra, not per-client.
#
# COMPLAINT_WEBHOOK_URL, if set, takes priority over direct SMTP: the
# complaint's subject/body/recipients are POSTed to this n8n webhook
# instead, and n8n's own email-send node does the actual delivery -
# confirmed necessary in production after direct SMTP from Railway hit
# an unresolvable connection timeout to the clinic's mail server (very
# likely an outbound port/firewall restriction on Railway's side, or an
# IP-allowlist on the mail server's side - n8n's own network path is
# already confirmed working for this same system's WhatsApp messaging).
COMPLAINT_WEBHOOK_URL: str = os.getenv("COMPLAINT_WEBHOOK_URL", "")

SMTP_HOST: str = os.getenv("SMTP_HOST", "")
SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM_EMAIL: str = os.getenv("SMTP_FROM_EMAIL", "") or SMTP_USERNAME

# Port 465 is implicit SSL (smtplib.SMTP_SSL from the first connection) -
# fundamentally different from port 587's STARTTLS (a plain connection
# upgraded to TLS after connecting). Auto-detect from the port unless
# explicitly overridden via SMTP_USE_SSL, since getting this wrong fails
# the connection entirely rather than just being insecure.
_smtp_use_ssl_override = os.getenv("SMTP_USE_SSL")
if _smtp_use_ssl_override is not None:
    SMTP_USE_SSL: bool = _smtp_use_ssl_override.strip().lower() not in ("false", "0", "")
else:
    SMTP_USE_SSL = SMTP_PORT == 465

SMTP_USE_TLS: bool = os.getenv("SMTP_USE_TLS", "true").strip().lower() not in ("false", "0", "") and not SMTP_USE_SSL


# ==========================================================
# Default Country / Phone Normalization
# ==========================================================

DEFAULT_COUNTRY_CODE: str = os.getenv("DEFAULT_COUNTRY_CODE", "20")  # Egypt


# ==========================================================
# LangGraph / Thread Settings
# ==========================================================

THREAD_ID_PREFIX: str = "guest-cancel"

# Maximum LangGraph steps in a single turn. Guards the agent->tools->
# agent cycle, which is otherwise bounded only by the model deciding to
# stop - and when it doesn't, the turn never returns and the patient
# receives nothing at all (confirmed twice in production). A normal turn
# uses a handful of steps; even the deepest real flow stays far inside
# this, so hitting it means something is genuinely looping.
GRAPH_RECURSION_LIMIT: int = int(os.getenv("GRAPH_RECURSION_LIMIT", "30"))

# After this many seconds of no message on a given session_id, the next
# message starts a completely fresh conversation (new thread_id) instead
# of resuming the old one - see main.py's send_message().
SESSION_TIMEOUT_SECONDS: int = int(os.getenv("SESSION_TIMEOUT_SECONDS", "3600"))  # 1 hour

# Shorter grace period specifically after a cancellation completes
# successfully: if no follow-up message arrives within this window, the
# NEXT message starts fresh. A follow-up within this window continues
# the same conversation as normal (no repeated greeting). See main.py's
# _config_for()/_cancellation_just_succeeded().
POST_SUCCESS_TIMEOUT_SECONDS: int = int(os.getenv("POST_SUCCESS_TIMEOUT_SECONDS", "600"))  # 10 minutes

# Caps how many of the most recent chat messages (human/AI/tool) are
# actually sent to the LLM each turn - the checkpointer still keeps the
# FULL history for the thread (nothing is deleted), this only trims what
# graph.py's agent() sends in the prompt. Without this, a long-running
# conversation resends its entire history (plus the ~1000+ token system
# prompt) on every single turn, growing without bound and driving up
# both cost and latency turn over turn. 40 messages is roughly the last
# 15-20 back-and-forth exchanges, comfortably more than the flows in
# prompts.py ever need to look back on. See graph.py's agent().
MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "40"))


# ==========================================================
# OpenAI (language/dialect detection, ref/phone extraction, selection
# matching - the only three LLM touchpoints in the hybrid design)
# ==========================================================

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4.1")  # upgraded from gpt-4.1-mini for better dialect/persona instruction-following
OPENAI_TIMEOUT_SECONDS: float = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "10"))

# NOTE: there is deliberately no "run without an LLM" flag any more.
# The old hybrid design could fall back to deterministic heuristics when
# no API key was present; this architecture cannot - the LLM decides
# every conversational step - so a flag implying otherwise was a
# misleading leftover. OPENAI_API_KEY is required, as stated in the
# README and requirements.txt.


# ==========================================================
# MULTI-AGENT (supervisor + specialist agents)
# ==========================================================
# The graph is now a supervisor pattern: a deterministic `router` node
# picks which specialist agent owns the current turn, and that agent -
# and only that agent - talks to the LLM for that turn. Each specialist
# gets a SCOPED system prompt (shared core + its own flow only) and a
# SCOPED tool subset, instead of one 90 KB prompt + 28 tools for every
# single message.
#
# Every flag below exists so this can be dialled back to the exact
# previous behaviour without editing code, if anything ever misbehaves
# in production.


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# MASTER KILL SWITCH. False -> the graph compiles as the old single
# `agent` node with the old full prompt and all 28 tools, i.e. byte-for
# -byte the previous architecture. Nothing else in this file matters.
MULTI_AGENT_ENABLED: bool = _flag("MULTI_AGENT_ENABLED", True)

# False -> every specialist still gets its own scoped PROMPT, but is
# bound to ALL tools (prompt-level multi-agent only, zero risk of a
# needed tool being unavailable). Use this if a flow ever stalls because
# a specialist wanted a tool outside its subset.
AGENT_TOOL_SCOPING: bool = _flag("AGENT_TOOL_SCOPING", True)

# "deterministic" (default) -> routing is pure code: weighted intent
#     cues + stickiness. Costs zero extra LLM calls and zero extra
#     latency, and is 100% reproducible - the same message always routes
#     to the same agent.
# "llm" -> ambiguous messages (and only those) additionally get a small
#     classification call. More flexible, but adds a call per ambiguous
#     turn and makes routing non-deterministic.
ROUTER_MODE: str = os.getenv("ROUTER_MODE", "deterministic").strip().lower()

# The reply normalizer (agents/response_contract.py) that guarantees
# every agent's output has identical shape. False -> only the two
# original normalizations (extra-question trimming, emoji list numbers)
# run, as before.
REPLY_NORMALIZATION_ENABLED: bool = _flag("REPLY_NORMALIZATION_ENABLED", True)


# ==========================================================
# INTERIM "PLEASE WAIT" MESSAGES (progress.py)
# ==========================================================
# `/chat` is request/response, so an interim line returned inside that
# response would arrive in the same instant as the answer. To actually
# reach the patient WHILE a tool is running, the agent pushes it to a
# second webhook (normally an n8n branch that forwards to Messenger).
#
# OFF BY DEFAULT: it needs that webhook and the n8n branch to exist
# first. Until it is switched on, nothing about the agent changes.
PROGRESS_ENABLED: bool = _flag("PROGRESS_ENABLED", False)

# "webhook" -> POST to PROGRESS_WEBHOOK_URL.
# "log"     -> log the line instead of sending it. Use this to see and
#              tune the messages locally before wiring up n8n.
PROGRESS_MODE: str = os.getenv("PROGRESS_MODE", "webhook").strip().lower()

PROGRESS_WEBHOOK_URL: str = os.getenv("PROGRESS_WEBHOOK_URL", "").strip()

# How long a turn must already have been running before the patient is
# told to wait. Most tool calls finish well inside this, so most turns
# still produce exactly one message. Raise it if the interim line feels
# too eager, lower it if patients are waiting in silence.
# How long a tool phase must ALREADY have been running before the
# patient is told to wait.
#
# RAISED FROM 1.5s. At 1.5s the timer fired on turns that were about to
# finish anyway: the interim line and the real answer left the app
# within a few tens of milliseconds of each other, and since they are
# two separate deliveries through n8n, their ARRIVAL order is not
# guaranteed. Confirmed in production - the "please wait" line showing
# up underneath the answer it was supposed to precede, which reads as
# though the assistant lost track of the conversation.
#
# Measured on the live medtown deployment, a turn that needs an interim
# message at all runs 5-12s; the turns that were producing out-of-order
# messages were finishing in 2-4s. 3.0s sits between the two, so slow
# turns still get their line and quick ones stay silent.
#
# This is a latency-vs-noise trade-off, not a correctness fix: the
# ordering guard in progress.py is what makes it safe. Raise it further
# if the warning it logs still appears.
PROGRESS_DELAY_SECONDS: float = float(os.getenv("PROGRESS_DELAY_SECONDS", "3.0"))

# Kept short: this is a courtesy message, and its delivery must never
# hold anything up.
PROGRESS_TIMEOUT_SECONDS: float = float(os.getenv("PROGRESS_TIMEOUT_SECONDS", "5"))


# ==========================================================
# Logging
# ==========================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def configure_logging() -> None:
    """Configure root logging once for the whole application."""

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


logger = logging.getLogger(__name__)


# ==========================================================
# client_config.csv / dialect_templates.csv loading
# ==========================================================

# Keys present in BOTH files. For these, client_config.csv (when the
# client has a non-empty value) takes precedence over the dialect
# default. Verified against both CSV headers.
_CLIENT_OVERRIDE_KEYS = (
    "msg_unknown_fallback",
    "msg_media_canned",
    "msg_handoff_confirmation",
    "msg_back_to_ai",
    "msg_patient_booking_number",
    "msg_booking_confirmation",
    "msg_booking_success",
    "msg_On_failure",
)


def _resolve_data_file(filename: str) -> Optional[Path]:
    """Search _CANDIDATE_DATA_DIRS in order for `filename`. Returns the
    first match, or None if it isn't found in any candidate location."""

    for directory in _CANDIDATE_DATA_DIRS:
        candidate = directory / filename
        if candidate.exists():
            return candidate

    return None


def _read_csv_rows(filename: str) -> list:
    """Read a CSV (UTF-8 with BOM) into a list of dict rows, searching
    _CANDIDATE_DATA_DIRS for `filename`. Returns an empty list (with a
    warning naming every path that was tried) if it isn't found anywhere,
    so callers can fall back to built-in defaults rather than crashing."""

    resolved = _resolve_data_file(filename)

    if resolved is None:
        tried = ", ".join(str(d / filename) for d in _CANDIDATE_DATA_DIRS)
        logger.warning("Config CSV '%s' not found. Tried: %s", filename, tried)
        return []

    with open(resolved, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=1)
def _all_client_configs() -> Dict[str, dict]:
    """client_id -> row dict, loaded once and cached."""

    rows = _read_csv_rows("client_config.csv")
    return {row["client_id"]: row for row in rows if row.get("client_id")}


@lru_cache(maxsize=1)
def _all_dialect_templates() -> Dict[str, dict]:
    """Dialect name (as written in the CSV, e.g. "Egyptian") -> row dict,
    loaded once and cached."""

    rows = _read_csv_rows("dialect_templates.csv")
    return {row["Dialect"].strip(): row for row in rows if row.get("Dialect")}


def get_client_config(client_id: str) -> Optional[dict]:
    """Return the raw client_config.csv row for `client_id`, or None if
    that client isn't configured."""

    return _all_client_configs().get(client_id)


def get_dialect_template(dialect: str) -> Optional[dict]:
    """Return the raw dialect_templates.csv row for `dialect` (e.g.
    "Egyptian", "Saudi"), or None if that dialect isn't configured."""

    if not dialect:
        return None

    # dialect_templates.csv has at least one row with trailing whitespace
    # in its Dialect column ("Saudi ") - normalize both sides.
    target = dialect.strip().lower()
    for name, row in _all_dialect_templates().items():
        if name.strip().lower() == target:
            return row

    return None


def _unwrap_quotes(value: str) -> str:
    """Remove a pair of quote characters wrapping an entire value.

    Message templates are authored in a spreadsheet / Data Table, where
    it is natural to type quotes around a sentence to show where it
    begins and ends. Those quotes are part of the VALUE, not of the CSV
    encoding, so they survive parsing and get sent to the patient.

    Confirmed in production data: `msg_On_failure` reads
    '"حدث خطأ تقني 😕. تحب تحاول مرة ثانية؟"' for BOTH clients -
    quote marks included. It is a message the patient only ever sees
    when something has already gone wrong, so it is the least likely to
    be noticed in testing and the worst moment to look sloppy.

    Only stripped when a matching pair wraps the WHOLE value, so a
    quotation used INSIDE a message is untouched.
    """

    if not isinstance(value, str):
        return value

    text = value.strip()
    if len(text) < 2:
        return value

    for opening, closing in (('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u00ab", "\u00bb")):
        if text[0] == opening and text[-1] == closing:
            inner = text[1:-1].strip()
            # Don't strip when the value is several quoted fragments
            # ("a" and "b") - only when it is one wrapped sentence.
            if opening not in inner and closing not in inner:
                return inner

    return value


def get_messages(client_id: str, dialect: Optional[str] = None, client_row_override: Optional[dict] = None) -> dict:
    """
    Build the merged message-template dict used by build_response and the
    system prompt equivalent throughout the graph.

    Resolution order:
      1. Start with the dialect row for `dialect`, or the client's own
         `Dialect` column (from `client_row_override` if given, else
         client_config.csv) if `dialect` isn't given.
      2. Overlay any non-empty client-row values for the keys both
         sources define (_CLIENT_OVERRIDE_KEYS).

    `client_row_override`: when given (not None), this dict is used
    AS the client's config row INSTEAD of looking client_id up in
    client_config.csv - this is how a client's full config, sent fresh
    by n8n on every /chat request (its own Data Table row, which is n8n's
    single source of truth for client config), takes over from this
    project's bundled CSV without needing a matching row there too. Pass
    None (the default) to keep the original CSV-lookup behavior - used by
    the CLI and anything invoking the graph directly without going
    through n8n. An override that resolves to falsy (empty dict/None) is
    treated as "use the CSV" too, since an override that provides
    literally nothing is indistinguishable from not overriding at all.

    Returns a plain dict (never raises); missing config degrades to an
    empty dict, and format_message()'s own built-in English/Arabic
    fallback strings (utils/formatter equivalent in prompts.py) take over
    from there.
    """

    client_row = client_row_override or get_client_config(client_id) or {}

    effective_dialect = dialect or client_row.get("Dialect")

    dialect_row = get_dialect_template(effective_dialect) or {}

    merged = dict(dialect_row)

    for key in _CLIENT_OVERRIDE_KEYS:
        value = client_row.get(key)
        if value:
            merged[key] = _unwrap_quotes(value)

    # A few non-message fields nodes/prompts need directly, not just for
    # templating - kept alongside the message dict so callers only need
    # one merged object per (client_id, dialect) pair.
    merged["_clinic_name"] = client_row.get("clinic_name")
    merged["_clinic_name_ar"] = client_row.get("clinic_name_ar")
    merged["_agent_name"] = client_row.get("agent_name")
    merged["_agent_name_ar"] = client_row.get("agent_name_ar")
    merged["_base_url"] = (
        _ENV_BASE_URL_OVERRIDE
        or client_row.get("base_url")
        or (BASE_URL if client_row else None)
    )
    # doctors_base_url falls back to base_url when the client's own
    # config row doesn't set one explicitly - most clients run doctors/
    # specialties off the SAME server as bookings, and requiring a
    # second, separately-configured column for that (when it's usually
    # identical) is one more place for the two to silently drift apart.
    # A client that genuinely has no doctors/specialties feature at all
    # (a real, deliberate distinction - not just an unset field) should
    # set its OWN column to something falsy/blank explicitly if their
    # config source supports that, or the environment override below can
    # still force it off/on globally.
    merged["_doctors_base_url"] = (
        _ENV_DOCTORS_BASE_URL_OVERRIDE
        or client_row.get("doctors_base_url")
        or merged["_base_url"]
    )
    merged["_phone_example"] = client_row.get("phone_example")
    # Bilingual branch name pairs from the client's own config row, as
    # [{"name": <as the booking API knows it>, "aliases": [...]}, ...].
    #
    # The booking API returns each branch under ONE name only (typically
    # the English one, e.g. "Al Nozha"), but patients naturally type the
    # Arabic one ("النزهة") - and fuzzy matching cannot bridge those two,
    # since they share no letters at all. Confirmed real production
    # failure: a patient was told their branch didn't exist after typing
    # its Arabic name, then the SAME branch matched instantly when they
    # retyped it as "nozha". The config row is the only place both names
    # for the same branch appear together, so it's the only thing that
    # can link them - see tools._branch_alias_map().
    #
    # Kept generic (a list, not branch1_/branch2_ specific) so a client
    # with more than two branches only needs more columns, not a code
    # change here.
    branch_aliases = []
    index = 1
    while True:
        name = client_row.get(f"branch{index}_name")
        name_ar = client_row.get(f"branch{index}_name_ar")
        if not name and not name_ar:
            # Allow one gap (e.g. branch2 removed but branch3 still set)
            # before giving up, but don't scan forever.
            if index > 10:
                break
            index += 1
            if index > 10:
                break
            continue
        aliases = [str(v).strip() for v in (name, name_ar) if v and str(v).strip()]
        if len(aliases) > 1:
            branch_aliases.append({"name": aliases[0], "aliases": aliases})
        index += 1
        if index > 10:
            break
    merged["_branch_aliases"] = branch_aliases
    merged["_country_codes_hint"] = client_row.get("country_codes_hint")
    merged["_timezone"] = client_row.get("timezone") or DEFAULT_TIMEZONE
    merged["_knowledge_base_file"] = client_row.get("knowledge_base_file") or ""
    merged["_complaint_email_to"] = client_row.get("complaint_email_to") or ""
    merged["_dialect_name"] = effective_dialect
    merged["_dialect_instruction"] = dialect_row.get("dialect_instruction") or client_row.get(
        "dialect_instruction"
    )

    return merged


# ==========================================================
# Startup self-check for complaint delivery
# ==========================================================
#
# A clinic can have complaint_email_to filled in perfectly in
# client_config.csv and STILL never receive a single complaint, because
# the transport that actually delivers it (n8n webhook or SMTP) is
# configured entirely separately, via env vars. Nothing used to surface
# that mismatch - the agent would collect a full complaint and only then
# fail, with the reason buried in one log line. Log it once at startup
# instead, naming exactly what's missing.


def complaint_transport_status() -> dict:
    """Describe how complaint emails would be delivered right now.

    Returns {"ready": bool, "transport": "webhook"/"smtp"/None,
             "missing": [names of unset env vars]}.
    """

    if COMPLAINT_WEBHOOK_URL:
        return {"ready": True, "transport": "webhook", "missing": []}

    missing = [
        name for name, value in (
            ("SMTP_HOST", SMTP_HOST),
            ("SMTP_USERNAME", SMTP_USERNAME),
            ("SMTP_PASSWORD", SMTP_PASSWORD),
        ) if not value
    ]

    if missing:
        return {"ready": False, "transport": None, "missing": ["COMPLAINT_WEBHOOK_URL"] + missing}

    return {"ready": True, "transport": "smtp", "missing": []}


def check_complaint_delivery_config() -> None:
    """Log a warning for every client that has complaint recipients
    configured while no delivery transport exists. Called from
    configure_logging()'s callers at startup."""

    clients_expecting_complaints = [
        client_id for client_id, row in _all_client_configs().items()
        if (row.get("complaint_email_to") or "").strip()
    ]

    if not clients_expecting_complaints:
        return

    status = complaint_transport_status()

    if status["ready"]:
        logger.info(
            "Complaint delivery ready via %s for client(s): %s",
            status["transport"], ", ".join(clients_expecting_complaints),
        )
        return

    logger.warning(
        "COMPLAINT EMAIL WILL FAIL: client(s) %s have complaint_email_to configured, but no delivery "
        "transport is set up. Set COMPLAINT_WEBHOOK_URL (recommended - routes through n8n), or all of "
        "SMTP_HOST / SMTP_USERNAME / SMTP_PASSWORD. Currently unset: %s",
        ", ".join(clients_expecting_complaints), ", ".join(status["missing"]),
    )
