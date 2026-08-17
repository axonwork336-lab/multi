"""
The supervisor.

Decides which specialist owns the current turn. It runs ONCE per user
turn, at the top of the graph - never inside the agent<->tools loop - so
a turn that makes six tool calls still routes exactly once.

WHY IT IS DETERMINISTIC BY DEFAULT
----------------------------------
Three reasons, in order of importance:

  1. CONSISTENCY, which is the whole point of this refactor. The same
     sentence must always reach the same specialist and therefore always
     produce the same shape of reply. An LLM classifier re-deciding on
     every turn is exactly the "it answers me one way and answers him
     another way" problem, moved one layer down.

  2. COST AND LATENCY. An LLM router adds a full model call to every
     single turn, before the turn's real work has even started.

  3. TESTABILITY. The existing test suite scripts the LLM's replies one
     per `.invoke()`; a router that quietly consumed one of those calls
     would desynchronise every scripted conversation in the repo.

`ROUTER_MODE=llm` is available for anyone who wants the flexible
version - it only fires on genuinely ambiguous turns, never on clear
ones - but it is off by default.

STICKINESS IS THE OTHER HALF
----------------------------
Most messages inside a flow carry no intent words at all: "نعم", "١",
"123456", "+201001234567", "الخميس". Re-classifying those from scratch
would scatter one conversation across several specialists. So the rule
is: a message with no clear cue KEEPS the current specialist. Only a
clear, deliberate change of subject moves the conversation, and only a
completed flow releases it back to the concierge.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import config
from agents.registry import AGENT_NAMES, CONCIERGE

logger = logging.getLogger(__name__)


# ==========================================================
# Score thresholds
# ==========================================================

# A cue this strong switches the conversation even mid-flow.
_SWITCH_THRESHOLD = 8

# Enough to pick a specialist when nothing is active yet, but not enough
# to interrupt a flow already in progress.
_START_THRESHOLD = 4


# ==========================================================
# Intent cues
#
# Written as (weight, pattern) pairs. Patterns are matched against a
# normalised copy of the message (Arabic diacritics stripped, alef/ya/ta
# marbuta unified, Arabic-Indic digits converted), so "أبغى" and "ابغي"
# and "ابغى" all hit the same rule without listing every spelling.
#
# Weights: 10 = an unambiguous verb+object phrase; 6 = a strong single
# keyword; 3 = a topical hint that only decides an otherwise-empty turn.
# ==========================================================

_DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670\u0640]")
_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ALEF_RE = re.compile(r"[أإآٱ]")
_WHITESPACE_RE = re.compile(r"\s+")


_CUES: Dict[str, List[Tuple[int, str]]] = {

    "cancel": [
        (10, r"(?:الغ|إلغ|ابطال|ابغى\s*الغ|عايز\s*الغ|عاوز\s*الغ|ابي\s*الغ|حاب\s*الغ)\w*\s*"
             r"(?:ال)?(?:حجز|موعد|معاد|ميعاد)"),
        (10, r"\bcancel\b[^.\n]{0,20}\b(?:booking|appointment|reservation|it)\b"),
        (10, r"\b(?:booking|appointment|reservation)\b[^.\n]{0,20}\bcancel"),
        # "ألغيه"/"ألغيها"/"ابطلها" - verb + attached object pronoun, i.e.
        # "cancel it". Unambiguous, so it is strong enough to interrupt
        # another flow ("خلاص ألغيه بقى" said mid-reschedule).
        (10, r"(?:^|\s)(?:الغي|ابطل)\w*(?:ه|ها)(?:\s|$)"),
        (10, r"\bcancel\s+it\b"),
        # `\w*` on purpose: Arabic attaches the object pronoun to the
        # verb, so "ألغيه"/"ألغيها" ("cancel it") are one word. Anchoring
        # with (?:\s|$) missed every one of them.
        (6, r"(?:^|\s)(?:الغاء|الغي|ابطل)\w*(?:\s|$)"),
        (6, r"(?:^|\s)اريد\s*الالغاء"),
        (6, r"\bcancel(?:lation|ling)?\b"),
        (6, r"\bcall\s*off\b"),
        (3, r"(?:مش\s*هقدر\s*(?:اجي|احضر)|ما\s*اقدر\s*اجي|لن\s*احضر)"),
        (3, r"\b(?:can(?:'|’)?t|cannot|won(?:'|’)?t)\s+(?:make|come|attend)\b"),
    ],

    "reschedule": [
        (10, r"(?:تاجيل|تأجيل|اجل|أجل|تغيير|غير|اغير|أغير|تعديل|عدل|اعدل|نقل|انقل|قدم|تقديم)\w*\s*"
             r"(?:ال)?(?:حجز|موعد|معاد|ميعاد|الوقت|التاريخ)"),
        (10, r"(?:ال)?(?:حجز|موعد|معاد|ميعاد)\w*\s*"
             r"(?:الى|إلى|ل)\s*(?:يوم|وقت|تاريخ|ميعاد|معاد)\s*(?:تاني|اخر|آخر|ثاني|جديد)"),
        (10, r"\b(?:reschedul|postpon|posptone)\w*"),
        (10, r"\b(?:change|move|shift|push|switch)\b[^.\n]{0,25}\b(?:booking|appointment|slot|time|date)\b"),
        (10, r"\bearlier\s+(?:slot|time|appointment)\b"),
        # Verb + attached object pronoun: "أأجله", "أجلها", "تأجيله",
        # "أغيره", "ننقله" - all one word, all meaning "move it".
        (6, r"(?:^|\s)\w{0,2}(?:اجل|تاجيل|غير|نقل)\w*(?:ه|ها)(?:\s|$)"),
        (6, r"\bnew\s+(?:time|date|slot)\s+for\s+(?:my|the)\s+(?:booking|appointment)\b"),
    ],

    "booking": [
        (10, r"(?:احجز|أحجز|اححز|حجز|ابغى\s*حجز|عايز\s*احجز|عاوز\s*احجز|ابي\s*احجز|"
             r"حاب\s*احجز|ودي\s*احجز|اريد\s*حجز|نفسي\s*احجز)\w*\s*"
             r"(?:موعد|معاد|ميعاد|كشف|عند|مع|جديد)"),
        (10, r"(?:موعد|معاد|ميعاد|كشف)\s*جديد"),
        (10, r"\bbook\b[^.\n]{0,25}\b(?:appointment|slot|consultation|visit|doctor)\b"),
        (10, r"\bnew\s+(?:appointment|booking)\b"),
        (10, r"\bmake\s+(?:an?\s+)?appointment\b"),
        (6, r"(?:^|\s)(?:احجز|أحجز|احجزلي|احجز\s*لي|اريد\s*الحجز|عايز\s*حجز|عاوز\s*حجز|ابغى\s*احجز)(?:\s|$)"),
        (6, r"\b(?:booking|reserve|schedule)\s+(?:an?\s+)?(?:appointment|visit|consultation)\b"),
        (3, r"(?:متاح|فاضي|مواعيد\s*متاحه|في\s*مواعيد)"),
        (3, r"\bavailable\s+(?:slots?|times?|appointments?)\b"),
    ],

    "medical": [
        (10, r"(?:عندي|بعاني|اعاني|بحس|احس|حاسس|حاسه|بشتكي|فيني|يوجعني|بيوجعني|بيوجعوني)\s*"
             r"\w*\s*(?:الم|وجع|صداع|حراره|سخونه|مغص|دوخه|دوار|كحه|كحة|سعال|ترجيع|قيء|"
             r"غثيان|اسهال|امساك|حكه|طفح|تعب|ارهاق|ضيق|تنميل|حساسيه)"),
        (10, r"\b(?:i\s+have|i(?:'|’)?ve\s+got|i\s+feel|suffering\s+from|experiencing)\b"
             r"[^.\n]{0,30}\b(?:pain|ache|fever|headache|dizziness|nausea|cough|rash|swelling|"
             r"bleeding|burning|numbness|blurred)\b"),
        (10, r"(?:اي|أي|انهي|أنهي|مين)\s*(?:دكتور|طبيب|تخصص|قسم)\s*(?:يناسب|مناسب|اروح|أروح|اشوف|ازور)"),
        (10, r"\bwhich\s+(?:doctor|specialty|speciality|department)\b"),
        (6, r"(?:^|\s)(?:وجع|الم|ألم|صداع|مغص|دوخه|دوخة|سخونيه|سخونية|حراره|حرارة|كحه|كحة|"
             r"غثيان|قيء|ترجيع|اسهال|امساك|طفح|حكه|حكة|تنميل|ضيق\s*تنفس|نزيف)(?:\s|$)"),
        (6, r"\b(?:pain|ache|fever|headache|migraine|nausea|dizzy|dizziness|cough|rash|"
             r"swelling|bleeding|sore\s+throat|shortness\s+of\s+breath)\b"),
        (6, r"(?:تعبان|تعبانه|مريض|مريضه|مش\s*قادر\s*اتنفس|ما\s*اقدر\s*اتنفس)"),
        # BODY PART + a hurting verb, with no "عندي/بحس" lead-in. Arabic
        # attaches the pronoun to both words ("بطني بتوجعني", "راسي
        # بيوجعني", "ضهري وجعان"), which the "عندي + symptom" patterns
        # above cannot see. Confirmed miss in production: "بطني بتوجعني"
        # routed to the concierge instead of medical guidance.
        (10, r"(?:بطني|معدتي|راسي|رأسي|ضهري|ظهري|صدري|رقبتي|عيني|عينيا|سناني|"
             r"ضرسي|حلقي|زوري|قلبي|كتفي|ركبتي|رجلي|ايدي|جسمي|صدرى)\s*"
             r"\w*\s*(?:بتوجع|بيوجع|توجع|يوجع|وجعان|وجعانه|بتألم|بيألم|تعبان|تعبانه|"
             r"مولعه|مولع|بتحرق|بيحرق|منتفخ|منتفخه)"),
        (10, r"\bmy\s+(?:stomach|head|back|chest|neck|throat|tooth|teeth|eye|eyes|"
             r"knee|leg|arm|shoulder|belly|tummy)\s+"
             r"(?:hurts?|aches?|is\s+(?:hurting|aching|sore|swollen|killing))"),
        # Pain severe enough to be described by its effect rather than
        # named as a symptom.
        (6, r"(?:مش\s*قادر|ما\s*اقدر|مش\s*عارف|مو\s*قادر)\s*\w*\s*"
            r"(?:من\s*)?(?:الوجع|الالم|الالام|التعب|الصداع)"),
        (6, r"(?:الوجع|الالم)\s*(?:مش|ما)\s*(?:بيروح|يروح|بينتهي)"),
        (3, r"(?:نصيحه\s*طبيه|استشاره\s*طبيه|توجيه\s*طبي)"),
        (3, r"\bmedical\s+(?:advice|guidance|consultation)\b"),
    ],

    "complaint": [
        (10, r"(?:عندي|اقدم|أقدم|ابغى\s*اقدم|عايز\s*اقدم|حاب\s*اقدم|اريد\s*تقديم|بدي\s*قدم)\s*"
             r"\w*\s*(?:شكوى|شكويه|شكوه|شكاوي|اقتراح|مقترح|ملاحظه|ملاحظة)"),
        (10, r"\b(?:file|submit|make|raise|lodge|register)\s+(?:an?\s+)?"
             r"(?:complaint|grievance|suggestion|feedback)\b"),
        (6, r"(?:^|\s)(?:شكوى|شكويه|شكوه|شكاوي|اشتكي|أشتكي|بشتكي\s*من|اقتراح|مقترح)(?:\s|$)"),
        (6, r"\b(?:complaint|complain|grievance|suggestion)\b"),
        (3, r"(?:خدمه\s*سيئه|معامله\s*سيئه|مستاء|مستاءه|زعلان\s*من|غير\s*راضي)"),
        (3, r"\b(?:poor|bad|terrible|awful)\s+(?:service|treatment|experience)\b"),
        (3, r"\bunhappy\s+with\b"),
    ],

    "faq": [
        (10, r"(?:فين|وين|اين|أين|ايه\s*عنوان|ما\s*هو\s*عنوان)\s*\w*\s*(?:الفرع|المستشفى|العياده|العيادة)"),
        (10, r"\b(?:where\s+is|what(?:'|’)?s\s+the\s+address\s+of)\b[^.\n]{0,25}"
             r"\b(?:branch|hospital|clinic)\b"),
        (10, r"(?:ايه|إيه|ما\s*هي|شنو|وش)\s*(?:هي\s*)?(?:الخدمات|خدماتكم|التخصصات|تخصصاتكم)"),
        (10, r"\bwhat\s+(?:services|specialt(?:y|ies)|departments)\b"),
        # Unambiguous enough to interrupt another flow: nobody asks
        # about opening hours as part of confirming a cancellation.
        (10, r"(?:مواعيد\s*العمل|ساعات\s*العمل|متى\s*تفتحون|امتى\s*بتفتحوا)"),
        (10, r"\b(?:opening|working)\s+hours\b"),
        (6, r"(?:بتفتحوا|بتقفلوا|رقم\s*التواصل|رقم\s*الهاتف\s*للمستشفى|"
             r"الرؤيه|الرساله|القيم|التامين)"),
        (6, r"\bcontact\s+number\b|\bvision\s+and\s+mission\b|"
            r"\binsurance\b|\bpolic(?:y|ies)\b"),
        (3, r"(?:عن\s*المستشفى|معلومات\s*عن|فروعكم|كام\s*فرع)"),
        (3, r"\babout\s+the\s+hospital\b|\byour\s+branches\b"),
    ],

    CONCIERGE: [
        # A human handoff request always comes back to the concierge -
        # it is not a flow, it is an exit.
        (10, r"(?:^|\s)(?:موظف|موظفه|بشري|انسان|إنسان|احد\s*من\s*الموظفين|خدمه\s*العملاء|"
             r"خدمة\s*العملاء|ممثل|حولني|حوليني|كلمني\s*مع\s*حد)(?:\s|$)"),
        (10, r"\b(?:human|agent|representative|customer\s+service|real\s+person|"
             r"speak\s+to\s+(?:someone|staff))\b"),
    ],
}


def _fold_arabic(text: str) -> str:
    """The letter-folding half of `normalize()`.

    Applied to the CUE PATTERNS as well as to messages, so a cue can be
    written the natural way ("شكوى", "أبغى", "دوخة") and still match a
    normalised message where those became "شكوي", "ابغي", "دوخه".
    Getting this wrong is silent - the cue simply never fires - so it is
    done mechanically rather than by remembering to spell every cue in
    its folded form.

    Deliberately does NOT lower-case, collapse whitespace or convert
    digits: those steps are safe on a message but would corrupt regex
    metacharacters in a pattern.
    """

    result = _DIACRITICS_RE.sub("", text)
    result = _ALEF_RE.sub("ا", result)
    return (result.replace("ى", "ي").replace("ة", "ه")
                  .replace("ؤ", "و").replace("ئ", "ي"))


_COMPILED: Dict[str, List[Tuple[int, "re.Pattern"]]] = {
    agent: [(weight, re.compile(_fold_arabic(pattern))) for weight, pattern in cues]
    for agent, cues in _CUES.items()
}


# ==========================================================
# Normalisation
# ==========================================================

def normalize(text: str) -> str:
    """Folds the spelling variations Arabic users actually type, so one
    cue pattern covers "أبغى"/"ابغي"/"ابغى" instead of three."""

    if not text:
        return ""

    result = _fold_arabic(text.translate(_DIGIT_MAP))
    result = _WHITESPACE_RE.sub(" ", result)
    return result.strip().lower()


# ==========================================================
# Scoring
# ==========================================================

def score_message(text: str) -> Dict[str, int]:
    """Returns {agent: score} for one message. Exposed for the tests and
    for anyone tuning the cue lists."""

    normalized = normalize(text)
    if not normalized:
        return {}

    scores: Dict[str, int] = {}

    for agent, cues in _COMPILED.items():
        matched = [weight for weight, pattern in cues if pattern.search(normalized)]
        if matched:
            # The strongest cue decides, with a small bonus for each
            # corroborating one - so "عايز ألغي الحجز" (phrase + keyword)
            # outranks a bare "إلغاء" appearing in passing.
            scores[agent] = max(matched) + (len(matched) - 1)

    return scores


def _best(scores: Dict[str, int]) -> Tuple[Optional[str], int]:
    if not scores:
        return None, 0
    agent = max(scores, key=lambda key: (scores[key], key))
    return agent, scores[agent]


# ==========================================================
# Flow completion (releases stickiness)
# ==========================================================

# When one of these has just succeeded, the flow that owned the
# conversation is over. The next message that carries no cue of its own
# goes back to the concierge instead of being glued to a finished flow.
_TERMINAL_TOOLS = {
    "cancel_appointment": ("success",),
    "reschedule_appointment": ("success",),
    "create_new_booking": ("success",),
    "send_complaint_email": ("sent",),
}


def _flow_just_completed(messages: List) -> bool:
    """Did the PREVIOUS turn finish its flow?

    The message being routed is itself the newest human message, so the
    scan starts just above it and stops at the human turn before that -
    i.e. it looks at exactly one completed turn, never further back. A
    cancellation that succeeded ten turns ago must not keep releasing
    the conversation forever.
    """

    history = list(messages or [])

    last_human = None
    for index in range(len(history) - 1, -1, -1):
        if getattr(history[index], "type", None) == "human":
            last_human = index
            break

    if last_human is None:
        return False

    for message in reversed(history[:last_human]):
        if getattr(message, "type", None) == "human":
            return False

        if getattr(message, "type", None) != "tool":
            continue

        name = getattr(message, "name", "") or ""
        statuses = _TERMINAL_TOOLS.get(name)
        if not statuses:
            continue

        content = str(getattr(message, "content", "") or "")
        if any(f'"status": "{status}"' in content or f"'status': '{status}'" in content
               for status in statuses):
            return True

    return False


def _latest_human_text(messages: List) -> str:
    for message in reversed(messages or []):
        if getattr(message, "type", None) == "human":
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


# ==========================================================
# The routing decision
# ==========================================================

def route_turn(messages: List, active_agent: Optional[str] = None) -> Tuple[str, str]:
    """
    Returns `(agent_name, reason)`. The reason is logged, never shown to
    the patient.

    The rules, in the order they are applied:

      1. No message to read      -> keep the active specialist.
      2. Strong cue for someone
         other than the active
         specialist              -> switch (a deliberate change of
                                    subject, e.g. "خلاص ألغيه بقى" while
                                    booking).
      3. Nothing active yet      -> the best cue above the start
                                    threshold, otherwise the concierge.
      4. The active flow just
         completed AND this
         message has no cue      -> back to the concierge.
      5. Anything else           -> keep the active specialist. This is
                                    the case that covers "نعم", an OTP,
                                    a phone number, a menu number, a
                                    weekday - i.e. most of a real
                                    conversation.
    """

    if active_agent not in AGENT_NAMES:
        active_agent = None

    text = _latest_human_text(messages)

    if not text.strip():
        return (active_agent or CONCIERGE), "no user message - kept current specialist"

    scores = score_message(text)
    candidate, score = _best(scores)

    if config.ROUTER_MODE == "llm" and score < _START_THRESHOLD:
        llm_choice = _classify_with_llm(text, active_agent)
        if llm_choice:
            return llm_choice, "llm router (message had no deterministic cue)"

    if candidate and score >= _SWITCH_THRESHOLD and candidate != active_agent:
        return candidate, f"strong cue for {candidate} (score {score})"

    if active_agent is None:
        if candidate and score >= _START_THRESHOLD:
            return candidate, f"opening cue for {candidate} (score {score})"
        return CONCIERGE, "no clear intent yet - concierge opens the conversation"

    if _flow_just_completed(messages):
        # The previous flow finished, so nothing is being interrupted -
        # a WEAK cue is enough to start the next one. Booking a slot and
        # then saying "طيب ممكن أأجله؟" is the obvious case: mid-flow
        # that hint would rightly be ignored, but right after a
        # completed booking it is plainly a new request.
        if candidate and score >= _START_THRESHOLD:
            return candidate, f"{active_agent} completed - {candidate} takes over (score {score})"
        if not scores:
            return CONCIERGE, f"{active_agent} flow completed - released to concierge"

    if candidate and candidate != active_agent and score >= _START_THRESHOLD:
        # A weak hint mid-flow is usually part of the flow itself
        # ("متاح إمتى؟" while booking is a booking question, not a new
        # intent), so it does NOT move the conversation.
        logger.debug(
            "router: weak cue for %s (score %d) ignored - %s still owns this flow",
            candidate, score, active_agent,
        )

    return active_agent, f"{active_agent} still owns this flow"


# ==========================================================
# Optional LLM routing (ROUTER_MODE=llm)
# ==========================================================

_LLM_ROUTER_PROMPT = """You classify one patient message for a hospital assistant.

Reply with EXACTLY ONE of these words and nothing else:
cancel      - wants to cancel an existing appointment
reschedule  - wants to move an existing appointment to another time
booking     - wants a brand new appointment
medical     - describes a symptom or asks which doctor/specialty they need
faq         - asks about the hospital: services, branches, hours, policies
complaint   - wants to file a complaint or a suggestion
concierge   - anything else, a greeting, or unclear

Currently active flow: {active}
Patient message: {message}"""


def _classify_with_llm(text: str, active_agent: Optional[str]) -> Optional[str]:
    """Only reached when ROUTER_MODE=llm AND the deterministic cues found
    nothing. Any failure returns None and the deterministic path
    continues - routing must never be able to break a conversation."""

    try:
        from langchain_core.messages import HumanMessage
        import graph  # imported lazily: graph imports this package

        llm = getattr(graph, "_llm", None)
        if llm is None:
            return None

        prompt = _LLM_ROUTER_PROMPT.format(
            active=active_agent or "none", message=text[:500],
        )
        answer = llm.invoke([HumanMessage(content=prompt)])
        choice = str(getattr(answer, "content", "")).strip().lower()

        for name in AGENT_NAMES:
            if choice.startswith(name):
                return name

    except Exception:
        logger.warning("router: LLM classification failed - using deterministic result", exc_info=True)

    return None
