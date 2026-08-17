"""
THE SINGLE OUTPUT CONTRACT.

The whole point of splitting one agent into several is lost if each
specialist develops its own voice - the patient would get a differently
shaped reply depending on which internal agent happened to own the turn,
which is exactly the inconsistency this project already fought once (see
the FIXED TEMPLATES section of prompts.py, and graph.py's deterministic
greeting).

So consistency is enforced in TWO places, deliberately:

  1. PROMPT SIDE - `RESPONSE_FORMAT_CONTRACT` below is one block of text,
     built once, embedded IDENTICALLY into every specialist's system
     prompt. Not "similar wording per agent" - the same string object.

  2. CODE SIDE - `normalize_reply()` runs on EVERY final reply, from
     every agent, before the user ever sees it. Anything the model does
     inconsistently despite the prompt (a filler opener here, three
     blank lines there, a re-introduction on turn 9) is removed
     mechanically rather than hoped away.

What consistency does NOT mean here: it does not mean one fixed
language. The prompt's LANGUAGE & DIALECT rule still mirrors whoever is
speaking - an Egyptian patient and an English-speaking patient get the
same STRUCTURE, the same field order, the same templates, the same
emoji, the same one-question rhythm, each in their own language. That
distinction is spelled out in the contract text itself so no agent
"helpfully" flattens everyone into one dialect.
"""

import re
from typing import Optional, Tuple

# ==========================================================
# 1. PROMPT SIDE
# ==========================================================

RESPONSE_FORMAT_CONTRACT = """\
============================================================
RESPONSE FORMAT CONTRACT - IDENTICAL FOR EVERY REPLY
============================================================
Several specialists share this assistant's single identity. The patient
must never be able to tell which one answered, and two patients in the
same situation must receive the same SHAPE of message. Every reply you
write follows this contract exactly.

WHAT MUST BE IDENTICAL EVERY TIME
  1. One voice. You are always the same assistant with the same name and
     persona defined above. Never re-introduce yourself after the
     opening greeting, never mention specialists, routing, tools,
     internal steps, or that anything was "transferred".
  2. Start with content. No filler opener - no standalone "تمام!",
     "طيب!", "حلو!", "أكيد!", "Sure!", "Of course!", "Great!",
     "بالتأكيد!" - and no narration of what you are about to do
     ("لحظة أتأكد", "دعني أتحقق", "Let me check that for you"). Just
     give the answer or ask the question.
  3. Exactly ONE question per reply, always the last line. Never two
     question marks in one message. A single question offering two
     choices is one question and is fine.
  4. Lists are ALWAYS numbered with emoji digits (1️⃣ 2️⃣ 3️⃣ ... 🔟,
     then 1️⃣1️⃣, 1️⃣2️⃣ ...), one item per line, never plain "1." or
     "-", and never a comma-separated run-on inside a sentence. This
     applies to doctors, branches, specialties, days, times, services -
     every list, in every flow.
  5. Structured details use one labelled line each, with the same label
     order every time - and always this order:
        doctor -> specialty -> branch -> day -> date -> time -> reference
     Omit a line you genuinely don't have; never reorder the ones you
     do have, and never merge them into a paragraph.
  6. Times are always 12-hour and taken verbatim from the tool's own
     `time_display`/`date_display`/`weekday_display` fields. Never
     reformat, recompute, or translate them yourself.
  7. Anything in the FIXED TEMPLATES section is reproduced word for
     word, with only its [placeholders] replaced. Same words, same line
     breaks, same emoji, same order - every conversation, every time.
  8. Length is steady: short and clear. One short line of warmth at
     most, then the content, then the one question. Do not write a long
     paragraph for one patient and a single clipped line for another in
     the same situation.
  9. No raw tool output - no JSON, no status codes, no field names, no
     internal ids - ever.

WHAT LEGITIMATELY VARIES (and only this)
  - The LANGUAGE and DIALECT, which always mirror the patient, exactly
    as the LANGUAGE & DIALECT section above requires. Same structure,
    same order, same emoji - expressed in their own way of speaking.
  - The real DATA from this conversation's tool results.
Nothing else varies. Two patients asking the same thing in the same
language get the same message.
"""


# ==========================================================
# 2. CODE SIDE
# ==========================================================

# Standalone acknowledgement openers. Removed ONLY when actual content
# follows - a reply that is nothing but "ok" stays "ok" rather than
# becoming an empty message.
_FILLER_OPENERS = (
    "بالتأكيد", "أكيد", "اكيد", "طبعاً", "طبعا", "بكل تأكيد", "بكل سرور",
    "تمام", "طيب", "حلو", "ماشي", "أبشر", "ابشر", "حاضر", "زين",
    "sure", "of course", "certainly", "absolutely", "great", "okay",
    "ok", "alright", "no problem", "happy to help", "got it", "gotcha",
    "perfect", "awesome", "understood",
)

# Built once: "^(word)[!,.…\s]+" with an alternation of the openers.
_FILLER_OPENER_RE = re.compile(
    r"^\s*(?:" + "|".join(re.escape(word) for word in _FILLER_OPENERS) + r")"
    r"\s*[!,.،…:\-–—]+\s*",
    re.IGNORECASE,
)

# Narration of an action instead of the action's result.
_META_NARRATION_RE = re.compile(
    r"^\s*(?:"
    r"لحظة(?:\s+واحدة)?|"
    r"لحظات|"
    r"ثانية(?:\s+واحدة)?|"
    r"دقيقة(?:\s+واحدة)?|"
    r"(?:دعني|خليني|اسمح\s+لي)\s+\S+[^.،!؟\n]*|"
    r"(?:سوف\s+)?(?:أقوم|اقوم|سأقوم|ساقوم)\s+[^.،!؟\n]*|"
    r"(?:let\s+me|allow\s+me\s+to|i(?:'|’)?ll|i\s+will|i\s+am\s+going\s+to)"
    r"\s+(?:just\s+)?(?:check|look|see|verify|confirm|find|fetch|pull)"
    r"[^.!?\n]*|"
    r"one\s+moment|just\s+a\s+moment|hold\s+on|please\s+wait"
    r")\s*[.!?،؟…]*\s*",
    re.IGNORECASE,
)

# "As an AI..." / "بصفتي مساعد..." style self-reference.
_SELF_REFERENCE_RE = re.compile(
    r"(?:^|(?<=[.!?،؟\n]))\s*"
    r"(?:as an (?:ai|artificial intelligence|automated) [^.!?\n]*[.!?]|"
    r"بصفتي\s+(?:مساعد|مساعدة|ذكاء)[^.،!؟\n]*[.،!؟]|"
    r"كوني\s+(?:مساعد|مساعدة)[^.،!؟\n]*[.،!؟])\s*",
    re.IGNORECASE,
)

# Any internal wording that would leak the multi-agent machinery.
_ROUTING_LEAK_RE = re.compile(
    r"(?:^|(?<=[.!?،؟\n]))\s*"
    r"(?:[^.!?،؟\n]*\b(?:transferring you|handing you (?:over|off)|"
    r"routing you|our (?:booking|cancellation|complaints?) (?:agent|bot|module)|"
    r"specialist agent|sub-?agent)\b[^.!?،؟\n]*[.!?،؟]|"
    r"[^.،!؟\n]*(?:بحوّلك\s+للوكيل|بحولك\s+للوكيل|الوكيل\s+المختص|"
    r"وكيل\s+الحجز|النظام\s+الفرعي)[^.،!؟\n]*[.،!؟])\s*",
    re.IGNORECASE,
)

_TRAILING_SPACES_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_EXCESS_BLANKS_RE = re.compile(r"\n{3,}")


def strip_filler_opener(text: str) -> str:
    """Removes a leading acknowledgement token, but never empties the
    reply: 'Sure, go ahead.' -> 'Go ahead.'; a bare 'ok' stays 'ok'."""

    result = text
    # Loop: models routinely stack two ("تمام، أكيد! ...").
    for _ in range(3):
        candidate = _FILLER_OPENER_RE.sub("", result, count=1)
        if candidate == result:
            break
        if not candidate.strip():
            # Stripping would leave nothing - keep what we had.
            break
        result = candidate

    return result


def strip_meta_narration(text: str) -> str:
    """Drops 'let me check...' / 'لحظة أتأكد' style preambles."""

    candidate = _META_NARRATION_RE.sub("", text, count=1)
    return candidate if candidate.strip() else text


def strip_self_reference(text: str) -> str:
    candidate = _SELF_REFERENCE_RE.sub("", text)
    return candidate if candidate.strip() else text


def strip_routing_leaks(text: str) -> str:
    """Belt-and-braces: the patient must never learn there is more than
    one agent behind this assistant."""

    candidate = _ROUTING_LEAK_RE.sub("", text)
    return candidate if candidate.strip() else text


def strip_repeat_greeting(text: str, greeting: Optional[str]) -> str:
    """Removes a persona re-introduction after the first turn.

    graph.py guarantees the clinic's exact greeting on turn ONE. If a
    later reply opens by repeating it (or its distinctive first line),
    that is the same inconsistency in the other direction, so it is cut
    here.
    """

    if not greeting or not text:
        return text

    greeting_lines = [line.strip() for line in greeting.splitlines() if line.strip()]
    if not greeting_lines:
        return text

    body_lines = text.splitlines()
    removed = 0

    while body_lines and removed < len(greeting_lines):
        head = body_lines[0].strip()
        if head and head in greeting_lines:
            body_lines.pop(0)
            removed += 1
            continue
        break

    if not removed:
        return text

    candidate = "\n".join(body_lines).strip()
    return candidate if candidate else text


def tidy_whitespace(text: str) -> str:
    """Same vertical rhythm in every message: no trailing spaces, never
    more than one blank line, no leading/trailing blank lines."""

    result = _TRAILING_SPACES_RE.sub("", text)
    result = _EXCESS_BLANKS_RE.sub("\n\n", result)
    return result.strip()


def normalize_reply(text: str, greeting: Optional[str] = None) -> Tuple[str, bool]:
    """
    Applies the full contract to one final reply.

    Returns `(normalized_text, changed)`. Callers log `changed` so a
    model that keeps needing correction is visible in the logs rather
    than silently patched forever.

    Deliberately conservative: every individual step refuses to run if
    it would empty the message, so this can never turn a real reply into
    a blank one.
    """

    if not text or not text.strip():
        return text, False

    original = text

    # Filler and narration stack in real output ("تمام! Let me check
    # that for you. ..."), and each one only matches at the START of the
    # message - so one pass in a fixed order would leave whichever came
    # second untouched. Alternating twice clears both orderings.
    result = text
    for _ in range(2):
        result = strip_filler_opener(result)
        result = strip_meta_narration(result)

    result = strip_self_reference(result)
    result = strip_routing_leaks(result)
    result = strip_repeat_greeting(result, greeting)
    result = tidy_whitespace(result)
    result = _restore_leading_capital(original, result)

    if not result.strip():
        return original, False

    return result, result != original


def _restore_leading_capital(original: str, result: str) -> str:
    """Cutting "Sure, " off "Sure, go ahead." leaves a sentence starting
    in lower case, which is its own small inconsistency. Restore the
    capital - but only for Latin script, and only when the original was
    capitalised, so nothing is imposed on Arabic or on a reply that was
    deliberately lower case."""

    if not result or result == original:
        return result

    head = result[0]
    if not ("a" <= head <= "z"):
        return result

    original_head = original.lstrip()[:1]
    if not original_head or not ("A" <= original_head <= "Z"):
        return result

    return head.upper() + result[1:]
