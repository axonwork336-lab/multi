"""
Adds a verifier that catches INVENTED BRANCH NAMES in the agent's replies.

Run it once, from your project folder:

    python add_branch_verifier.py

It edits graph.py in place (after writing graph.py.backup), and refuses
to change anything if it can't find an exact anchor - so a version
mismatch fails loudly instead of corrupting the file.

WHAT IT FIXES
-------------
Confirmed real production failure: right after د. طه مبروك was
confirmed, the reply offered three branches - الدقي، زايد، مصر الجديدة -
with NO tool call made that turn at all. One of those branches doesn't
exist for the clinic, and the doctor works at only one of the others.
The patient chose an impossible option and was corrected two messages
later.

Prompt rules did not stop this (it has now happened repeatedly), because
the model wasn't calling a tool and getting it wrong - it was writing
branch names from memory without calling anything. The only thing that
can catch that is a check on the finished reply.

HOW IT WORKS
------------
Every branch name the reply mentions is compared against the branch
names that ACTUALLY appeared in this conversation's tool results, plus
the clinic's own configured branches. Anything else is invented.

DEFAULT IS LOG-ONLY. It writes a WARNING naming the invented branch and
otherwise leaves the reply alone, so you can watch real traffic and see
the false-positive rate before trusting it to intervene. Set

    BRANCH_VERIFIER_STRICT=true

to have it re-ask the model once with a correction instead of sending
the invented list. Start in log-only mode.
"""

import os
import re
import shutil
import sys

TARGET = "graph.py"

VERIFIER_CODE = '''

# ==========================================================
# Invented-branch-name verifier
# ==========================================================
#
# WHY THIS EXISTS: the model has, more than once, listed branches it was
# never given. The worst confirmed case offered "فرع الدقي / فرع زايد /
# فرع مصر الجديدة" for a doctor who works at exactly one branch, with
# zero tool calls made that turn - so there was no wrong tool result to
# blame, and no prompt rule could have been "followed" at the moment it
# mattered, because the model simply wrote names from memory.
#
# Every other guard in this file shapes what the model does BEFORE it
# writes. This one checks what it actually wrote, which is the only
# place a name invented out of nothing can still be caught.
#
# Deliberately narrow: it only looks at branch names, only in final
# replies, and only flags a name that appears NOWHERE in this
# conversation's tool results or in the clinic's own configured
# branches. A name the model got from a tool is never flagged, however
# it's phrased.

_BRANCH_MENTION_RE = re.compile(r"فرع\\s+([^\\n،,.؟?:()\\[\\]0-9️⃣]{2,25})")

# Words that follow "فرع" in ordinary questions rather than naming one -
# "أي فرع تفضل؟" is not a branch called "تفضل". Without this the
# verifier flags its own clinic on a perfectly correct question, and a
# check that cries wolf gets switched off.
_NOT_A_BRANCH_NAME = {
    "تفضل", "تفضلين", "تفضلي", "معين", "معيّن", "تاني", "ثاني", "تانية", "ثانية",
    "قريب", "قريبة", "مناسب", "مناسبة", "محدد", "محددة", "يناسبك", "تحب", "تحبين",
    "اخر", "آخر", "أخرى", "اخرى", "غيره", "غيرها", "كذا", "معينة",
    "تحجز", "تحجزين", "احجز", "أحجز", "فيه", "فيها", "كمان", "ايضا", "أيضا",
    "ولا", "او", "أو", "من", "في", "علي", "على", "عند", "عندنا", "عندكم",
    "متاح", "متاحة", "المتاحة", "المتاح", "بس", "برضه", "برضو", "هو", "هي",
    "العيادة", "العياده", "عيادة", "عياده", "اللي", "الي", "التي", "الذي",
    "تزور", "تزوري", "تختار", "تختاري", "يزور",
}

_NOT_A_BRANCH_NAME_NORM = {tools._normalize_arabic(w) for w in _NOT_A_BRANCH_NAME}

_BRANCH_VERIFIER_STRICT = os.getenv("BRANCH_VERIFIER_STRICT", "false").strip().lower() in ("1", "true", "yes")


def _norm_ar(text: str) -> str:
    """Arabic-aware normalization, so a branch written with a different
    alef/ya/ta-marbuta form still matches the configured spelling.

    Confirmed real false positive: the reply's "الدقي" was flagged as
    invented while the config held the same branch under a slightly
    different spelling - whitespace-only comparison could not see they
    were the same name, and a verifier that accuses correct replies gets
    switched off."""

    return tools._normalize_arabic(_normalize_for_compare(text))


def _known_branch_text(state: AgentState) -> str:
    """Everything this conversation has legitimately been told about
    branches: the raw text of every tool result so far, plus the
    clinic's own configured branch names.

    Kept as one big string and tested with substring containment rather
    than parsed into a name list. Tool results nest branch names under
    several different keys depending on which tool produced them, and a
    parser that missed one would flag a perfectly real branch - a false
    accusation is worse here than a missed one, because it would block
    a correct reply."""

    parts = []

    for msg in state.get("messages", []):
        if getattr(msg, "type", None) == "tool" or getattr(msg, "name", None):
            content = getattr(msg, "content", "")
            if content:
                parts.append(str(content))

    templates = state.get("templates") or {}
    for entry in templates.get("_branch_aliases") or []:
        parts.extend(entry.get("aliases") or [])

    raw_config = state.get("raw_client_config") or {}
    for key, value in raw_config.items():
        if "branch" in key.lower() and isinstance(value, str):
            parts.append(value)

    return _norm_ar(" | ".join(str(p) for p in parts))


def _find_invented_branches(reply_text: str, state: AgentState) -> list:
    """Branch names the reply mentions that this conversation was never
    actually given. Empty list means nothing to flag."""

    if not reply_text or "فرع" not in reply_text:
        return []

    known = _known_branch_text(state)
    if not known:
        # Nothing to compare against (no tool results yet, no config) -
        # stay silent rather than flagging everything.
        return []

    invented = []
    for match in _BRANCH_MENTION_RE.finditer(reply_text):
        name = _norm_ar(match.group(1))
        if not name or len(name) < 3:
            continue
        # A branch name is the run of words right after "فرع", stopping
        # at the first word that clearly isn't part of a name ("أي فرع
        # تفضل تحجز فيه؟" -> nothing; "فرع المعادي كمان" -> "المعادي").
        # Names are 1-3 words in practice, so anything longer is prose
        # that happened to follow the word "فرع", not a name.
        words = []
        for word in name.split():
            if word in _NOT_A_BRANCH_NAME or word in _NOT_A_BRANCH_NAME_NORM:
                break
            words.append(word)
            if len(words) == 3:
                break
        if not words:
            continue
        name = " ".join(words)
        if len(name) < 3:
            continue
        if name in known:
            continue
        # Also accept a partial: "الشيخ زايد" mentioned as "زايد".
        if any(part in known for part in name.split() if len(part) >= 3):
            continue
        if name not in invented:
            invented.append(name)

    return invented


_BRANCH_CORRECTION_DIRECTIVE = (
    "============================================================\\n"
    "YOU LISTED A BRANCH THAT DOES NOT EXIST - REWRITE YOUR REPLY\\n"
    "============================================================\\n"
    "Your previous draft named at least one branch that has NOT appeared "
    "in any tool result in this conversation and is not one of this "
    "clinic's configured branches: {names}\\n\\n"
    "You may only name branches that came from a tool result. If you "
    "need the branch list for a confirmed doctor, call the tool that "
    "returns it rather than writing names from memory - a doctor's "
    "branches are returned to you as `branchesForDoctor` the moment the "
    "doctor is confirmed, and `list_branches_for_specialty` returns them "
    "too.\\n\\n"
    "Rewrite the reply now using ONLY real branches, or call the tool "
    "first if you don't have them.\\n\\n"
)

_DATE_IN_REPLY_RE = re.compile(r"\\d{1,2}[-/]\\d{1,2}[-/]\\d{2,4}")
_TIME_IN_REPLY_RE = re.compile(r"\\d{1,2}:\\d{2}")

_AVAILABILITY_TOOLS = (
    "list_available_days_for_booking", "get_available_slots_for_booking",
    "get_available_reschedule_slots", "resolve_available_day",
    "get_doctor_schedule", "get_doctor_schedule_for_booking",
    "find_best_doctor_in_specialty", "lookup_appointment",
    "check_booking_status", "create_new_booking",
)


def _reply_invents_availability(reply_text, state) -> bool:
    """True when the reply states an appointment date or times that no
    availability tool in this conversation ever returned.

    WHY: confirmed real production failure, the worst yet - after a
    branch was picked the reply offered "يوم الثلاثاء 30-05-2024" and a
    list of times, with NO availability tool called at all. The date was
    in the PAST and every time was invented; a patient could have
    accepted an appointment that existed nowhere. The date/time
    directives elsewhere only fire AFTER a tool has run, so when the
    model skips the tool entirely nothing else can catch it."""

    if not reply_text:
        return False

    dates = _DATE_IN_REPLY_RE.findall(reply_text)
    times = _TIME_IN_REPLY_RE.findall(reply_text)
    if not dates and not times:
        return False

    tool_text = []
    for msg in state.get("messages", []):
        if getattr(msg, "name", None) in _AVAILABILITY_TOOLS:
            content = getattr(msg, "content", "")
            if content:
                tool_text.append(str(content))

    if not tool_text:
        return True

    joined = " ".join(tool_text)

    for value in dates:
        parts = [p.lstrip("0") for p in re.split(r"[-/]", value)]
        if not any(p and p in joined for p in parts):
            return True

    for value in times:
        hour = value.split(":")[0].lstrip("0")
        if hour and hour not in joined:
            return True

    return False


_AVAILABILITY_CORRECTION_DIRECTIVE = (
    "============================================================\\n"
    "YOU STATED A DATE/TIME NO TOOL GAVE YOU - REWRITE YOUR REPLY\\n"
    "============================================================\\n"
    "Your previous draft named an appointment date or times that no "
    "availability tool returned in this conversation. Those are "
    "invented - a patient could accept an appointment that does not "
    "exist anywhere in the booking system.\\n\\n"
    "NEVER state a date or time from memory, from reasoning, or from a "
    "doctor's general working hours. Call "
    "`list_available_days_for_booking` for the real days, then "
    "`get_available_slots_for_booking` with that day's own from_date/"
    "to_date for the real times, and use ONLY what they return.\\n\\n"
    "Call the tool now instead of writing a date yourself.\\n\\n"
)

'''

HOOK_ANCHOR = """        if normalized != response.content:
            response = AIMessage(content=normalized)"""

HOOK_REPLACEMENT = '''        # LAST LINE OF DEFENCE: did this reply name a branch nobody ever
        # gave it? See _find_invented_branches - this is the only check
        # that can catch a name written from memory with no tool call
        # behind it.
        if _reply_invents_availability(normalized, state):
            logger.error(
                "agent[%s]: reply stated an appointment date/time that NO availability tool "
                "returned - this is a fabricated appointment | strict_mode=%s | reply=%r",
                agent_name, _BRANCH_VERIFIER_STRICT, normalized,
            )
            if _BRANCH_VERIFIER_STRICT:
                retry = _llm_for(agent_name).invoke(
                    [SystemMessage(content=_AVAILABILITY_CORRECTION_DIRECTIVE + system_content)] + history
                )
                if getattr(retry, "tool_calls", None):
                    updates["messages"] = [retry]
                    updates["target_language"] = target_language
                    return updates
                if retry.content and not _reply_invents_availability(retry.content, state):
                    logger.info("agent[%s]: fabricated availability corrected on retry", agent_name)
                    normalized = _emojify_list_numbers(retry.content)

        invented = _find_invented_branches(normalized, state)
        if invented:
            logger.warning(
                "agent[%s]: reply named branch(es) that appear in NO tool result and "
                "are not configured for this client: %s | strict_mode=%s | reply=%r",
                agent_name, invented, _BRANCH_VERIFIER_STRICT, normalized,
            )

            if _BRANCH_VERIFIER_STRICT:
                # Re-ask once, with the correction at the very top so it
                # outranks everything else for this retry.
                correction = _BRANCH_CORRECTION_DIRECTIVE.format(names=", ".join(invented))
                retry = _llm_for(agent_name).invoke(
                    [SystemMessage(content=correction + system_content)] + history
                )
                if getattr(retry, "tool_calls", None):
                    # It chose to go and fetch the real list - let the
                    # normal tools loop run instead of sending anything.
                    updates["messages"] = [retry]
                    updates["target_language"] = target_language
                    return updates
                if retry.content:
                    still_invented = _find_invented_branches(retry.content, state)
                    if still_invented:
                        logger.error(
                            "agent[%s]: reply STILL named invented branch(es) after correction: %s",
                            agent_name, still_invented,
                        )
                    else:
                        logger.info("agent[%s]: invented branches corrected on retry", agent_name)
                        normalized = _emojify_list_numbers(retry.content)

        if normalized != response.content:
            response = AIMessage(content=normalized)'''


def main() -> None:
    if not os.path.exists(TARGET):
        print(f"  ERROR: {TARGET} not found. Run this from your project folder.")
        sys.exit(1)

    with open(TARGET, encoding="utf-8") as handle:
        src = handle.read()

    if "_find_invented_branches" in src:
        print("  Already patched - nothing to do.")
        return

    problems = []
    if "def _run_agent(" not in src:
        problems.append("could not find _run_agent()")
    if HOOK_ANCHOR not in src:
        problems.append("could not find the reply-normalization hook point")
    if "def _normalize_for_compare(" not in src:
        problems.append("could not find _normalize_for_compare()")
    if "^import os" not in src and "\nimport os" not in src and not src.startswith("import os"):
        pass  # handled below

    if problems:
        print("  ERROR: this graph.py doesn't look like the expected version:")
        for p in problems:
            print(f"    - {p}")
        print("  Nothing was changed.")
        sys.exit(1)

    shutil.copy(TARGET, TARGET + ".backup")
    print(f"  backup written: {TARGET}.backup")

    # `os` is used by the verifier for its env-var flag.
    if not re.search(r"^import os$", src, re.MULTILINE):
        src = src.replace("import json\n", "import json\nimport os\n", 1)
        print("  added: import os")

    # Insert the verifier just above _run_agent, so it's defined before use.
    src = src.replace("def _run_agent(", VERIFIER_CODE + "\ndef _run_agent(", 1)
    print("  added: branch verifier functions")

    src = src.replace(HOOK_ANCHOR, HOOK_REPLACEMENT, 1)
    print("  added: verifier call inside _run_agent")

    with open(TARGET, "w", encoding="utf-8") as handle:
        handle.write(src)

    import ast
    try:
        ast.parse(src)
    except SyntaxError as exc:
        print(f"\n  ERROR: the result doesn't parse ({exc}). Restoring the backup.")
        shutil.copy(TARGET + ".backup", TARGET)
        sys.exit(1)

    print("\n  Done. graph.py parses cleanly.")
    print("  Running in LOG-ONLY mode - watch for this line in your logs:")
    print("    agent[...]: reply named branch(es) that appear in NO tool result ...")
    print("  Once you're happy with the false-positive rate, set BRANCH_VERIFIER_STRICT=true.")


if __name__ == "__main__":
    main()
