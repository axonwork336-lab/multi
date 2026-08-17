"""
Slices the ALREADY-BUILT system prompt into its named sections.

WHY IT WORKS THIS WAY: `prompts.AGENT_SYSTEM_PROMPT_TEMPLATE` is one
~90 KB f-string full of `{placeholders}` that `build_system_prompt()`
fills in from the tenant's CSV row. Cutting the TEMPLATE into per-agent
pieces would mean rewriting that file and re-deriving which placeholder
belongs to which piece - a large, high-risk edit to the single most
important asset in the project.

So instead we let `build_system_prompt()` run EXACTLY as before,
untouched, and split its OUTPUT on the `====` banner headers the prompt
already uses. Consequences:

  - prompts.py stays authoritative. Editing a flow there automatically
    flows into the right specialist, with no mapping to keep in sync.
  - A tenant's CSV values are already substituted before we split, so
    every specialist sees its own tenant's real wording.
  - If a header is ever renamed and a section goes missing, we fail
    SAFE: the caller falls back to the whole prompt (see registry.py),
    which is precisely the old single-agent behaviour.
"""

import hashlib
import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


# A banner header in the prompt looks like:
#     ============================================================
#     NEW BOOKING FLOW (create a brand new appointment)
#     ============================================================
_HEADER_RE = re.compile(
    r"^={40,}[ \t]*\n(?P<title>[^\n]+)\n={40,}[ \t]*\n",
    re.MULTILINE,
)


# Short, stable keys used everywhere else in this package, mapped from a
# prefix of the section's real title. Prefix matching (not equality) so
# a later wording tweak to a heading's tail doesn't silently drop a
# section.
_TITLE_PREFIX_TO_KEY = (
    ("LANGUAGE & DIALECT", "language"),
    ("DEFAULT DIALECT", "dialect"),
    ("REFERENCE PHRASES", "reference_phrases"),
    ("FIXED TEMPLATES", "fixed_templates"),
    ("YOUR JOB", "your_job"),
    ("MEDICAL GUIDANCE FLOW", "medical"),
    ("CONVERSATION FLOW", "cancel"),
    ("RESCHEDULE FLOW", "reschedule"),
    ("GENERAL HOSPITAL INFO", "faq"),
    ("DOCTOR / BRANCH INFO", "entity_info"),
    ("NEW BOOKING FLOW", "booking"),
    ("COMPLAINT FLOW", "complaint"),
    ("GLOBAL HARD RULES", "hard_rules"),
)

# The opening line ("You are {agent_name}, the ... assistant for
# {clinic_name}.") sits before any header.
PREAMBLE_KEY = "preamble"

# Every key the rest of the package expects to exist.
REQUIRED_KEYS = frozenset(
    [PREAMBLE_KEY] + [key for _, key in _TITLE_PREFIX_TO_KEY]
)


def _key_for_title(title: str) -> str:
    clean = title.strip()
    for prefix, key in _TITLE_PREFIX_TO_KEY:
        if clean.upper().startswith(prefix):
            return key
    # Unknown heading: keep it under a derived key so it is never lost -
    # `registry.py` appends any unrecognised section to every agent
    # rather than dropping it.
    return "extra:" + re.sub(r"[^a-z0-9]+", "_", clean.lower()).strip("_")


def _banner(title: str) -> str:
    bar = "=" * 60
    return f"{bar}\n{title}\n{bar}\n"


# Cache keyed by a hash of the built prompt itself, NOT by client_id.
# That matters: graph.load_config deliberately rebuilds the prompt every
# single turn so edits to prompts.py or the CSVs reach conversations
# already in progress. Hashing the content preserves that - change the
# prompt and the cache key changes with it - while still avoiding a
# re-split on every turn of every session.
_CACHE: Dict[str, Dict[str, str]] = {}
_CACHE_MAX_ENTRIES = 32


def split_sections(built_prompt: str) -> Dict[str, str]:
    """
    Returns {key: section_text}, where each section_text INCLUDES its
    own `====` banner so the pieces can simply be concatenated back
    together in any order without losing the visual structure the model
    was trained on for this prompt.
    """

    digest = hashlib.sha256(built_prompt.encode("utf-8")).hexdigest()

    cached = _CACHE.get(digest)
    if cached is not None:
        return cached

    sections: Dict[str, str] = {}
    order: List[str] = []

    matches = list(_HEADER_RE.finditer(built_prompt))

    preamble = built_prompt[: matches[0].start()] if matches else built_prompt
    sections[PREAMBLE_KEY] = preamble.strip()
    order.append(PREAMBLE_KEY)

    for index, match in enumerate(matches):
        title = match.group("title").strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(built_prompt)

        key = _key_for_title(title)
        body = built_prompt[start:end].strip("\n")

        text = _banner(title) + body

        if key in sections:
            # Duplicate heading - concatenate rather than overwrite, so
            # no instruction is ever silently lost.
            sections[key] = sections[key] + "\n\n" + text
        else:
            sections[key] = text
            order.append(key)

    sections["__order__"] = "\n".join(order)

    missing = REQUIRED_KEYS - set(sections)
    if missing:
        logger.warning(
            "agents.sections: expected prompt section(s) not found: %s. "
            "Specialist agents will fall back to the full prompt.",
            sorted(missing),
        )

    if len(_CACHE) >= _CACHE_MAX_ENTRIES:
        _CACHE.clear()
    _CACHE[digest] = sections

    return sections


def has_all_required(sections: Dict[str, str]) -> bool:
    """True when every section the specialists rely on was found."""

    return not (REQUIRED_KEYS - set(sections))


def extra_keys(sections: Dict[str, str]) -> List[str]:
    """Unrecognised sections, which every agent receives verbatim."""

    return sorted(k for k in sections if k.startswith("extra:"))
