"""
Multi-agent layer for the Guest Booking Agent.

This package turns the previous SINGLE agent (one 90 KB system prompt +
28 tools handling six unrelated flows) into a supervisor pattern:

    router (deterministic)  ->  one specialist agent  ->  tools  -> ...

Nothing in `api.py`, `tools.py`, `config.py`'s CSV loading, `rag.py`, or
the CSV files themselves is touched. `prompts.py`'s big template is not
rewritten either - it is SLICED at runtime into its existing sections,
and each specialist is handed the shared core plus only the flow it
owns.

Modules
-------
sections          : splits the built system prompt into named sections
registry          : the specialist definitions (prompt sections + tools)
router            : the supervisor - decides who owns the current turn
response_contract : the single output format every agent must produce
"""

from agents.registry import (
    AGENT_NAMES,
    AGENT_SPECS,
    CONCIERGE,
    AgentSpec,
    get_spec,
    tools_for,
)
from agents.response_contract import (
    RESPONSE_FORMAT_CONTRACT,
    normalize_reply,
)
from agents.router import route_turn

__all__ = [
    "AGENT_NAMES",
    "AGENT_SPECS",
    "CONCIERGE",
    "AgentSpec",
    "get_spec",
    "tools_for",
    "RESPONSE_FORMAT_CONTRACT",
    "normalize_reply",
    "route_turn",
]
