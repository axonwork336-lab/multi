# Guest Booking Cancellation Agent - LLM-Tool-Calling Architecture

**This is a full architecture rewrite**, not an incremental patch - done
deliberately, at explicit request, reversing an earlier design choice.
See `*.pre_rewrite_backup` files in this folder for the previous
deterministic-router version.

## Multi-agent architecture (current)

The single `agent` node described below has been split into a supervisor
plus seven specialists - see **[README_MULTIAGENT.md](README_MULTIAGENT.md)**
for the full description, the routing table and the rollback flags.

    load_config -> router -> agent_<specialist> <-> tools -> END

Nothing in the deployment, the `/chat` contract, `api.py`, `tools.py` or
the CSVs changed. `MULTI_AGENT_ENABLED=false` restores the exact graph
documented in the rest of this file.

## What changed

| File | Status |
|---|---|
| `api.py` | **Unchanged** - raw HTTP calls to Company Booking APIs |
| `config.py` | **Unchanged** - client_config.csv/dialect_templates.csv loading, base_url resolution |
| `client_config.csv`, `dialect_templates.csv` | **Unchanged** |
| `state.py` | Rewritten - 38-field state machine -> `{client_id, session_id, channel_phone, templates, system_prompt, messages}` |
| `tools.py` | Rewritten - every tool now returns `{"status": "..."}` + data only, never a sentence |
| `prompts.py` | Rewritten - one comprehensive LLM system prompt (STEP 1-4 flow + hard rules) instead of 4 narrow classifier prompts |
| `graph.py` | Rewritten - `load_config -> agent <-> tools -> END` loop instead of a 20-node deterministic router |
| `main.py` | Rewritten - one function, `send_message()`, instead of a fixed set of start/resume functions |
| `app.py` | Rewritten - thin wrapper around `main.send_message()`, response shape simplified to `{"reply": "..."}` |
| `requirements.txt` | Added `langchain-openai` |

## Why each file changed (mapped to your Problems 1-7)

- **Problem 3 (tool-first -> LLM-first)**: `graph.py`'s router/20 nodes are gone; there's now one `agent` node (the LLM, bound to every tool) and one `tools` node. The LLM decides when to call a tool via `route_after_agent` checking `tool_calls` on its latest message - never the other way around.
- **Problem 7 (tools return data only)**: every function in `tools.py` now returns a plain `{"status": ...}` dict (`"found_one"`, `"otp_required"`... — actually `"otp_sent"`/`"otp_valid"`/`"otp_invalid"` etc.) with structured data alongside, never a formatted sentence. `prompts.py`'s system prompt is explicit: *"NEVER show raw tool output... always translate it into a natural sentence."*
- **Problem 4/6 (understand intent first, distinguish input types via the LLM)**: STEP 1 of the system prompt requires asking which method before doing anything; the LLM (not a heuristic) decides whether free text is a ref, a phone, an OTP, a selection, or a confirmation, because it now has full conversation context instead of one message in isolation.
- **Problem 5 (natural phone-format guidance)**: `validate_phone_format` returns `{"status": "invalid"}` only; the LLM composes the actual guidance sentence, in-language, per the system prompt's STEP 2 instructions.
- **Problems 1/2 (client_config/dialect_templates)**: `load_config` (in `graph.py`) still calls the same, unchanged `config.get_messages()` and now feeds `_clinic_name`, `_agent_name`, `_dialect_instruction`, and `_phone_example` directly into the system prompt (`prompts.build_system_prompt`) - so every single reply is generated under that tenant's persona/dialect instruction, not just certain canned strings.

## Important operational consequence (please read)

**`OPENAI_API_KEY` is now mandatory**, not optional. The previous hybrid design could run with zero LLM calls (heuristics did everything); this one cannot - the LLM makes every decision, so without a real key the agent cannot function at all, in local dev, on Railway, or anywhere else.

## API contract change (n8n needs to know this)

The old `/chat` response had `status`/`interrupt`/`appointments` fields for n8n to branch on. **That's gone.** The new contract is just:

Request (unchanged):
```json
{"session_id": "123", "client_id": "Dar El Oyoun-demo", "message": "...", "channel_phone": "..."}
```

Response (simplified):
```json
{"reply": "..."}
```

There is no more "interrupt type" to branch on in n8n - every turn returns exactly one natural-language reply string, whether the conversation is asking a follow-up question or confirming a cancellation. Just forward `reply` to Messenger every time; no branching needed.

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in a real OPENAI_API_KEY - required now
uvicorn app:app --reload --port 8000
```

```bash
python3 test_agent_graph.py   # graph-level tests (scripted fake LLM, mocked Company APIs)
python3 test_app_http.py      # HTTP-level tests (same approach, over FastAPI TestClient)
python3 main.py                # CLI - same free-text conversation, new internals
```

## Railway / GitHub / n8n deployment

Unchanged from before: `Procfile` (`web: uvicorn app:app --host 0.0.0.0 --port $PORT`), push to GitHub, Railway "Deploy from GitHub repo", set env vars (now `OPENAI_API_KEY` is required, not optional) in Railway's Variables tab, generate a domain, point n8n's HTTP Request node at `<domain>/chat`. See `*.pre_rewrite_backup`-adjacent history for the detailed step list - the deployment mechanics themselves did not change, only the app's internal behavior and the `/chat` response shape (see above).
