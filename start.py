"""
Production launcher.

WHY THIS FILE EXISTS
--------------------
The obvious start command is:

    uvicorn app:app --host 0.0.0.0 --port $PORT

and it fails on Railway with:

    Error: Invalid value for '--port': '$PORT' is not a valid integer

because `$PORT` is a SHELL feature. When the platform runs the command
directly (exec form, no `sh -c` wrapping it), nothing ever expands the
variable, so uvicorn receives the four literal characters `$PORT` and
rejects them. Whether that wrapping happens depends on how the service
was created - Dockerfile vs. Procfile vs. a custom start command typed
into the dashboard - which is exactly the kind of invisible difference
that makes one deployment work and the next one crash.

So this file removes the shell from the equation entirely: the port is
read in Python, from the environment, where no expansion is needed. The
start command becomes `python start.py`, which has no variables in it at
all and therefore behaves identically under every launcher.

    python start.py

Local development is unaffected - `uvicorn app:app --reload` still works
exactly as before, and is still what the README recommends for local
use, since this file deliberately does not enable --reload.
"""

import os


def _port() -> int:
    """The port to bind, from the platform's own variable.

    Falls back to 8000 for local runs. A malformed value is treated as
    absent rather than fatal: a container that starts on the wrong port
    is debuggable, one that refuses to start at all just shows a crash
    loop with no useful message.
    """

    raw = (os.getenv("PORT") or "").strip()

    if not raw:
        return 8000

    try:
        return int(raw)
    except ValueError:
        print(f"start.py: PORT={raw!r} is not a number - falling back to 8000")
        return 8000


def main() -> None:
    import uvicorn

    port = _port()
    print(f"start.py: serving app:app on 0.0.0.0:{port}")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        # Deliberately off: this is the production entry point. Reload
        # watches the filesystem and forks a second process, neither of
        # which belongs in a container.
        reload=False,
    )


if __name__ == "__main__":
    main()
