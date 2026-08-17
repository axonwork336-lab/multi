# Deterministic build for Railway.
#
# WHY THIS EXISTS: Railway's railpack builder injects the service's
# environment variables into the BUILD as secrets, and aborts the whole
# build when one of them can't be resolved ("secret X not found") - even
# though this application needs none of them at build time. Every value
# it reads has a default in config.py and is read at RUNTIME, so a build
# should never depend on them at all.
#
# Railway uses this Dockerfile in preference to railpack whenever it is
# present, which removes that failure mode entirely. Environment
# variables are still injected normally when the container runs.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so this layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code, CSV config, and the per-client knowledge bases.
COPY . .

# Railway sets PORT at runtime; 8000 is only the local default.
EXPOSE 8000

# No shell, no variable expansion: start.py reads PORT itself. See the
# comment at the top of that file for why $PORT in a start command is
# unreliable across launchers.
CMD ["python", "start.py"]
