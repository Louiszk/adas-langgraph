# This image contains only the stable sandbox runtime.
FROM python:3.11-slim

ARG ADAS_SANDBOX_DEPENDENCIES

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The build argument is derived from config.settings.dependencies.
# Docker caches this layer, and sandbox.py gives dependency changes a new image tag.
RUN python -m pip install --no-cache-dir ${ADAS_SANDBOX_DEPENDENCIES}
