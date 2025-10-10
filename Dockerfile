# =========================================
# floc: Genomic clustering tool
# Python 3.10, wheels-only installs
# =========================================
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Force wheels; avoids compiling SciPy / scikit-learn / sourmash
    PIP_ONLY_BINARY=:all:

# Minimal runtime packages (CA certs, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Dep layer: copy just metadata so we can cache dependency resolution
COPY pyproject.toml README.md LICENSE /app/

# If your package code uses src/ for PEP 517 build backends to inspect,
# you can optionally pre-copy a minimal src tree; usually not required.

# Upgrade pip + build backend (PEP 517)
RUN python -m pip install --upgrade pip build

# Create a virtual environment (optional but keeps global site clean)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Resolve & download wheels for project (without source code yet)
# This builds a wheel using only the metadata/config; if build requires code,
# the next layer will rebuild the wheel after copying the source.
RUN python -m build --wheel --outdir /tmp/dist || true

# ---- App layer: now copy source and build/install the package
COPY src/ /app/src/
# If you have additional files needed at build time, copy them too:
# COPY MANIFEST.in /app/
# COPY other_files/ /app/other_files/

# Build wheel and install it
RUN python -m build --wheel --outdir /tmp/dist \
 && pip install /tmp/dist/*.whl

# Create non-root user
RUN useradd -ms /bin/bash appuser
USER appuser

# Default working directory for runtime
WORKDIR /workspace

# The package exposes a console_script entrypoint named "floc"
ENTRYPOINT ["floc"]
# Or, if you prefer explicit python module invocation:
# ENTRYPOINT ["python", "-m", "floc.cli"]
