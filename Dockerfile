# Unfacet: self-hosted mesh -> STEP converter
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    NUMBA_CACHE_DIR=/srv/.cache/numba

# OpenCascade's wheel pulls in VTK, which needs a few X/GL client libraries even headless
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglu1-mesa libxrender1 libxext6 libx11-6 libgomp1 libfontconfig1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
RUN useradd --system --uid 1000 --home /srv unfacet && mkdir -p /srv/.cache/numba && chown -R unfacet /srv
USER unfacet
# compile the fitting kernels into the image; if the host CPU differs they recompile once at start-up
RUN python -c "from app.core import jit; jit.warmup()"

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -fs http://localhost:8080/api/health || exit 1

# Sessions live in memory, so keep a single web process; heavy work fans out to CPU_WORKERS processes.
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "*"]
