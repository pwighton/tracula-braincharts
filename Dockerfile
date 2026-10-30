FROM python:3.11.13-slim-bookworm

# curl/unzip for fetching the models
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# braincharts FA model files: fetched from the pinned upstream commit, checksum-verified.
ENV TRACULA_BRAINCHARTS_MODEL_DIR=/opt/tracula-braincharts/braincharts
COPY tracula_braincharts/data/model-checksums.sha256 /opt/tracula-braincharts/src/tracula_braincharts/data/model-checksums.sha256
COPY tools/fetch_models.sh /opt/tracula-braincharts/src/tools/fetch_models.sh
RUN bash /opt/tracula-braincharts/src/tools/fetch_models.sh "${TRACULA_BRAINCHARTS_MODEL_DIR}"

# Python environment (pcntoolkit 0.35 supports Python 3.10-3.12)
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Python dependencies. pcntoolkit requires torch; the CPU-only build avoids pulling in CUDA.
# constraints.txt pins the rest of the resolved set (regenerate with tools/freeze_constraints.sh).
COPY constraints.txt /opt/tracula-braincharts/src/constraints.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
    && pip install --no-cache-dir -c /opt/tracula-braincharts/src/constraints.txt pcntoolkit==0.35.0

# Python package
COPY pyproject.toml README.md LICENSE /opt/tracula-braincharts/src/
COPY tracula_braincharts /opt/tracula-braincharts/src/tracula_braincharts
RUN pip install --no-cache-dir -c /opt/tracula-braincharts/src/constraints.txt "/opt/tracula-braincharts/src[dev]"

# Tests are copied so the suite can be run inside the image:  docker run --entrypoint pytest tracula-braincharts
COPY tests /opt/tracula-braincharts/src/tests
WORKDIR /opt/tracula-braincharts/src

# The container is normally run as the calling user (--user $(id -u):$(id -g)), who has no home
# directory inside the image; point caches somewhere writable.
ENV MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    PYTENSOR_FLAGS=base_compiledir=/tmp/pytensor

# Provenance: set by the Makefile at build time
ARG GIT_SHA=unknown
ARG IMAGE_TAG=pwighton/tracula-braincharts:dev
ENV TRACULA_BRAINCHARTS_GIT_SHA=${GIT_SHA} \
    TRACULA_BRAINCHARTS_IMAGE=${IMAGE_TAG}

ENTRYPOINT ["tracula-braincharts"]
CMD ["--help"]
