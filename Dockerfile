FROM python:3.11.13-slim-bookworm

# curl/unzip for fetching FSL and the models; dc for TBSS (tbss_1_preproc, tbss_3_postreg); libquadmath0 for the
# libgfortran.so.3 bundled with FSL
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    unzip \
    dc \
    libquadmath0 \
    && rm -rf /var/lib/apt/lists/*

# FSL 6.0.5.2 (the training data's TBSS used FSL 6.0.5), installed the way FSL's installer did: the full release
# tarball, then FSL's own etc/fslconf/post_install.sh, which builds $FSLDIR/fslpython (FSL's conda environment:
# fsl_sub, fslpy's imcp/immv/imglob, FSLeyes, ...); the packages' post-link scripts put wrappers for their
# commands in $FSLDIR/bin. Two changes, because FSL's 2022 setup no longer installs as is:
#  - fslpython_install.sh downloads "Miniconda3-latest", which today stops at Anaconda's terms-of-service prompt.
#    It gets a pinned Miniforge (conda-forge only) instead, downloaded here, checksum-verified and handed to the
#    script as a file:// URL.
#  - fslpython_environment.yml is replaced by tools/fslpython_environment.yml: FSL's list without the `defaults`
#    channel and fsl-tirl (see that file).
# post_install.sh exits 0 even when a step fails, so the result is checked.
ENV FSLDIR=/usr/local/fsl
ARG FSL_TARBALL=fsl-6.0.5.2-centos7_64.tar.gz
ARG FSL_SHA256=dd41fcc3f457617d750394e9872b1a87c34838003b9929808dea280cd9d7a0d2
ARG MINIFORGE_VERSION=26.5.3-0
ARG MINIFORGE_SHA256=14db468222ad564658656f769506056209b6dc375f5e7dfd31eb5ebbf08fa529
COPY tools/fslpython_environment.yml /opt/tracula-braincharts/src/tools/fslpython_environment.yml
RUN set -eux; \
    curl -fsSL -o /tmp/fsl.tar.gz "https://fsl.fmrib.ox.ac.uk/fsldownloads/${FSL_TARBALL}"; \
    echo "${FSL_SHA256}  /tmp/fsl.tar.gz" | sha256sum -c -; \
    tar -xzf /tmp/fsl.tar.gz -C "$(dirname "${FSLDIR}")"; \
    rm /tmp/fsl.tar.gz; \
    installer="Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh"; \
    mkdir /tmp/miniforge; \
    curl -fsSL -o "/tmp/miniforge/${installer}" \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${installer}"; \
    echo "${MINIFORGE_SHA256}  /tmp/miniforge/${installer}" | sha256sum -c -; \
    conf="${FSLDIR}/etc/fslconf"; \
    sed -i -e 's|^miniconda_url=.*|miniconda_url="file:///tmp/miniforge"|' \
        -e "s|Miniconda3-latest-Linux-x86_64.sh|${installer}|" "${conf}/fslpython_install.sh"; \
    cp /opt/tracula-braincharts/src/tools/fslpython_environment.yml "${conf}/fslpython_environment.yml"; \
    "${conf}/post_install.sh" -f "${FSLDIR}" -q; \
    for cmd in fslpython fsl_sub imcp immv imglob; do test -x "${FSLDIR}/bin/${cmd}"; done; \
    "${FSLDIR}/fslpython/bin/conda" clean -afy; \
    rm -rf /tmp/miniforge
# as $FSLDIR/etc/fslconf/fsl.sh sets them
ENV PATH="${FSLDIR}/bin:${PATH}" \
    FSLOUTPUTTYPE=NIFTI_GZ \
    FSLMULTIFILEQUIT=TRUE \
    FSLTCLSH=${FSLDIR}/bin/fsltclsh \
    FSLWISH=${FSLDIR}/bin/fslwish

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
# directory inside the image (HOME=/, not writable); point HOME and caches somewhere writable.
# That user has no passwd entry either: fsl_sub (tbss_2_reg) calls getpass.getuser() on import, which reads
# LOGNAME/USER before the passwd database, so set them.
# PyTensor (imported via pymc) isn't used for BLR: cxx= skips its C compiler and the missing-g++ warning.
ENV HOME=/tmp \
    LOGNAME=tracula-braincharts \
    USER=tracula-braincharts \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    PYTENSOR_FLAGS=base_compiledir=/tmp/pytensor,cxx=

# Provenance: set by the Makefile at build time
ARG GIT_SHA=unknown
ARG IMAGE_TAG=pwighton/tracula-braincharts:dev
ENV TRACULA_BRAINCHARTS_GIT_SHA=${GIT_SHA} \
    TRACULA_BRAINCHARTS_IMAGE=${IMAGE_TAG}

ENTRYPOINT ["tracula-braincharts"]
CMD ["--help"]
