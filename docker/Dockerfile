# syntax=docker/dockerfile:1
#
# fpv toolchain image (phase 1): NFIQ 2 v2.3.0 + NBIS 5.0.0 Bozorth3 + Python 3.12 / pyfing 0.7.3.
# Build:  docker compose build            (or: docker build -f docker/Dockerfile .)
# All downloads come from official sources and are verified against pinned SHA-256 digests.

############################
# Stage 1: NBIS (Bozorth3) #
############################
FROM ubuntu:24.04 AS nbis-builder

# Official NIST NBIS distribution (NIGOS). NIST does not publish a digest; this one was recorded
# when the archive was first downloaded and pins the exact bytes.
ARG NBIS_URL=https://nigos.nist.gov/nist/nbis/nbis_v5_0_0.zip
ARG NBIS_SHA256=0adf8ab0f6b0e4208de50ca00ba21d3d77112ecd66288757ddfed21f6bee92c3

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl unzip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN curl -fsSL --retry 3 -o nbis.zip "${NBIS_URL}" \
 && echo "${NBIS_SHA256}  nbis.zip" | sha256sum -c - \
 && unzip -q nbis.zip

WORKDIR /build/Rel_5.0.0
# - setup.sh requires an existing install dir.
# - `-m64` is an x86-only flag (breaks aarch64 gcc); -fPIC is kept.
# - `-fcommon` restores the pre-GCC10 tentative-definition behaviour NBIS relies on.
# - Bozorth3 only links its own libbozorth3.a and libm, so only commonnbis + bozorth3 are built.
RUN mkdir -p /opt/nbis \
 && ./setup.sh /opt/nbis --without-X11 --64 \
 && sed -i -E 's/^(ARCH_FLAG[[:space:]]*:=).*/\1 -fPIC/' rules.mak \
 && sed -i -E '/^CFLAGS[[:space:]]*:=/ s/$/ -fcommon/' rules.mak \
 && grep -q -- '-fcommon' rules.mak \
 && make config PACKAGES="commonnbis bozorth3" \
 && for p in commonnbis bozorth3; do (cd "$p" && make cpheaders) || exit 1; done \
 && (cd bozorth3 && make libs && make bins) \
 && install -D -m 0755 bozorth3/bin/bozorth3 /opt/nbis/bin/bozorth3 \
 && /opt/nbis/bin/bozorth3 -h 2>&1 | grep -q "Usage" \
 && echo "5.0.0 sha256:${NBIS_SHA256}" > /opt/nbis/VERSION

#####################
# Stage 2: runtime  #
#####################
FROM ubuntu:24.04 AS runtime

ARG TARGETARCH
ARG GIT_COMMIT=unknown
ARG UID=1000
ARG GID=1000

# NFIQ 2 v2.3.0 official GitHub release packages. NIST publishes ubuntu24 packages for amd64 only;
# for arm64 the official Debian-12 (rpos12) arm64 package is used, whose binary runs on ubuntu 24.04
# once the Ubuntu-named equivalents of its shared libraries are installed (verified with ldd below).
# Digests were recorded at first download (GitHub publishes none for these assets).
ARG NFIQ2_VERSION=2.3.0
ARG NFIQ2_BASE_URL=https://github.com/usnistgov/NFIQ2/releases/download/v2.3.0
ARG NFIQ2_DEB_AMD64=nfiq2_2.3.0-1_ubuntu24_amd64.deb
ARG NFIQ2_SHA256_AMD64=f8d3b7eec0d96d6e46416bf717ecfd3982b8aa328f700445ebf5fe48271c7c22
ARG NFIQ2_DEB_ARM64=nfiq2_2.3.0-1_rpos12_arm64.deb
ARG NFIQ2_SHA256_ARM64=7f7f114717f78e0cf1a8bcc3e61c5d5f246c53ee9747dafba51a47f2dd508ebd

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    KERAS_BACKEND=tensorflow \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_DETERMINISTIC_OPS=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    CUDA_VISIBLE_DEVICES="" \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    GIT_COMMIT=${GIT_COMMIT}

# Python 3.12 + NFIQ 2 runtime libraries (Ubuntu 24.04 names).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl \
      python3.12 python3.12-venv \
      libglib2.0-0t64 libgomp1 \
      libdb5.3++t64 libjpeg62 libjpeg-turbo8 libopenjp2-7 libpng16-16t64 \
      libsqlite3-0 libssl3t64 libtiff6 zlib1g \
 && rm -rf /var/lib/apt/lists/*

# NFIQ 2, selected by TARGETARCH, checksum-verified.
RUN set -eu; \
    case "${TARGETARCH}" in \
      amd64) deb="${NFIQ2_DEB_AMD64}"; sha="${NFIQ2_SHA256_AMD64}" ;; \
      arm64) deb="${NFIQ2_DEB_ARM64}"; sha="${NFIQ2_SHA256_ARM64}" ;; \
      *) echo "unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL --retry 3 -o "/tmp/${deb}" "${NFIQ2_BASE_URL}/${deb}"; \
    echo "${sha}  /tmp/${deb}" | sha256sum -c -; \
    if [ "${TARGETARCH}" = "amd64" ]; then \
      apt-get update && apt-get install -y --no-install-recommends "/tmp/${deb}" && rm -rf /var/lib/apt/lists/*; \
    else \
      dpkg-deb -x "/tmp/${deb}" /; \
    fi; \
    rm -f "/tmp/${deb}"; \
    if ldd /usr/local/nfiq2/bin/nfiq2 | grep -q "not found"; then ldd /usr/local/nfiq2/bin/nfiq2; exit 1; fi; \
    nfiq2 -h 2>&1 | grep -q "NFIQ 2: ${NFIQ2_VERSION}"

COPY --from=nbis-builder /opt/nbis/bin/bozorth3 /usr/local/bin/bozorth3
# Bozorth3 has no --version flag; `fpv doctor` reads the built NBIS release from this file.
COPY --from=nbis-builder /opt/nbis/VERSION /usr/local/share/nbis/VERSION

# Python venv with hash-pinned dependencies.
RUN python3.12 -m venv /opt/venv
COPY requirements-pipeline.lock /tmp/requirements-pipeline.lock
RUN pip install --require-hashes --no-deps --only-binary=:all: -r /tmp/requirements-pipeline.lock \
 && pip check \
 && rm /tmp/requirements-pipeline.lock \
 && python -c "import cv2; assert hasattr(cv2, 'ximgproc'), 'cv2.ximgproc missing'"

# Non-root user (replaces the stock `ubuntu` user so the UID can be matched to the host).
RUN userdel -r ubuntu 2>/dev/null || true \
 && groupadd -g "${GID}" fpv \
 && useradd -m -u "${UID}" -g "${GID}" -s /bin/bash fpv

WORKDIR /app
COPY --chown=fpv:fpv pyproject.toml README.md ./
COPY --chown=fpv:fpv src ./src
COPY --chown=fpv:fpv configs ./configs
COPY --chown=fpv:fpv tests ./tests
COPY --chown=fpv:fpv scripts ./scripts
RUN pip install --no-deps --no-build-isolation --no-index -e . \
 && mkdir -p data/raw data/processed data/quality_reports data/templates data/db data/probes \
 && chown -R fpv:fpv /app

USER fpv
CMD ["fpv", "doctor"]
