# Dependency verification test image
#
# Builds on the main VHS restore image and runs verify_deps.py
# against the actual installed packages, paths, and plugins.
#
# Usage (Linux — build main image locally, then test):
#   docker build -t vhs-restore .
#   docker build -f tests/Dockerfile --build-arg BASE_IMAGE=vhs-restore -t vhs-test .
#
# Usage (macOS — pull amd64 image from registry, then test):
#   docker pull --platform linux/amd64 ghcr.io/viktorstiskala/vhs-upscale:cu128
#   docker buildx build --platform linux/amd64 -f tests/Dockerfile -t vhs-test --load .

ARG BASE_IMAGE=ghcr.io/viktorstiskala/vhs-upscale:cu128
FROM ${BASE_IMAGE}

# Copy source files for path consistency checks
COPY upscale.vpy upscale.sh Dockerfile /tmp/src/

# Copy and run the verification script
COPY tests/verify_deps.py /tmp/verify_deps.py
RUN python3 /tmp/verify_deps.py /tmp/src
