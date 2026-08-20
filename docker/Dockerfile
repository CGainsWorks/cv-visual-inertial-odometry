# A container that runs the whole pipeline with no dataset and no GPU.
#
# `docker compose -f docker/docker-compose.yml run --rm vio` generates a
# synthetic visual-inertial sequence, runs both configurations over it and
# prints the comparison.  That is deliberately the default: it is the one
# command that works on a clean machine with nothing downloaded.
#
# To run on EuRoC instead, mount the dataset and point EUROC_ROOT at it; the
# compose file has the line commented out.

FROM python:3.12-slim

# opencv-python-headless still wants libGL's ABI present even though nothing
# here opens a window, and libglib is needed by its image codecs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY evaluation/ ./evaluation/
COPY Makefile .flake8 ./

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
# One thread per process: the estimator is a chain of small matrix operations
# and gains nothing from BLAS threading, while oversubscription costs plenty.
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

CMD ["python", "scripts/demo.py"]
