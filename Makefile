# Visual-inertial odometry from scratch.
#
# Three classes of target, and the distinction matters when reading the README:
#
#   FAST      seconds to a couple of minutes, no dataset needed.  Everything CI
#             runs, plus the synthetic demo.
#               make test  make lint  make demo  make check
#
#   DATASET   needs EuRoC on disk.  `make data` downloads 19 GB and extracts
#             about 19 GB more; `make experiments` is roughly 40 minutes on
#             eight cores and produces every number the README quotes.
#               make data  make run  make experiments  make loop  make calibrate
#
#   REPORT    turns finished runs into tables and figures.
#               make plots  make tables  make docs-check
#
# `make check` runs the FAST class and is exactly what CI does.

SHELL := /bin/sh
PYTHON ?= python
EUROC_ROOT ?= $(HOME)/datasets/euroc/sequences
RESULTS ?= results
SEQUENCE ?= V1_01_easy
MODE ?= vio
JOBS ?= 6

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Fast
# ---------------------------------------------------------------------------

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: check
check: lint test demo  ## Everything CI runs, no dataset needed

.PHONY: test
test:  ## Unit tests (no dataset, no display)
	$(PYTHON) -m pytest tests/ -q

.PHONY: test-fast
test-fast:  ## Unit tests without the slow statistical ones
	$(PYTHON) -m pytest tests/ -q -m "not slow"

.PHONY: lint
lint:  ## Formatting and obvious errors
	$(PYTHON) -m flake8 src/ tests/ scripts/ evaluation/

.PHONY: demo
demo:  ## Run both configurations on a generated sequence (~90 s, no dataset)
	$(PYTHON) scripts/demo.py

.PHONY: sim-dataset
sim-dataset:  ## Write a synthetic sequence to disk in EuRoC's own layout
	$(PYTHON) scripts/make_sim_dataset.py --out data/sim_sequence

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

.PHONY: data
data:  ## Download and extract EuRoC (about 19 GB down, 19 GB on disk)
	$(PYTHON) scripts/fetch_euroc.py --out $(dir $(EUROC_ROOT))

.PHONY: run
run:  ## One sequence: make run SEQUENCE=V1_01_easy MODE=vio
	$(PYTHON) scripts/run_sequence.py $(SEQUENCE) --mode $(MODE) \
		--data $(EUROC_ROOT) --out $(RESULTS)

.PHONY: experiments
experiments:  ## Every sequence in both configurations (~40 min on 8 cores)
	$(PYTHON) scripts/run_all.py --jobs $(JOBS) --data $(EUROC_ROOT) --out $(RESULTS)

.PHONY: vocabulary
vocabulary:  ## Train the bag-of-words vocabulary on MH_01 and MH_02 only
	$(PYTHON) scripts/build_vocabulary.py --data $(EUROC_ROOT)

.PHONY: loop
loop: vocabulary  ## Loop closure: detection scores plus ATE before and after
	$(PYTHON) scripts/run_loop_closure.py $(SEQUENCE) --data $(EUROC_ROOT) \
		--out $(RESULTS)

.PHONY: calibrate
calibrate:  ## Estimate the camera-IMU extrinsic and time offset from a sequence
	$(PYTHON) scripts/calibrate.py $(SEQUENCE) --data $(EUROC_ROOT) --out $(RESULTS)

.PHONY: ablate
ablate:  ## The ablations behind the claims in docs/results.md
	$(PYTHON) scripts/ablate.py --data $(EUROC_ROOT) --out $(RESULTS)

.PHONY: profile
profile:  ## Per-stage runtime breakdown on one sequence
	$(PYTHON) scripts/profile_run.py $(SEQUENCE) --data $(EUROC_ROOT) --out $(RESULTS)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

.PHONY: evaluate
evaluate:  ## Recompute every metric from the trajectory files on disk
	$(PYTHON) evaluation/evaluate.py --results $(RESULTS)

.PHONY: evo-check
evo-check:  ## Cross-check our metrics against evo on the same files
	$(PYTHON) evaluation/evo_check.py --results $(RESULTS)

.PHONY: plots
plots:  ## Regenerate every figure in docs/plots
	$(PYTHON) evaluation/plots.py --results $(RESULTS) --out docs/plots

.PHONY: tables
tables:  ## Rewrite the README and docs/results.md tables from $(RESULTS)
	$(PYTHON) scripts/make_tables.py --results $(RESULTS)

.PHONY: docs-check
docs-check:  ## Do the documents still agree with the results they quote?
	$(PYTHON) scripts/check_docs.py --results $(RESULTS)

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

.PHONY: clean
clean:  ## Remove generated results, figures and caches
	rm -rf $(RESULTS)/*.tum $(RESULTS)/*.json docs/plots/*.png models/*.pkl
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache

.PHONY: docker-build
docker-build:  ## Build the container
	docker build -f docker/Dockerfile -t cv-visual-inertial-odometry .

.PHONY: docker-demo
docker-demo:  ## Run the synthetic demo in the container, no dataset needed
	docker compose -f docker/docker-compose.yml run --rm vio
