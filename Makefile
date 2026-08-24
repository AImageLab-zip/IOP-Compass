SHELL := /bin/bash
CONFIG ?= configs/dataset_final_1000.yaml
RUN    ?= final_1000
SPLIT  ?= test

# IOPC_ENV points at a site script sourced before anything else (module loads, CUDA
# paths).  Unset on a normal machine, where the venv alone is enough.
IOPC_ENV ?=
ENVSH    := $(if $(IOPC_ENV),source $(IOPC_ENV) &&,)
PY       := $(ENVSH) source .venv/bin/activate && \
            PYTHONPATH=$(CURDIR)/src:$(CURDIR)/third_party \
            TORCH_HOME=$(CURDIR)/third_party/torch_home python

.PHONY: help env third-party test lint app audit splits prepare submit monitor \
        freeze final-test aggregate

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s$$'\t'

env: ## build the uv venv with pinned wheels
	bash internal/scripts/setup_env.sh

third-party: ## link SegmentAnyTooth / SAM 3 assets and record their hashes
	$(PY) scripts/fetch_third_party.py --record runs/$(RUN)/third_party.json --hash-weights

test: ## run the unit tests
	$(PY) -m pytest tests -q

lint: ## ruff over the package, the scripts and the viewer backend
	$(PY) -m ruff check src scripts internal/scripts app/backend

app: ## run the viewer (API on :5000, UI on :5173)
	$(MAKE) -C app dev

prepare: ## rasters, image caches and decode verification (SLURM)
	$(PY) internal/scripts/submit_pipeline.py --config $(CONFIG) --run-name $(RUN) --stages prepare

audit: ## splits, manifest and the dataset audit (SLURM, gated on prepare)
	$(PY) internal/scripts/submit_pipeline.py --config $(CONFIG) --run-name $(RUN) --stages audit

splits: ## regenerate the frozen splits locally
	$(PY) scripts/create_splits.py --config $(CONFIG)

submit: ## submit the whole campaign except the test set
	$(PY) internal/scripts/submit_pipeline.py --config $(CONFIG) --run-name $(RUN) \
		--stages classify,roi,roi_eval,segment,maskrcnn,maskrcnn_infer,evaluate --split $(SPLIT)

monitor: ## poll SLURM, retry infrastructure faults, refresh runs/$(RUN)/STATUS_slurm.md
	$(PY) internal/scripts/monitor_slurm.py --run $(RUN) --watch --interval 300

freeze: ## hash the configuration before the test set is touched
	$(PY) scripts/freeze_experiment.py --config $(CONFIG) --run $(RUN)

final-test: ## evaluate the held-out test split once, under the lock
	$(PY) internal/scripts/run_final_test.py --config $(CONFIG) --run $(RUN) --final-test \
		--config-hash $$(cat runs/$(RUN)/frozen_config.sha256)

aggregate: ## collect every result file into one document
	$(PY) scripts/aggregate_results.py --run $(RUN) --config $(CONFIG)

