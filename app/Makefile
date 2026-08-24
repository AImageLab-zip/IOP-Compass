SHELL := /bin/bash
REPO  := $(CURDIR)/..

HOST ?= 127.0.0.1
PORT ?= 5000
SEGMENTERS ?= sat,mask-rcnn

PY := source $(REPO)/.venv/bin/activate && \
      PYTHONPATH=$(REPO)/src:$(REPO)/third_party \
      TORCH_HOME=$(REPO)/third_party/torch_home python

.PHONY: help check backend frontend dev install smoke

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s$$'\t'

install: ## install the frontend dependencies
	cd frontend && npm install

check: ## verify every weight the enabled segmenters need
	$(PY) backend/server.py --check --segmenters $(SEGMENTERS)

backend: ## run the API
	$(PY) backend/server.py --host $(HOST) --port $(PORT) --segmenters $(SEGMENTERS)

frontend: ## run the UI (proxies /api to the backend)
	cd frontend && npm run dev

smoke: ## drive the real API with one held-out patient
	$(PY) backend/smoke_test.py --config $(REPO)/configs/dataset_final_1000.yaml

dev: ## run both; the UI is on http://localhost:5173
	@$(MAKE) backend & \
	 $(MAKE) frontend; \
	 kill %1 2>/dev/null || true
