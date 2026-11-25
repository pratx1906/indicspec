# IndicSpec Makefile.  Run inside WSL/Linux (GNU make + POSIX shell).
# GPU targets are marked; they need the engine envs described in env/ENGINES.md.

SHELL := /bin/bash
# Keep the CPU venv on the Linux filesystem when the repo sits on /mnt/c (much faster).
export UV_PROJECT_ENVIRONMENT ?= $(if $(findstring /mnt/,$(CURDIR)),$(HOME)/.cache/indicspec/venv-cpu,.venv)
UV ?= uv
PY := $(UV) run python

.PHONY: help sync env-report test test-data test-all lint typecheck data figures smoke sweep analyze reproduce clean-pyc

help:
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  %-14s %s\n",$$1,$$2}'

sync: ## create/update the CPU analysis environment
	$(UV) sync --extra data

env-report: sync ## probe hardware/software -> env/probe.json
	$(PY) -m indicspec.env_probe --out env/probe.json

test: sync ## CPU-only fast tests (unit, property, schema)
	$(UV) run pytest -m "not gpu and not data and not slow"

test-data: sync ## tests needing downloaded tokenizers/datasets (network or HF cache)
	$(UV) run pytest -m "data and not gpu"

test-all: sync ## everything except GPU
	$(UV) run pytest -m "not gpu"

lint: sync ## ruff lint + format check + mypy
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests
	$(UV) run mypy

data: sync ## build the frozen IndicSpec-Bench and verify checksums (WS2)
	$(PY) -m indicspec.data.build --verify

figures: sync ## regenerate docs/assets/*.svg from results/tables
	$(UV) sync --extra analysis --extra data
	$(PY) -m indicspec.analysis.figures

smoke: ## [GPU] short generation per working arm (WS1)
	$(PY) -m indicspec.harness.smoke

sweep: ## [GPU] run a sweep config: make sweep CONFIG=configs/<name>.yaml
	$(PY) -m indicspec.harness.sweep --config $(CONFIG)

analyze: sync ## regenerate tables/figures from results/raw
	$(PY) -m indicspec.analysis.report

reproduce: data analyze ## rebuild every released table/figure from raw logs

clean-pyc:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
