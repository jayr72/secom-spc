PY := .venv/bin/python

.PHONY: all fast test setup

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

all: test
	$(PY) run_all.py

fast:
	$(PY) run_all.py --fast

test:
	$(PY) -m pytest -q tests
