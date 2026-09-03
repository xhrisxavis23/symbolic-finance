PY := python3

.PHONY: test test-fast slice vendor-diff

test:
	$(PY) -m pytest tests/ -v

test-fast:
	$(PY) -m pytest tests/ -v -m "not slow"

slice:
	$(PY) run_slice.py

vendor-diff:
	$(PY) tools/vendor_sync.py --check
