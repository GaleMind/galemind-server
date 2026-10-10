# Targets that are commands, not files.
.PHONY: all test format run rocksdb-lib models models-deps \
	coverage coverage-open coverage-tools \
	docker-build docker-run docker-push

all: format test 
	cargo build

test:
	cargo test

# Line coverage, written to $(COVERAGE_DIR) as both a browsable HTML tree and
# an lcov trace for CI to keep.
#
# cargo-llvm-cov instruments the source through LLVM instead of single-stepping
# the test binary under ptrace. That distinction earns its keep here: the store
# links a static RocksDB and the ONNX runtime spawns threads of its own, and a
# ptrace-based counter attributes both badly.
#
# The threshold is a variable rather than a literal so a branch that is
# deliberately mid-refactor can run `make coverage COVERAGE_MIN=0` instead of
# editing this file.
COVERAGE_MIN ?= 75
COVERAGE_DIR ?= reports/coverage

# What is not under test: the generated protobuf stubs, which the build scripts
# write under target/, and the integration harnesses under tests/, which are
# the test, not the subject. Neither belongs in the denominator. The end-to-end
# suite needs no exclusion -- it is its own workspace, so --workspace never
# reaches it.
COVERAGE_IGNORE ?= '(^|/)target/|(^|/)tests/|(^|/)build\.rs$$'

# Three passes over one set of profiles: measure once with --no-report, then
# render each format from the result. A single `cargo llvm-cov` invocation
# emits one format only, and re-running it per format would re-run the tests.
coverage: coverage-tools
	cargo llvm-cov clean --workspace
	cargo llvm-cov --workspace --no-report
	@mkdir -p $(COVERAGE_DIR)
	cargo llvm-cov report --ignore-filename-regex $(COVERAGE_IGNORE) \
		--html --output-dir $(COVERAGE_DIR)
	cargo llvm-cov report --ignore-filename-regex $(COVERAGE_IGNORE) \
		--lcov --output-path $(COVERAGE_DIR)/lcov.info
	@echo
	cargo llvm-cov report --ignore-filename-regex $(COVERAGE_IGNORE) \
		--fail-under-lines $(COVERAGE_MIN)
	@echo "HTML report: $(COVERAGE_DIR)/html/index.html"

# The same measurement, opened in a browser and without the gate, for reading
# which lines are missed rather than checking a number.
coverage-open: coverage-tools
	cargo llvm-cov clean --workspace
	cargo llvm-cov --workspace --ignore-filename-regex $(COVERAGE_IGNORE) \
		--open --output-dir $(COVERAGE_DIR)

# cargo-llvm-cov reads the raw profiles with the toolchain's own llvm-tools, so
# the component is as much a dependency as the subcommand itself.
coverage-tools:
	@command -v cargo-llvm-cov >/dev/null 2>&1 || cargo install cargo-llvm-cov --locked
	@rustup component list --installed 2>/dev/null | grep -q '^llvm-tools' \
		|| rustup component add llvm-tools-preview

format:
	cargo fmt

run:
	@if [ -f .env ]; then \
		set -a && . ./.env && set +a; \
	fi && \
	cargo run -p galemind serve

rocksdb-lib:
	./scripts/stage-rocksdb-lib.sh

docker-build:
	$(eval TIMESTAMP := $(shell date +%Y%m%d%H%M))
	docker build -t galemind-server:$(TIMESTAMP) .

# 8000 is REST (and /metrics), 8001 is gRPC.
docker-run:
	$(eval TIMESTAMP := $(shell date +%Y%m%d%H%M))
	docker run --rm -p 8000:8000 -p 8001:8001 galemind-server:$(TIMESTAMP)

docker-push:
	$(eval TIMESTAMP := $(shell date +%Y%m%d%H%M))
	docker push galemind-server:$(TIMESTAMP)

# Regenerates the model repository. The models are committed, so this is only
# needed to change or extend them; see scripts/requirements.txt for what it
# needs installed.
models:
	@if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi; \
	$$PY -I scripts/export_models.py && $$PY -I scripts/synthetic_models.py

models-deps:
	@if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi; \
	$$PY -m pip install -r scripts/requirements.txt
