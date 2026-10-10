# Targets that are commands, not files.
.PHONY: all test format run rocksdb-lib models models-deps \
	docker-build docker-run docker-push

all: format test 
	cargo build

test:
	cargo test

#coverage:
#	cargo tarpaulin --config tarpaulin.toml --fail-under 80

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
