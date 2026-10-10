# Generate the Predict Protocol v2 gRPC stubs for the Python client.

PROTO_DIR ?= ../../grpc_server/proto
OUT_DIR   ?= ./galemind
PROTO     ?= grpc_predict_v2.proto

# Prefer this package's virtualenv when one has been created, as the root
# Makefile does for the model exporters. The stub generator and the coverage
# plugin are the `dev` extra, so they are usually only in the venv.
PY ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

.PHONY: all generate test coverage clean

all: generate

## Generate the gRPC stubs from the service definition
generate:
	$(PY) -m grpc_tools.protoc \
		--proto_path=$(PROTO_DIR) \
		--python_out=$(OUT_DIR) \
		--grpc_python_out=$(OUT_DIR) \
		$(PROTO_DIR)/$(PROTO)
	@# The generated gRPC module imports its sibling as a top-level module,
	@# which breaks inside a package; make it a relative import.
	sed -i'' -e 's/^import grpc_predict_v2_pb2 as/from . import grpc_predict_v2_pb2 as/' \
		$(OUT_DIR)/grpc_predict_v2_pb2_grpc.py

## Run the client tests
test:
	$(PY) -m pytest -q

## Report line coverage for the client package
#
# The gRPC half of the client is only importable once `generate` has written
# the stubs, so coverage depends on them; without them those modules would be
# collected as import errors and scored as nothing.
coverage: generate
	$(PY) -m pytest -q \
		--cov=galemind --cov-report=term-missing \
		--cov-report=html:../../../reports/coverage-python \
		--cov-report=lcov:../../../reports/coverage-python/lcov.info

## Remove the generated stubs
clean:
	rm -f $(OUT_DIR)/grpc_predict_v2_pb2.py $(OUT_DIR)/grpc_predict_v2_pb2_grpc.py
	rm -rf $(OUT_DIR)/__pycache__ .pytest_cache .coverage
