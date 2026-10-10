# Generate the Predict Protocol v2 gRPC stubs for the Python client.

PROTO_DIR ?= ../../grpc_server/proto
OUT_DIR   ?= ./galemind
PROTO     ?= grpc_predict_v2.proto

.PHONY: all generate test clean

all: generate

## Generate the gRPC stubs from the service definition
generate:
	python3 -m grpc_tools.protoc \
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
	python333 -m pytest -q

## Remove the generated stubs
clean:
	rm -f $(OUT_DIR)/grpc_predict_v2_pb2.py $(OUT_DIR)/grpc_predict_v2_pb2_grpc.py
	rm -rf $(OUT_DIR)/__pycache__ .pytest_cache
