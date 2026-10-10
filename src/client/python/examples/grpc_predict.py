#!/usr/bin/env python3
"""Classifying a building's energy band over gRPC.

Start a server and generate the stubs first:

    cargo run -p galemind -- serve
    make generate
    python examples/grpc_predict.py

gRPC sends tensor contents as raw bytes rather than as JSON numbers, which
avoids per-element encoding and is the only form that carries 16-bit floats
exactly. The call shapes are otherwise the same as the HTTP client's.

The endpoint can be overridden with GALEMIND_GRPC_ENDPOINT.
"""

from __future__ import annotations

import os
import sys

from galemind import GrpcClient, InferenceError, Tensor

BUILDINGS = [
    [1800, 30, -4, 18, 22, 1, 0, 40],
    [400, 12, 8, 11, 21.5, 3, 0, 250],
    [55, 2, 18, 14, 21, 5, 1, 600],
]

# The bands the classifier was trained on, in label order.
BANDS = ["efficient", "typical", "wasteful"]


def main() -> int:
    endpoint = os.environ.get("GALEMIND_GRPC_ENDPOINT", "127.0.0.1:8001")

    try:
        with GrpcClient(endpoint) as client:
            if not client.server_ready():
                print(f"{endpoint} is not ready to serve models", file=sys.stderr)
                return 1
            print(f"connected to {endpoint} over gRPC")

            metadata = client.model_metadata("building_energy_class")
            outputs = ", ".join(f"{t.name}{t.shape}" for t in metadata.outputs)
            print(f"{metadata.name} returns {outputs}")

            features = Tensor(
                name="features",
                shape=[len(BUILDINGS), 8],
                datatype="FP32",
                data=BUILDINGS,          # nested rows are flattened for you
            )
            response = client.infer(
                "building_energy_class",
                [features],
                request_id="python-grpc",
            )

            bands = response.output("band").data
            probabilities = response.output("probabilities")
            for index, (building, band) in enumerate(zip(BUILDINGS, bands)):
                confidence = max(probabilities.row(index))
                name = BANDS[band] if 0 <= band < len(BANDS) else "unknown"
                print(f"  {building[0]:>6} m^2 -> {name} ({confidence:.0%} confident)")
    except InferenceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
