#!/usr/bin/env python3
"""Predicting building energy use over HTTP/REST.

Start a server first:

    cargo run -p galemind -- serve
    python examples/rest_predict.py

The endpoint can be overridden with GALEMIND_ENDPOINT.
"""

from __future__ import annotations

import os
import sys

from galemind import HttpClient, InferenceError

# Two buildings: a large, poorly insulated office on a cold evening, and a
# small, well insulated flat on a mild sunny afternoon.
FEATURE_NAMES = [
    "floor_area_m2",
    "occupancy",
    "outdoor_temp_c",
    "hour_of_day",
    "hvac_setpoint_c",
    "insulation_rating",
    "is_weekend",
    "solar_irradiance_w_m2",
]
BUILDINGS = [
    [1800, 30, -4, 18, 22, 1, 0, 40],
    [55, 2, 18, 14, 21, 5, 1, 600],
]


def main() -> int:
    endpoint = os.environ.get("GALEMIND_ENDPOINT", "http://127.0.0.1:8000")
    client = HttpClient(endpoint)

    try:
        if not client.server_live():
            print(f"{endpoint} is not live", file=sys.stderr)
            return 1

        server = client.server_metadata()
        print(f"connected to {server.name} {server.version} over HTTP")

        # Reading the metadata rather than hardcoding shapes means this keeps
        # working when the model is retrained with different inputs.
        metadata = client.model_metadata("building_energy")
        features = metadata.sole_input
        print(
            f"{metadata.name} ({metadata.platform}) takes "
            f"{features.name}{features.shape} of {features.datatype}"
        )

        # Both buildings go in one request: one round trip, one execution.
        response = client.predict("building_energy", BUILDINGS, request_id="python-rest")
        print(f"served by {response.model_name}/{response.model_version}")

        for building, kwh in zip(BUILDINGS, response.output("kwh").data):
            described = dict(zip(FEATURE_NAMES, building))
            print(
                f"  {described['floor_area_m2']:>6} m^2, "
                f"{described['occupancy']:>3} people, "
                f"{described['outdoor_temp_c']:>5} C outside -> {kwh:>8.1f} kWh/day"
            )
    except InferenceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
