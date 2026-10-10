#!/usr/bin/env python3
"""Generate GaleMind's test models from synthetic building-energy data.

Two models are written into the model repository, both with a dynamic batch
dimension so the server can merge independent requests into one ONNX call:

  building_energy       RandomForestRegressor  -> predicted daily kWh
  building_energy_class RandomForestClassifier -> efficiency band + probabilities

The dataset is synthetic: a plausible energy model of a building with additive
noise. Nothing here is downloaded, so the artifacts are reproducible from the
fixed seed alone.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

SEED = 20251010
OPSET = 17

FEATURES = [
    "floor_area_m2",
    "occupancy",
    "outdoor_temp_c",
    "hour_of_day",
    "hvac_setpoint_c",
    "insulation_rating",
    "is_weekend",
    "solar_irradiance_w_m2",
]

BANDS = ["efficient", "typical", "wasteful"]


def synthesize(n_samples: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Builds a synthetic building-energy dataset.

    Returns the feature matrix and the daily consumption in kWh.
    """
    rng = np.random.default_rng(seed)

    floor_area = rng.uniform(40.0, 2_000.0, n_samples)
    occupancy = np.clip(rng.poisson(floor_area / 60.0), 0, None).astype(float)
    outdoor_temp = rng.normal(12.0, 11.0, n_samples)
    hour = rng.integers(0, 24, n_samples).astype(float)
    setpoint = rng.normal(21.5, 1.5, n_samples)
    insulation = rng.integers(1, 6, n_samples).astype(float)
    is_weekend = (rng.random(n_samples) < 2 / 7).astype(float)
    solar = np.clip(rng.normal(220.0, 160.0, n_samples), 0.0, None)

    # Heating and cooling demand grow with the gap to the setpoint and shrink
    # with better insulation; occupancy and working hours add a plug load.
    temp_gap = np.abs(setpoint - outdoor_temp)
    envelope = floor_area / (1.0 + 0.45 * (insulation - 1.0))
    hvac = 0.055 * envelope * temp_gap
    occupied = ((hour >= 8) & (hour <= 19) & (is_weekend == 0)).astype(float)
    plug = 1.9 * occupancy * (0.35 + 0.65 * occupied)
    lighting = 0.012 * floor_area * np.clip(1.0 - solar / 900.0, 0.0, 1.0)
    baseline = 0.9 + 0.004 * floor_area

    kwh = hvac + plug + lighting + baseline
    kwh *= 1.0 + rng.normal(0.0, 0.05, n_samples)
    kwh = np.clip(kwh, 0.5, None)

    features = np.column_stack(
        [floor_area, occupancy, outdoor_temp, hour, setpoint, insulation, is_weekend, solar]
    )
    return features.astype(np.float32), kwh.astype(np.float32)


def efficiency_bands(features: np.ndarray, kwh: np.ndarray) -> np.ndarray:
    """Labels each sample by energy intensity per square metre."""
    intensity = kwh / features[:, FEATURES.index("floor_area_m2")]
    low, high = np.quantile(intensity, [1 / 3, 2 / 3])
    return np.digitize(intensity, [low, high]).astype(np.int64)


def write_model(
    repository: Path,
    name: str,
    version: str,
    onnx_model: onnx.ModelProto,
    config: dict,
) -> Path:
    model_dir = repository / name / version
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "model.onnx"
    path.write_bytes(onnx_model.SerializeToString())
    (repository / name / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return path


def describe(path: Path) -> str:
    model = onnx.load(path)

    def shape(value):
        return [d.dim_value if d.dim_value else -1 for d in value.type.tensor_type.shape.dim]

    inputs = ", ".join(f"{v.name}{shape(v)}" for v in model.graph.input)
    outputs = ", ".join(f"{v.name}{shape(v)}" for v in model.graph.output)
    return f"{path} ({path.stat().st_size:,} bytes)\n    in : {inputs}\n    out: {outputs}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models",
        help="model repository to write into (default: ./models)",
    )
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--trees", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument(
        "--clean", action="store_true", help="remove the two generated models first"
    )
    args = parser.parse_args()

    features, kwh = synthesize(args.samples)
    bands = efficiency_bands(features, kwh)

    if args.clean:
        for name in ("building_energy", "building_energy_class"):
            shutil.rmtree(args.repository / name, ignore_errors=True)

    initial_types = [("features", FloatTensorType([None, len(FEATURES)]))]
    common_config = {
        "platform": "onnx_onnxv1",
        "max_batch_size": args.max_batch_size,
        # No intra_op_threads: the server's --intra-op-threads governs. A 0
        # here would mean "let ONNX Runtime choose", which is one thread per
        # core for every loaded model.
        "features": FEATURES,
    }

    x_train, x_test, y_train, y_test = train_test_split(
        features, kwh, test_size=0.2, random_state=SEED
    )
    regressor = RandomForestRegressor(
        n_estimators=args.trees, max_depth=args.max_depth, random_state=SEED, n_jobs=-1
    ).fit(x_train, y_train)
    print(f"building_energy       r2={r2_score(y_test, regressor.predict(x_test)):.4f}")
    # Name the output explicitly; skl2onnx would otherwise call it "variable".
    regressor_onnx = to_onnx(
        regressor,
        initial_types=initial_types,
        final_types=[("kwh", FloatTensorType([None, 1]))],
        target_opset=OPSET,
    )
    path = write_model(
        args.repository,
        "building_energy",
        "1",
        regressor_onnx,
        {
            **common_config,
            "description": "Predicts daily building energy use in kWh from synthetic data.",
            "target": "kwh",
        },
    )
    print(describe(path))

    x_train, x_test, y_train, y_test = train_test_split(
        features, bands, test_size=0.2, random_state=SEED, stratify=bands
    )
    classifier = RandomForestClassifier(
        n_estimators=args.trees, max_depth=args.max_depth, random_state=SEED, n_jobs=-1
    ).fit(x_train, y_train)
    print(
        f"building_energy_class accuracy={accuracy_score(y_test, classifier.predict(x_test)):.4f}"
    )
    # zipmap=False keeps the probability output a plain tensor instead of a
    # sequence of maps, which the Predict Protocol v2 has no way to express.
    classifier_onnx = to_onnx(
        classifier,
        initial_types=initial_types,
        final_types=[
            ("band", Int64TensorType([None])),
            ("probabilities", FloatTensorType([None, len(BANDS)])),
        ],
        options={id(classifier): {"zipmap": False}},
        target_opset=OPSET,
    )
    path = write_model(
        args.repository,
        "building_energy_class",
        "1",
        classifier_onnx,
        {
            **common_config,
            "description": "Classifies a building's energy intensity band.",
            "classes": BANDS,
        },
    )
    print(describe(path))


if __name__ == "__main__":
    main()
