#!/usr/bin/env python3
"""Build GaleMind's synthetic ONNX model zoo: ten models, no downloads.

The repository needs more than two tree ensembles to be worth measuring. A
server is judged on how it behaves with a dozen models of different shapes
loaded at once -- a tiny preprocessor answering in microseconds beside a
generator that holds a core for tens of milliseconds -- and that mix is what
decides whether batching, the model cache and the thread plan are right.

Nothing here is trained, and nothing here is downloaded. Every weight comes
from a seeded generator at the scale the layer's initialisation would give it,
so the graphs are the real shapes and the real operators with arbitrary
numbers in them. That is what a serving test needs: the arithmetic is the
same, the tensor shapes are the same, the cost is the same. Predictions are
meaningless, and the config of each model says so.

StackGAN is here because it is the awkward case. Its two stages take an
embedding and a noise vector and return a 64x64 and then a 256x256 image, so
a response is megabytes rather than a row of floats, one inference is tens of
milliseconds rather than tens of microseconds, and the arena allocator wants
to keep every intermediate buffer it has ever used. Serving it next to a
scaler that answers in 20us is the test.

Written with onnx.helper rather than exported from a framework: these are
ten graphs built from fewer than a dozen operator kinds, and building them
directly keeps the whole zoo reproducible from this file and numpy alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SEED = 20251010
OPSET = 17

# Every model takes a dynamic first dimension under this name, which is what
# lets the server merge independent requests into one call.
BATCH = "batch"


def rng(salt: int) -> np.random.Generator:
    """A generator per model, so adding one does not change another's weights."""
    return np.random.default_rng(SEED + salt)


# -- graph building ---------------------------------------------------------


def value(name: str, shape: list, dtype=TensorProto.FLOAT):
    """A tensor whose first dimension is the batch."""
    return helper.make_tensor_value_info(name, dtype, [BATCH] + list(shape))


def const(name: str, array: np.ndarray):
    return numpy_helper.from_array(array.astype(np.float32), name)


def dense(rng_: np.random.Generator, name: str, fan_in: int, fan_out: int):
    """A Gemm's weight and bias, Xavier-scaled so activations stay in range."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return (
        const(f"{name}.weight", rng_.uniform(-limit, limit, (fan_in, fan_out))),
        const(f"{name}.bias", np.zeros(fan_out)),
    )


def kernel(rng_: np.random.Generator, name: str, out_ch: int, in_ch: int, size=3):
    """A Conv's kernel and bias, He-scaled for a ReLU that follows."""
    shape = (out_ch, in_ch) + ((size,) if isinstance(size, int) and size == 1 else (size, size))
    std = np.sqrt(2.0 / (in_ch * int(np.prod(shape[2:]))))
    return (
        const(f"{name}.weight", rng_.normal(0.0, std, shape)),
        const(f"{name}.bias", np.zeros(out_ch)),
    )


def kernel_1d(rng_: np.random.Generator, name: str, out_ch: int, in_ch: int, size: int):
    std = np.sqrt(2.0 / (in_ch * size))
    return (
        const(f"{name}.weight", rng_.normal(0.0, std, (out_ch, in_ch, size))),
        const(f"{name}.bias", np.zeros(out_ch)),
    )


def build(name: str, nodes, inputs, outputs, initializers, doc: str) -> onnx.ModelProto:
    """Assembles, checks and shape-infers a graph."""
    graph = helper.make_graph(nodes, name, inputs, outputs, initializers, doc_string=doc)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="galemind-synthetic-models",
        doc_string=doc,
    )
    model.ir_version = 9  # what ONNX Runtime 1.17+ reads
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


def upsample(nodes, initializers, source: str, out: str, factor: int = 2) -> str:
    """Nearest-neighbour resize of the spatial dimensions.

    The transposed convolution a generator would really use needs its own
    weights per stage; resize plus a convolution gives the same shapes and the
    same cost profile with less to go wrong, and is what most modern
    implementations do anyway to avoid checkerboard artefacts.
    """
    scales = f"{out}.scales"
    initializers.append(const(scales, np.array([1.0, 1.0, float(factor), float(factor)])))
    nodes.append(
        helper.make_node(
            "Resize",
            # The empty name is `roi`, which nearest-neighbour does not use.
            inputs=[source, "", scales],
            outputs=[out],
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        )
    )
    return out


def conv_block(
    nodes,
    initializers,
    rng_: np.random.Generator,
    source: str,
    name: str,
    out_ch: int,
    in_ch: int,
    activation: str | None = "Relu",
    strides=(1, 1),
) -> str:
    weight, bias = kernel(rng_, name, out_ch, in_ch)
    initializers.extend([weight, bias])
    nodes.append(
        helper.make_node(
            "Conv",
            [source, weight.name, bias.name],
            [f"{name}.out"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=list(strides),
        )
    )
    if activation is None:
        return f"{name}.out"
    nodes.append(helper.make_node(activation, [f"{name}.out"], [f"{name}.act"]))
    return f"{name}.act"


# -- the models -------------------------------------------------------------


def occupancy_mlp(hidden: tuple[int, int], salt: int) -> onnx.ModelProto:
    """A dense regressor: eight building features to a head count.

    The plainest shape in the zoo, and the one that shows the server's own
    overhead: a batch of these costs so little that whatever the response
    takes is almost entirely the server.
    """
    r = rng(salt)
    nodes, init = [], []
    source = "features"
    widths = [8, *hidden]
    for layer, (fan_in, fan_out) in enumerate(zip(widths, widths[1:])):
        weight, bias = dense(r, f"hidden{layer}", fan_in, fan_out)
        init.extend([weight, bias])
        nodes.append(
            helper.make_node("Gemm", [source, weight.name, bias.name], [f"hidden{layer}.out"])
        )
        nodes.append(helper.make_node("Relu", [f"hidden{layer}.out"], [f"hidden{layer}.act"]))
        source = f"hidden{layer}.act"

    weight, bias = dense(r, "output", hidden[-1], 1)
    init.extend([weight, bias])
    nodes.append(helper.make_node("Gemm", [source, weight.name, bias.name], ["occupancy"]))

    return build(
        "occupancy_mlp",
        nodes,
        [value("features", [8])],
        [value("occupancy", [1])],
        init,
        "Synthetic dense regressor: building features to expected occupancy.",
    )


def energy_autoencoder() -> onnx.ModelProto:
    """An autoencoder that reports its own reconstruction error.

    Two outputs of different widths from one input, which is the case that
    catches a server that assumes every output has the same shape as the
    input -- splitting a batched response has to split both correctly.
    """
    r = rng(2)
    nodes, init = [], []

    encode_w, encode_b = dense(r, "encode", 8, 4)
    decode_w, decode_b = dense(r, "decode", 4, 8)
    init.extend([encode_w, encode_b, decode_w, decode_b])
    nodes += [
        helper.make_node("Gemm", ["features", encode_w.name, encode_b.name], ["code.raw"]),
        helper.make_node("Relu", ["code.raw"], ["code"]),
        helper.make_node("Gemm", ["code", decode_w.name, decode_b.name], ["reconstruction"]),
        # The anomaly score is the squared error, summed across features.
        helper.make_node("Sub", ["features", "reconstruction"], ["residual"]),
        helper.make_node("Mul", ["residual", "residual"], ["squared"]),
        helper.make_node("ReduceSum", ["squared", "sum.axes"], ["anomaly_score"], keepdims=1),
    ]
    init.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "sum.axes"))

    return build(
        "energy_autoencoder",
        nodes,
        [value("features", [8])],
        [value("reconstruction", [8]), value("anomaly_score", [1])],
        init,
        "Synthetic autoencoder: reconstructs building features and scores the residual.",
    )


def thermal_cnn() -> onnx.ModelProto:
    """A small convolutional classifier over 32x32 single-channel images.

    A thermal camera frame of a facade, classified into four conditions. The
    first model in the zoo whose input is an image, so a request is kilobytes
    rather than a row of eight floats.
    """
    r = rng(3)
    nodes, init = [], []

    source = conv_block(nodes, init, r, "image", "conv0", out_ch=8, in_ch=1)
    nodes.append(
        helper.make_node("MaxPool", [source], ["pool0"], kernel_shape=[2, 2], strides=[2, 2])
    )
    source = conv_block(nodes, init, r, "pool0", "conv1", out_ch=16, in_ch=8)
    nodes.append(
        helper.make_node("MaxPool", [source], ["pool1"], kernel_shape=[2, 2], strides=[2, 2])
    )
    nodes.append(helper.make_node("Flatten", ["pool1"], ["flat"], axis=1))

    weight, bias = dense(r, "head", 16 * 8 * 8, 4)
    init.extend([weight, bias])
    nodes += [
        helper.make_node("Gemm", ["flat", weight.name, bias.name], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
    ]

    return build(
        "thermal_cnn",
        nodes,
        [value("image", [1, 32, 32])],
        [value("logits", [4]), value("probabilities", [4])],
        init,
        "Synthetic CNN: 32x32 thermal frame to four facade conditions.",
    )


def meter_digit_cnn() -> onnx.ModelProto:
    """A 28x28 digit classifier with a label output as well as scores.

    The classic shape, and the only model in the zoo returning an integer
    label beside its probabilities -- the datatype mix a response encoder has
    to get right.
    """
    r = rng(4)
    nodes, init = [], []

    source = conv_block(nodes, init, r, "image", "conv0", out_ch=16, in_ch=1)
    nodes.append(
        helper.make_node("MaxPool", [source], ["pool0"], kernel_shape=[2, 2], strides=[2, 2])
    )
    source = conv_block(nodes, init, r, "pool0", "conv1", out_ch=32, in_ch=16)
    nodes.append(
        helper.make_node("MaxPool", [source], ["pool1"], kernel_shape=[2, 2], strides=[2, 2])
    )
    nodes.append(helper.make_node("Flatten", ["pool1"], ["flat"], axis=1))

    weight, bias = dense(r, "head", 32 * 7 * 7, 10)
    init.extend([weight, bias])
    nodes += [
        helper.make_node("Gemm", ["flat", weight.name, bias.name], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
        helper.make_node("ArgMax", ["logits"], ["digit"], axis=1, keepdims=0),
    ]

    outputs = [
        helper.make_tensor_value_info("digit", TensorProto.INT64, [BATCH]),
        value("probabilities", [10]),
    ]
    return build(
        "meter_digit_cnn",
        nodes,
        [value("image", [1, 28, 28])],
        outputs,
        init,
        "Synthetic CNN: 28x28 meter photograph to a digit and its probabilities.",
    )


def stackgan_stage1(base_channels: int = 64) -> onnx.ModelProto:
    """StackGAN stage I: a text embedding and a noise vector to a 64x64 image.

    The conditioning is the real design -- the embedding is projected to a
    conditioning vector and concatenated with the noise before the first
    projection -- without the sampling, because a server must return the same
    image for the same request.
    """
    r = rng(5)
    nodes, init = [], []

    # Conditioning augmentation, reduced to its mean: a generator samples
    # around it, a server has to be deterministic.
    mu_w, mu_b = dense(r, "conditioning", 128, 64)
    init.extend([mu_w, mu_b])
    nodes += [
        helper.make_node("Gemm", ["text_embedding", mu_w.name, mu_b.name], ["conditioning.raw"]),
        helper.make_node("LeakyRelu", ["conditioning.raw"], ["conditioning"], alpha=0.2),
        helper.make_node("Concat", ["conditioning", "noise"], ["latent"], axis=1),
    ]

    project_w, project_b = dense(r, "project", 64 + 100, base_channels * 16)
    init.extend([project_w, project_b])
    init.append(
        numpy_helper.from_array(
            np.array([-1, base_channels, 4, 4], dtype=np.int64), "project.shape"
        )
    )
    nodes += [
        helper.make_node("Gemm", ["latent", project_w.name, project_b.name], ["project.out"]),
        helper.make_node("Relu", ["project.out"], ["project.act"]),
        helper.make_node("Reshape", ["project.act", "project.shape"], ["stage.4x4"]),
    ]

    # 4x4 -> 64x64, halving the channels at each step.
    source = "stage.4x4"
    channels = base_channels
    for step, resolution in enumerate([8, 16, 32]):
        source = upsample(nodes, init, source, f"up{step}")
        source = conv_block(
            nodes, init, r, source, f"up{step}.conv", out_ch=channels // 2, in_ch=channels
        )
        channels //= 2
        assert resolution  # the resolution each step reaches, for the reader

    source = upsample(nodes, init, source, "up3")
    source = conv_block(
        nodes, init, r, source, "to_image", out_ch=3, in_ch=channels, activation=None
    )
    nodes.append(helper.make_node("Tanh", [source], ["image_64"]))

    return build(
        "stackgan_stage1",
        nodes,
        [value("text_embedding", [128]), value("noise", [100])],
        [value("image_64", [3, 64, 64])],
        init,
        "Synthetic StackGAN stage I: text embedding and noise to a 64x64 image.",
    )


def stackgan_stage2() -> onnx.ModelProto:
    """StackGAN stage II: a 64x64 image and its embedding to 256x256.

    The heaviest model in the zoo by far: it reads an image, encodes it,
    concatenates the spatially replicated embedding, and upsamples four
    times. One inference is tens of milliseconds and a response is 768 KB of
    floats, which is what makes it the model to watch when tuning batch size
    and memory.
    """
    r = rng(6)
    nodes, init = [], []

    # Encode the stage I image down to 16x16.
    source = conv_block(
        nodes, init, r, "image_64", "down0", out_ch=16, in_ch=3, strides=(2, 2)
    )
    source = conv_block(nodes, init, r, source, "down1", out_ch=32, in_ch=16, strides=(2, 2))

    # The embedding is projected and replicated across the feature map, which
    # is how the stage II generator conditions on the text.
    embed_w, embed_b = dense(r, "embed", 128, 16)
    init.extend([embed_w, embed_b])
    init.append(numpy_helper.from_array(np.array([-1, 16, 1, 1], dtype=np.int64), "embed.shape"))
    nodes += [
        helper.make_node("Gemm", ["text_embedding", embed_w.name, embed_b.name], ["embed.out"]),
        helper.make_node("Relu", ["embed.out"], ["embed.act"]),
        helper.make_node("Reshape", ["embed.act", "embed.shape"], ["embed.map"]),
    ]
    replicated = upsample(nodes, init, "embed.map", "embed.tiled", factor=16)
    nodes.append(helper.make_node("Concat", [source, replicated], ["joint"], axis=1))

    # A residual block at 16x16, then 16 -> 256 in four steps.
    source = conv_block(nodes, init, r, "joint", "residual", out_ch=32, in_ch=48)
    channels = 32
    for step, out_ch in enumerate([16, 16, 8]):
        source = upsample(nodes, init, source, f"up{step}")
        source = conv_block(
            nodes, init, r, source, f"up{step}.conv", out_ch=out_ch, in_ch=channels
        )
        channels = out_ch

    source = upsample(nodes, init, source, "up3")
    source = conv_block(
        nodes, init, r, source, "to_image", out_ch=3, in_ch=channels, activation=None
    )
    nodes.append(helper.make_node("Tanh", [source], ["image_256"]))

    return build(
        "stackgan_stage2",
        nodes,
        [value("image_64", [3, 64, 64]), value("text_embedding", [128])],
        [value("image_256", [3, 256, 256])],
        init,
        "Synthetic StackGAN stage II: refines a 64x64 image to 256x256.",
    )


def load_forecast_tcn() -> onnx.ModelProto:
    """A dilated 1-D convolution stack: a week of hourly load to the next day.

    The sequence case. Dilations widen the receptive field without a
    recurrence, so the whole thing is three convolutions and a projection --
    and unlike an LSTM it batches cleanly, which is the point of choosing it.
    """
    r = rng(7)
    nodes, init = [], []

    source = "sequence"
    channels = 1
    for step, (out_ch, dilation) in enumerate([(16, 1), (16, 2), (8, 4)]):
        weight, bias = kernel_1d(r, f"tcn{step}", out_ch, channels, 5)
        init.extend([weight, bias])
        # Padded on both sides to keep the length, so the projection's input
        # width does not depend on the dilation.
        pad = 2 * dilation
        nodes += [
            helper.make_node(
                "Conv",
                [source, weight.name, bias.name],
                [f"tcn{step}.out"],
                kernel_shape=[5],
                dilations=[dilation],
                pads=[pad, pad],
            ),
            helper.make_node("Relu", [f"tcn{step}.out"], [f"tcn{step}.act"]),
        ]
        source = f"tcn{step}.act"
        channels = out_ch

    nodes.append(helper.make_node("Flatten", [source], ["flat"], axis=1))
    weight, bias = dense(r, "forecast", 8 * 168, 24)
    init.extend([weight, bias])
    nodes.append(helper.make_node("Gemm", ["flat", weight.name, bias.name], ["forecast"]))

    return build(
        "load_forecast_tcn",
        nodes,
        [value("sequence", [1, 168])],
        [value("forecast", [24])],
        init,
        "Synthetic dilated-convolution forecaster: 168 hourly readings to 24 hours ahead.",
    )


def feature_scaler() -> onnx.ModelProto:
    """Standardisation, as a model of its own.

    Three operators and no weights to speak of, so it answers in
    microseconds. It is in the zoo as the floor: whatever latency the server
    reports for this is the server's own, with no model to blame.
    """
    r = rng(8)
    mean = r.uniform(0.0, 100.0, 8)
    std = r.uniform(1.0, 20.0, 8)

    init = [const("mean", mean), const("std", std)]
    nodes = [
        helper.make_node("Sub", ["features", "mean"], ["centred"]),
        helper.make_node("Div", ["centred", "std"], ["scaled"]),
    ]
    return build(
        "feature_scaler",
        nodes,
        [value("features", [8])],
        [value("scaled", [8])],
        init,
        "Standardises the eight building features; the zoo's cheapest model.",
    )


# -- writing ----------------------------------------------------------------


def config(description: str, max_batch_size: int, **extra) -> dict:
    return {
        "platform": "onnx_onnxv1",
        "max_batch_size": max_batch_size,
        # intra_op_threads is deliberately absent: a model that sets it
        # overrides the server's --intra-op-threads for itself, and a 0 there
        # means "let ONNX Runtime choose", which is one thread per core for
        # every loaded model. Ten of those on a twelve-core machine is 120
        # threads. Only a model that really wants its own operator
        # parallelism should say so.
        "description": description,
        "synthetic": True,
        "note": "Weights are random; the shapes and the cost are real, the predictions are not.",
        **extra,
    }


def write(repository: Path, name: str, version: str, model: onnx.ModelProto, cfg: dict) -> Path:
    directory = repository / name / version
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.onnx"
    path.write_bytes(model.SerializeToString())
    (repository / name / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return path


def describe(path: Path) -> str:
    model = onnx.load(path)

    def shape(v):
        return [
            d.dim_param if d.dim_param else d.dim_value for d in v.type.tensor_type.shape.dim
        ]

    inputs = ", ".join(f"{v.name}{shape(v)}" for v in model.graph.input)
    outputs = ", ".join(f"{v.name}{shape(v)}" for v in model.graph.output)
    return (
        f"{path.parent.parent.name}/{path.parent.name}"
        f"  {path.stat().st_size / 1024:,.0f} KiB\n"
        f"    in : {inputs}\n    out: {outputs}"
    )


def models() -> dict:
    """Every model this script writes, and the settings it is served with.

    The batch limits differ on purpose: a scaler can take a batch of 64
    because a row costs nothing, and stage II takes four because a row is
    768 KB of output and tens of milliseconds of work.
    """
    return {
        "occupancy_mlp": [
            (
                "1",
                lambda: occupancy_mlp((32, 16), salt=1),
                config("Dense occupancy regressor (32x16 hidden).", 32, features=8),
            ),
            # A second version so that version resolution and
            # /v2/models/info have something real to report: an unversioned
            # request resolves to 2, and 1 stays servable by name.
            (
                "2",
                lambda: occupancy_mlp((64, 32), salt=11),
                config("Dense occupancy regressor (64x32 hidden).", 32, features=8),
            ),
        ],
        "energy_autoencoder": [
            (
                "1",
                energy_autoencoder,
                config("Reconstruction and anomaly score for building features.", 32),
            )
        ],
        "thermal_cnn": [
            (
                "1",
                thermal_cnn,
                config(
                    "Four-class classifier over 32x32 thermal frames.",
                    16,
                    classes=["sound", "thermal_bridge", "moisture", "insulation_gap"],
                ),
            )
        ],
        "meter_digit_cnn": [
            ("1", meter_digit_cnn, config("Digit classifier over 28x28 meter photographs.", 16))
        ],
        "stackgan_stage1": [
            (
                "1",
                stackgan_stage1,
                config("StackGAN stage I: 64x64 image from an embedding and noise.", 8),
            )
        ],
        "stackgan_stage2": [
            (
                "1",
                stackgan_stage2,
                config("StackGAN stage II: 256x256 refinement of a stage I image.", 4),
            )
        ],
        "load_forecast_tcn": [
            ("1", load_forecast_tcn, config("24-hour load forecast from 168 hourly readings.", 32))
        ],
        "feature_scaler": [
            ("1", feature_scaler, config("Standardises the eight building features.", 64))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models",
        help="model repository to write into (default: ./models)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="MODEL",
        help="write only this model; repeat for several",
    )
    parser.add_argument(
        "--list", action="store_true", help="print the model names and exit"
    )
    args = parser.parse_args()

    catalogue = models()
    if args.list:
        for name in catalogue:
            print(name)
        return

    wanted = args.only or list(catalogue)
    unknown = [name for name in wanted if name not in catalogue]
    if unknown:
        parser.error(f"unknown model(s): {', '.join(unknown)}")

    total = 0
    for name in wanted:
        for version, builder, cfg in catalogue[name]:
            path = write(args.repository, name, version, builder(), cfg)
            total += path.stat().st_size
            print(describe(path))

    print(f"\n{len(wanted)} model(s) written, {total / (1024 * 1024):,.1f} MiB in total")


if __name__ == "__main__":
    main()
