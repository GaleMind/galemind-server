"""Tests for the Python client, served by a stub HTTP server.

The stub answers exactly what the specification says a compliant server
answers, so these tests check the client against the protocol rather than
against our server's particular behaviour.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Iterator, List, Tuple

import pytest

from galemind import HttpClient, InferenceError, Tensor
from galemind.tensors import element_count, flatten, from_raw, to_raw

#: Requests the stub server received, newest last.
RECEIVED: List[Tuple[str, str, Dict[str, Any]]] = []

MODEL_METADATA = {
    "name": "building_energy",
    "versions": ["1"],
    "platform": "onnx_onnxv1",
    "inputs": [{"name": "features", "datatype": "FP32", "shape": [-1, 8]}],
    "outputs": [{"name": "kwh", "datatype": "FP32", "shape": [-1, 1]}],
}


class Handler(BaseHTTPRequestHandler):
    """A minimal compliant server, enough to exercise the client."""

    def log_message(self, *_: Any) -> None:
        pass

    def _send(self, status: int, body: Any = None) -> None:
        payload = b"" if body is None else json.dumps(body).encode()
        self.send_response(status)
        if payload:
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - the base class names it this
        RECEIVED.append(("GET", self.path, {}))

        if self.path in ("/v2/health/live", "/v2/health/ready"):
            self._send(200)
        elif self.path == "/v2/models/building_energy/ready":
            self._send(200)
        elif self.path == "/v2/models/ghost/ready":
            self._send(400)
        elif self.path == "/v2":
            self._send(200, {"name": "galemind", "version": "0.2.0", "extensions": []})
        elif self.path in (
            "/v2/models/building_energy",
            "/v2/models/building_energy/versions/1",
        ):
            self._send(200, MODEL_METADATA)
        elif self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"gm_inference_request_success 1\n")
        else:
            self._send(404, {"error": f"no such endpoint: {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        RECEIVED.append(("POST", self.path, body))

        if not self.path.endswith("/infer"):
            self._send(404, {"error": "no such endpoint"})
            return
        if "ghost" in self.path:
            self._send(404, {"error": "model 'ghost' is not known to this server"})
            return

        rows = body["inputs"][0]["shape"][0]
        requested = {output["name"] for output in body.get("outputs", [])}
        outputs = [
            {
                "name": "kwh",
                "shape": [rows, 1],
                "datatype": "FP32",
                "data": [float(index) for index in range(rows)],
            },
            {
                "name": "band",
                "shape": [rows],
                "datatype": "INT64",
                "data": [1] * rows,
            },
        ]
        if requested:
            outputs = [output for output in outputs if output["name"] in requested]

        answer: Dict[str, Any] = {
            "model_name": "building_energy",
            "model_version": "1",
            "outputs": outputs,
        }
        if "id" in body:
            answer["id"] = body["id"]
        self._send(200, answer)


@pytest.fixture()
def client() -> Iterator[HttpClient]:
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    RECEIVED.clear()
    try:
        yield HttpClient(f"http://127.0.0.1:{server.server_port}", timeout=5.0)
    finally:
        server.shutdown()
        server.server_close()


# -- tensors --------------------------------------------------------------


def test_nested_and_flat_data_describe_the_same_tensor():
    nested = Tensor(name="t", shape=[2, 2], datatype="FP32", data=[[1, 2], [3, 4]])
    flat = Tensor(name="t", shape=[2, 2], datatype="FP32", data=[1, 2, 3, 4])
    assert nested.data == flat.data == [1, 2, 3, 4]
    assert flatten([[1, [2]], 3]) == [1, 2, 3]


def test_a_tensor_whose_data_contradicts_its_shape_is_refused():
    with pytest.raises(ValueError, match="4 elements"):
        Tensor(name="t", shape=[2, 2], datatype="FP32", data=[1.0])


def test_a_variable_dimension_is_not_a_concrete_shape():
    with pytest.raises(ValueError, match="not concrete"):
        element_count([-1, 4])


def test_rows_are_addressable():
    tensor = Tensor(name="t", shape=[3, 2], datatype="INT64", data=[1, 2, 3, 4, 5, 6])
    assert tensor.rows == 3
    assert tensor.row(1) == [3, 4]


@pytest.mark.parametrize(
    "datatype,values",
    [
        ("BOOL", [True, False]),
        ("UINT8", [0, 255]),
        ("UINT32", [1, 4294967295]),
        ("INT8", [-128, 127]),
        ("INT64", [-5, 5]),
        ("FP32", [1.5, -2.5]),
        ("FP64", [0.5, -0.125]),
        ("FP16", [1.5, -0.25]),
    ],
)
def test_raw_packing_round_trips_every_fixed_width_datatype(datatype, values):
    raw = to_raw(datatype, values)
    assert from_raw(datatype, raw, len(values)) == pytest.approx(values)


def test_raw_packing_round_trips_variable_width_bytes():
    values = [b"abc", b"", b"de"]
    raw = to_raw("BYTES", values)
    assert from_raw("BYTES", raw, 3) == values


def test_raw_unpacking_rejects_a_payload_of_the_wrong_size():
    with pytest.raises(ValueError, match="expected 8"):
        from_raw("FP32", b"\x00" * 6, 2)


def test_text_bytes_stay_readable_in_json_and_binary_declares_base64():
    text = Tensor(name="t", shape=[2], datatype="BYTES", data=[b"hello", b"world"])
    assert text.to_json()["data"] == ["hello", "world"]
    assert "parameters" not in text.to_json()

    binary = Tensor(name="t", shape=[1], datatype="BYTES", data=[b"\xff\x00\xfe"])
    body = binary.to_json()
    assert body["parameters"]["binary_encoding"] == "base64"
    assert Tensor.from_json(body).data == [b"\xff\x00\xfe"]


# -- the six calls --------------------------------------------------------


def test_health_calls_report_true_and_false(client):
    assert client.server_live() is True
    assert client.server_ready() is True
    assert client.model_ready("building_energy") is True
    assert client.model_ready("ghost") is False


def test_server_metadata_is_read(client):
    metadata = client.server_metadata()
    assert metadata.name == "galemind"
    assert metadata.extensions == []


def test_model_metadata_describes_the_inputs_and_outputs(client):
    metadata = client.model_metadata("building_energy")
    assert metadata.platform == "onnx_onnxv1"
    assert metadata.versions == ["1"]
    assert metadata.sole_input.name == "features"
    assert metadata.sole_input.row_width == 8
    assert metadata.sole_input.batchable


def test_a_version_is_put_in_the_url(client):
    client.model_metadata("building_energy", version="1")
    assert RECEIVED[-1][1] == "/v2/models/building_energy/versions/1"


def test_inference_sends_the_protocols_body_and_reads_the_answer(client):
    tensor = Tensor(
        name="features", shape=[2, 8], datatype="FP32", data=[0.0] * 16
    )
    response = client.infer("building_energy", [tensor], request_id="req-1")

    _, path, body = RECEIVED[-1]
    assert path == "/v2/models/building_energy/infer"
    assert body["id"] == "req-1"
    assert body["inputs"][0]["datatype"] == "FP32"
    assert body["inputs"][0]["shape"] == [2, 8]

    assert response.model_name == "building_energy"
    assert response.model_version == "1"
    assert response.id == "req-1"
    assert response.output("kwh").shape == [2, 1]
    assert response.output("band").data == [1, 1]


def test_only_the_requested_outputs_come_back(client):
    tensor = Tensor(name="features", shape=[1, 8], datatype="FP32", data=[0.0] * 8)
    response = client.infer("building_energy", [tensor], outputs=["band"])
    assert [t.name for t in response.outputs] == ["band"]

    with pytest.raises(KeyError, match="no output named"):
        response.output("kwh")


def test_predict_fills_the_input_from_the_models_metadata(client):
    response = client.predict("building_energy", [[1.0] * 8, [2.0] * 8])

    _, _, body = RECEIVED[-1]
    assert body["inputs"][0]["name"] == "features"
    assert body["inputs"][0]["shape"] == [2, 8]
    assert response.output("kwh").rows == 2


def test_predict_checks_the_row_width_before_sending(client):
    with pytest.raises(InferenceError, match="takes 8"):
        client.predict("building_energy", [[1.0, 2.0]])
    assert RECEIVED[-1][1] == "/v2/models/building_energy"


def test_a_server_failure_keeps_the_servers_own_wording(client):
    tensor = Tensor(name="features", shape=[1, 8], datatype="FP32", data=[0.0] * 8)
    with pytest.raises(InferenceError) as failure:
        client.infer("ghost", [tensor])

    assert str(failure.value) == "model 'ghost' is not known to this server"
    assert failure.value.is_client_error


def test_an_unreachable_server_is_reported_clearly():
    unreachable = HttpClient("http://127.0.0.1:1", timeout=1.0)
    with pytest.raises(InferenceError, match="cannot reach"):
        unreachable.server_live()


def test_metrics_are_returned_as_text(client):
    assert "gm_inference_request_success" in client.metrics()
