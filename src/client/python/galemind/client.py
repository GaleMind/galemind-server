"""Predict Protocol v2 clients.

The HTTP client needs only the standard library. The gRPC client needs
``grpcio`` and the generated stubs (see the Makefile), and is imported lazily so
that the HTTP client works without them.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .tensors import Tensor, element_count, from_raw, to_raw

DEFAULT_HTTP_ENDPOINT = "http://127.0.0.1:8000"
DEFAULT_GRPC_ENDPOINT = "127.0.0.1:8001"


class InferenceError(RuntimeError):
    """A failure the server reported, or one reaching it.

    The server's own wording is kept: it knows what went wrong.
    """

    def __init__(self, message: str, *, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def is_client_error(self) -> bool:
        """Whether the request was at fault rather than the server."""
        return self.status is not None and 400 <= self.status < 500


@dataclass
class ServerMetadata:
    """The server's own description of itself."""

    name: str
    version: str
    extensions: List[str] = field(default_factory=list)


@dataclass
class TensorMetadata:
    """One model input or output. A variable dimension is ``-1``."""

    name: str
    datatype: str
    shape: List[int]

    @property
    def batchable(self) -> bool:
        """Whether the leading dimension is variable."""
        return bool(self.shape) and self.shape[0] < 0

    @property
    def row_width(self) -> int:
        """Values in one row of this tensor."""
        return element_count(self.shape[1:]) if len(self.shape) > 1 else 1


@dataclass
class ModelMetadata:
    """What the server reports about a model."""

    name: str
    platform: str
    inputs: List[TensorMetadata]
    outputs: List[TensorMetadata]
    versions: List[str] = field(default_factory=list)

    @property
    def sole_input(self) -> TensorMetadata:
        """The single input, for models that take exactly one."""
        if len(self.inputs) != 1:
            names = ", ".join(t.name for t in self.inputs)
            raise InferenceError(
                f"model {self.name!r} takes {len(self.inputs)} inputs ({names}), "
                "so each must be given explicitly"
            )
        return self.inputs[0]


@dataclass
class InferenceResponse:
    """One inference result."""

    model_name: str
    model_version: str
    outputs: List[Tensor]
    id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

    def output(self, name: str) -> Tensor:
        """One output by name."""
        for tensor in self.outputs:
            if tensor.name == name:
                return tensor
        available = ", ".join(t.name for t in self.outputs)
        raise KeyError(f"no output named {name!r}; the response holds {available}")


class HttpClient:
    """Talks the protocol over HTTP, using only the standard library."""

    def __init__(self, endpoint: str = DEFAULT_HTTP_ENDPOINT, timeout: float = 30.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    # -- the six calls ----------------------------------------------------

    def server_live(self) -> bool:
        return self._health("/v2/health/live")

    def server_ready(self) -> bool:
        return self._health("/v2/health/ready")

    def model_ready(self, model: str, version: Optional[str] = None) -> bool:
        return self._health(self._model_path(model, version, "/ready"))

    def server_metadata(self) -> ServerMetadata:
        body = self._get("/v2")
        return ServerMetadata(
            name=body["name"],
            version=body["version"],
            extensions=body.get("extensions", []),
        )

    def model_metadata(self, model: str, version: Optional[str] = None) -> ModelMetadata:
        body = self._get(self._model_path(model, version, ""))
        tensors = lambda key: [  # noqa: E731 - a local alias reads better here
            TensorMetadata(name=t["name"], datatype=t["datatype"], shape=list(t["shape"]))
            for t in body.get(key, [])
        ]
        return ModelMetadata(
            name=body["name"],
            platform=body["platform"],
            versions=body.get("versions", []),
            inputs=tensors("inputs"),
            outputs=tensors("outputs"),
        )

    def infer(
        self,
        model: str,
        inputs: Sequence[Tensor],
        *,
        version: Optional[str] = None,
        request_id: Optional[str] = None,
        outputs: Optional[Sequence[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> InferenceResponse:
        """Runs an inference and returns the response."""
        body: Dict[str, Any] = {"inputs": [tensor.to_json() for tensor in inputs]}
        if request_id is not None:
            body["id"] = request_id
        if outputs:
            body["outputs"] = [{"name": name} for name in outputs]
        if parameters:
            body["parameters"] = parameters

        answer = self._post(self._model_path(model, version, "/infer"), body)
        return InferenceResponse(
            model_name=answer["model_name"],
            model_version=answer.get("model_version", ""),
            id=answer.get("id"),
            parameters=answer.get("parameters"),
            outputs=[Tensor.from_json(output) for output in answer["outputs"]],
        )

    def predict(
        self,
        model: str,
        rows: Sequence[Sequence[float]],
        *,
        version: Optional[str] = None,
        request_id: Optional[str] = None,
        outputs: Optional[Sequence[str]] = None,
    ) -> InferenceResponse:
        """Runs an inference from rows of numbers.

        The model's own metadata supplies the input's name, datatype and row
        width, so a caller only has to provide the values.
        """
        metadata = self.model_metadata(model, version)
        spec = metadata.sole_input

        for index, row in enumerate(rows):
            if len(row) != spec.row_width:
                raise InferenceError(
                    f"row {index + 1} holds {len(row)} values but input "
                    f"{spec.name!r} takes {spec.row_width}"
                )

        flat = [value for row in rows for value in row]
        shape = [len(rows)] + list(spec.shape[1:])
        tensor = Tensor(name=spec.name, shape=shape, datatype=spec.datatype, data=flat)
        return self.infer(
            model,
            [tensor],
            version=version or (metadata.versions[-1] if metadata.versions else None),
            request_id=request_id,
            outputs=outputs,
        )

    def metrics(self) -> str:
        """The Prometheus exposition text, for monitoring and load reports."""
        return self._request("GET", "/metrics", None).decode()

    # -- plumbing ---------------------------------------------------------

    def _model_path(self, model: str, version: Optional[str], suffix: str) -> str:
        if version:
            return f"/v2/models/{model}/versions/{version}{suffix}"
        return f"/v2/models/{model}{suffix}"

    def _get(self, path: str) -> Dict[str, Any]:
        return json.loads(self._request("GET", path, None))

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return json.loads(self._request("POST", path, json.dumps(body).encode()))

    def _health(self, path: str) -> bool:
        """Reads a health endpoint, where the status code is the whole answer."""
        request = urllib.request.Request(self.endpoint + path, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status == 200
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                return False
            raise InferenceError(f"{path} answered {e.code}", status=e.code) from e
        except urllib.error.URLError as e:
            raise InferenceError(f"cannot reach {self.endpoint}{path}: {e.reason}") from e

    def _request(self, method: str, path: str, payload: Optional[bytes]) -> bytes:
        request = urllib.request.Request(self.endpoint + path, data=payload, method=method)
        if payload is not None:
            request.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                message = json.loads(raw)["error"]
            except (ValueError, KeyError, TypeError):
                message = raw.decode(errors="replace") or f"HTTP {e.code}"
            raise InferenceError(message, status=e.code) from e
        except urllib.error.URLError as e:
            raise InferenceError(f"cannot reach {self.endpoint}{path}: {e.reason}") from e


class GrpcClient:
    """Talks the protocol over gRPC.

    Needs ``grpcio`` and the stubs generated by ``make generate``.
    """

    def __init__(self, endpoint: str = DEFAULT_GRPC_ENDPOINT) -> None:
        try:
            import grpc  # noqa: F401 - imported for its side effect of existing

            from . import grpc_predict_v2_pb2 as pb
            from . import grpc_predict_v2_pb2_grpc as pb_grpc
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise InferenceError(
                "the gRPC client needs grpcio and the generated stubs: "
                "pip install 'galemind-client[grpc]' && make generate"
            ) from e

        self._pb = pb
        self.endpoint = endpoint.replace("http://", "").replace("https://", "")
        self._channel = grpc.insecure_channel(self.endpoint)
        self._stub = pb_grpc.GRPCInferenceServiceStub(self._channel)

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> "GrpcClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def server_live(self) -> bool:
        return self._stub.ServerLive(self._pb.ServerLiveRequest()).live

    def server_ready(self) -> bool:
        return self._stub.ServerReady(self._pb.ServerReadyRequest()).ready

    def model_ready(self, model: str, version: Optional[str] = None) -> bool:
        request = self._pb.ModelReadyRequest(name=model, version=version or "")
        return self._stub.ModelReady(request).ready

    def server_metadata(self) -> ServerMetadata:
        answer = self._stub.ServerMetadata(self._pb.ServerMetadataRequest())
        return ServerMetadata(
            name=answer.name, version=answer.version, extensions=list(answer.extensions)
        )

    def model_metadata(self, model: str, version: Optional[str] = None) -> ModelMetadata:
        request = self._pb.ModelMetadataRequest(name=model, version=version or "")
        answer = self._stub.ModelMetadata(request)
        convert = lambda ts: [  # noqa: E731
            TensorMetadata(name=t.name, datatype=t.datatype, shape=list(t.shape)) for t in ts
        ]
        return ModelMetadata(
            name=answer.name,
            platform=answer.platform,
            versions=list(answer.versions),
            inputs=convert(answer.inputs),
            outputs=convert(answer.outputs),
        )

    def infer(
        self,
        model: str,
        inputs: Sequence[Tensor],
        *,
        version: Optional[str] = None,
        request_id: Optional[str] = None,
        outputs: Optional[Sequence[str]] = None,
    ) -> InferenceResponse:
        """Runs an inference, sending contents in the raw form.

        Raw contents are what the protocol recommends for performance, and the
        only form able to carry FP16 and BF16 exactly.
        """
        request = self._pb.ModelInferRequest(
            model_name=model,
            model_version=version or "",
            id=request_id or "",
        )
        for tensor in inputs:
            request.inputs.append(
                self._pb.ModelInferRequest.InferInputTensor(
                    name=tensor.name,
                    datatype=tensor.datatype,
                    shape=tensor.shape,
                )
            )
            request.raw_input_contents.append(to_raw(tensor.datatype, tensor.data))
        for name in outputs or []:
            request.outputs.append(
                self._pb.ModelInferRequest.InferRequestedOutputTensor(name=name)
            )

        answer = self._stub.ModelInfer(request)
        tensors = []
        for index, output in enumerate(answer.outputs):
            count = element_count(list(output.shape))
            raw = answer.raw_output_contents[index]
            tensors.append(
                Tensor(
                    name=output.name,
                    shape=list(output.shape),
                    datatype=output.datatype,
                    data=from_raw(output.datatype, raw, count),
                )
            )

        return InferenceResponse(
            model_name=answer.model_name,
            model_version=answer.model_version,
            id=answer.id or None,
            outputs=tensors,
        )
