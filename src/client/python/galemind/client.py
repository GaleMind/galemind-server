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


def _tensor_metadata(tensors: Sequence[Dict[str, Any]]) -> List[TensorMetadata]:
    """Reads the protocol's tensor metadata objects."""
    return [
        TensorMetadata(name=t["name"], datatype=t["datatype"], shape=list(t["shape"]))
        for t in tensors
    ]


@dataclass(frozen=True)
class CatalogueEntry:
    """One model in `GET /v2/models/list`."""

    name: str
    versions: List[str]
    latest_version: str
    platform: str
    loaded_versions: List[str]
    #: Whether a request naming no version would be served without a load.
    ready: bool


@dataclass(frozen=True)
class VersionInfo:
    """One version in `GET /v2/models/info/{model}`."""

    version: str
    platform: str
    digest: str
    loaded: bool
    is_latest: bool
    inputs: List[TensorMetadata]
    outputs: List[TensorMetadata]
    size_bytes: Optional[int] = None
    max_batch_size: Optional[int] = None
    intra_op_threads: Optional[int] = None


@dataclass(frozen=True)
class ModelInfo:
    """Every version of one model."""

    name: str
    latest_version: str
    versions: List[VersionInfo]


class HttpClient:
    """Talks the protocol over HTTP, using only the standard library."""

    def __init__(
        self,
        endpoint: str = DEFAULT_HTTP_ENDPOINT,
        timeout: float = 30.0,
        token: Optional[str] = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        #: Sent as `Authorization: Bearer <token>` when the server requires
        #: one. The server is open unless configured otherwise.
        self.token = token

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
        return ModelMetadata(
            name=body["name"],
            platform=body["platform"],
            versions=body.get("versions", []),
            inputs=_tensor_metadata(body.get("inputs", [])),
            outputs=_tensor_metadata(body.get("outputs", [])),
        )

    # -- the catalogue, which the protocol does not define ----------------

    def model_list(self) -> List[CatalogueEntry]:
        """Every model the server can serve, from `/v2/models/list`.

        Loads nothing, so a repository larger than the server's model cache
        can be listed without evicting a model that requests are using.
        """
        body = self._get("/v2/models/list")
        return [
            CatalogueEntry(
                name=model["name"],
                versions=list(model.get("versions", [])),
                latest_version=model["latest_version"],
                platform=model.get("platform", ""),
                loaded_versions=list(model.get("loaded_versions", [])),
                ready=bool(model.get("ready", False)),
            )
            for model in body.get("models", [])
        ]

    def model_info(self, model: str, load: bool = True) -> ModelInfo:
        """Every version of one model, from `/v2/models/info/{model}`.

        A signature comes from a loaded session, so by default the server
        loads each version that is not already in memory. Pass
        ``load=False`` to skip that: the versions, their settings and their
        digests come back with empty inputs and outputs.
        """
        body = self._get(f"/v2/models/info/{model}?load={'true' if load else 'false'}")
        return ModelInfo(
            name=body["name"],
            latest_version=body["latest_version"],
            versions=[
                VersionInfo(
                    version=version["version"],
                    platform=version.get("platform", ""),
                    digest=version.get("digest", ""),
                    loaded=bool(version.get("loaded", False)),
                    is_latest=bool(version.get("is_latest", False)),
                    inputs=_tensor_metadata(version.get("inputs", [])),
                    outputs=_tensor_metadata(version.get("outputs", [])),
                    size_bytes=version.get("size_bytes"),
                    max_batch_size=version.get("max_batch_size"),
                    intra_op_threads=version.get("intra_op_threads"),
                )
                for version in body.get("versions", [])
            ],
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

    def _authorize(self, request: "urllib.request.Request") -> None:
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")

    def _health(self, path: str) -> bool:
        """Reads a health endpoint, where the status code is the whole answer."""
        request = urllib.request.Request(self.endpoint + path, method="GET")
        # Health endpoints are never authenticated, but a proxy in front of
        # the server may want the header anyway.
        self._authorize(request)
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
        self._authorize(request)

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

    def __init__(
        self, endpoint: str = DEFAULT_GRPC_ENDPOINT, token: Optional[str] = None
    ) -> None:
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
        # gRPC metadata is HTTP/2 headers, so this is the same
        # `authorization: Bearer <token>` the HTTP client sends, and the
        # server checks it against the same authenticator.
        self._metadata = [("authorization", f"Bearer {token}")] if token else None

    def _call(self, method: str, request: Any) -> Any:
        """Makes one call, with the credential if there is one.

        Every call goes through here so that a token cannot be forgotten on
        one of them -- the failure that looks like a server bug.
        """
        return getattr(self._stub, method)(request, metadata=self._metadata)

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> "GrpcClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def server_live(self) -> bool:
        return self._call("ServerLive", self._pb.ServerLiveRequest()).live

    def server_ready(self) -> bool:
        return self._call("ServerReady", self._pb.ServerReadyRequest()).ready

    def model_ready(self, model: str, version: Optional[str] = None) -> bool:
        request = self._pb.ModelReadyRequest(name=model, version=version or "")
        return self._call("ModelReady", request).ready

    def server_metadata(self) -> ServerMetadata:
        answer = self._call("ServerMetadata", self._pb.ServerMetadataRequest())
        return ServerMetadata(
            name=answer.name, version=answer.version, extensions=list(answer.extensions)
        )

    def model_metadata(self, model: str, version: Optional[str] = None) -> ModelMetadata:
        request = self._pb.ModelMetadataRequest(name=model, version=version or "")
        answer = self._call("ModelMetadata", request)
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

    # -- the catalogue, which the protocol does not define ----------------

    def model_list(self) -> List[CatalogueEntry]:
        """Every model the server can serve, from `/v2/models/list`.

        Loads nothing, so a repository larger than the server's model cache
        can be listed without evicting a model that requests are using.
        """
        body = self._get("/v2/models/list")
        return [
            CatalogueEntry(
                name=model["name"],
                versions=list(model.get("versions", [])),
                latest_version=model["latest_version"],
                platform=model.get("platform", ""),
                loaded_versions=list(model.get("loaded_versions", [])),
                ready=bool(model.get("ready", False)),
            )
            for model in body.get("models", [])
        ]

    def model_info(self, model: str, load: bool = True) -> ModelInfo:
        """Every version of one model, from `/v2/models/info/{model}`.

        A signature comes from a loaded session, so by default the server
        loads each version that is not already in memory. Pass
        ``load=False`` to skip that: the versions, their settings and their
        digests come back with empty inputs and outputs.
        """
        body = self._get(f"/v2/models/info/{model}?load={'true' if load else 'false'}")
        return ModelInfo(
            name=body["name"],
            latest_version=body["latest_version"],
            versions=[
                VersionInfo(
                    version=version["version"],
                    platform=version.get("platform", ""),
                    digest=version.get("digest", ""),
                    loaded=bool(version.get("loaded", False)),
                    is_latest=bool(version.get("is_latest", False)),
                    inputs=_tensor_metadata(version.get("inputs", [])),
                    outputs=_tensor_metadata(version.get("outputs", [])),
                    size_bytes=version.get("size_bytes"),
                    max_batch_size=version.get("max_batch_size"),
                    intra_op_threads=version.get("intra_op_threads"),
                )
                for version in body.get("versions", [])
            ],
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

        answer = self._call("ModelInfer", request)
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
