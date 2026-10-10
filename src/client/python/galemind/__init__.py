"""A Predict Protocol v2 client for the GaleMind inference server."""

from .client import (
    DEFAULT_GRPC_ENDPOINT,
    DEFAULT_HTTP_ENDPOINT,
    CatalogueEntry,
    GrpcClient,
    HttpClient,
    InferenceError,
    InferenceResponse,
    ModelInfo,
    ModelMetadata,
    ServerMetadata,
    TensorMetadata,
    VersionInfo,
)
from .tensors import BINARY_ENCODING_PARAMETER, ELEMENT_SIZE, Tensor, datatypes

__version__ = "0.2.0"

__all__ = [
    "BINARY_ENCODING_PARAMETER",
    "DEFAULT_GRPC_ENDPOINT",
    "DEFAULT_HTTP_ENDPOINT",
    "ELEMENT_SIZE",
    "CatalogueEntry",
    "GrpcClient",
    "HttpClient",
    "InferenceError",
    "InferenceResponse",
    "ModelInfo",
    "ModelMetadata",
    "ServerMetadata",
    "Tensor",
    "TensorMetadata",
    "VersionInfo",
    "datatypes",
    "__version__",
]
