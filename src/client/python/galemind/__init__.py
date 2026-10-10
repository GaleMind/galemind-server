"""A Predict Protocol v2 client for the GaleMind inference server."""

from .client import (
    DEFAULT_GRPC_ENDPOINT,
    DEFAULT_HTTP_ENDPOINT,
    GrpcClient,
    HttpClient,
    InferenceError,
    InferenceResponse,
    ModelMetadata,
    ServerMetadata,
    TensorMetadata,
)
from .tensors import BINARY_ENCODING_PARAMETER, ELEMENT_SIZE, Tensor, datatypes

__version__ = "0.2.0"

__all__ = [
    "BINARY_ENCODING_PARAMETER",
    "DEFAULT_GRPC_ENDPOINT",
    "DEFAULT_HTTP_ENDPOINT",
    "ELEMENT_SIZE",
    "GrpcClient",
    "HttpClient",
    "InferenceError",
    "InferenceResponse",
    "ModelMetadata",
    "ServerMetadata",
    "Tensor",
    "TensorMetadata",
    "datatypes",
    "__version__",
]
