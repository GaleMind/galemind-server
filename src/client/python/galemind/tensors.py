"""Tensor datatypes and contents, as the Predict Protocol v2 defines them."""

from __future__ import annotations

import base64
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

#: Size in bytes of one element of each fixed-width datatype.
ELEMENT_SIZE: Dict[str, int] = {
    "BOOL": 1,
    "UINT8": 1,
    "UINT16": 2,
    "UINT32": 4,
    "UINT64": 8,
    "INT8": 1,
    "INT16": 2,
    "INT32": 4,
    "INT64": 8,
    "FP16": 2,
    "FP32": 4,
    "FP64": 8,
    "BF16": 2,
}

#: Datatypes whose elements are variable-length byte strings.
VARIABLE_WIDTH = {"BYTES"}

#: `struct` format character for each datatype that `struct` can pack.
_STRUCT_FORMAT: Dict[str, str] = {
    "BOOL": "?",
    "UINT8": "B",
    "UINT16": "H",
    "UINT32": "I",
    "UINT64": "Q",
    "INT8": "b",
    "INT16": "h",
    "INT32": "i",
    "INT64": "q",
    "FP16": "e",
    "FP32": "f",
    "FP64": "d",
}

#: The parameter naming how BYTES strings are encoded in JSON.
BINARY_ENCODING_PARAMETER = "binary_encoding"


def datatypes() -> List[str]:
    """Every datatype the protocol defines."""
    return sorted(set(ELEMENT_SIZE) | VARIABLE_WIDTH)


def element_count(shape: Sequence[int]) -> int:
    """How many elements a shape describes."""
    count = 1
    for dimension in shape:
        if dimension < 0:
            raise ValueError(f"shape {list(shape)} is not concrete")
        count *= dimension
    return count


def flatten(data: Any) -> List[Any]:
    """Flattens nested tensor data into row-major order.

    The protocol allows data either nested or already flat, so both forms are
    accepted and produce the same result.
    """
    if not isinstance(data, (list, tuple)):
        return [data]

    flat: List[Any] = []
    for item in data:
        if isinstance(item, (list, tuple)):
            flat.extend(flatten(item))
        else:
            flat.append(item)
    return flat


def to_raw(datatype: str, values: Sequence[Any]) -> bytes:
    """Packs elements into the protocol's raw little-endian form."""
    if datatype == "BYTES":
        out = bytearray()
        for value in values:
            element = value.encode() if isinstance(value, str) else bytes(value)
            out += struct.pack("<I", len(element)) + element
        return bytes(out)

    fmt = _STRUCT_FORMAT.get(datatype)
    if fmt is None:
        raise ValueError(f"{datatype} has no raw packing in this client")
    return struct.pack(f"<{len(values)}{fmt}", *values)


def from_raw(datatype: str, raw: bytes, count: int) -> List[Any]:
    """Unpacks the protocol's raw little-endian form."""
    if datatype == "BYTES":
        values: List[bytes] = []
        offset = 0
        while len(values) < count:
            (length,) = struct.unpack_from("<I", raw, offset)
            offset += 4
            values.append(raw[offset : offset + length])
            offset += length
        return values

    fmt = _STRUCT_FORMAT.get(datatype)
    if fmt is None:
        raise ValueError(f"{datatype} has no raw unpacking in this client")
    expected = count * ELEMENT_SIZE[datatype]
    if len(raw) != expected:
        raise ValueError(
            f"raw contents hold {len(raw)} bytes, expected {expected} for "
            f"{count} {datatype} elements"
        )
    return list(struct.unpack(f"<{count}{fmt}", raw))


@dataclass
class Tensor:
    """A named tensor: its shape, datatype and flattened contents."""

    name: str
    shape: List[int]
    datatype: str
    data: List[Any] = field(default_factory=list)
    parameters: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.data = flatten(self.data)
        expected = element_count(self.shape)
        if len(self.data) != expected:
            raise ValueError(
                f"tensor {self.name!r} declares shape {self.shape} "
                f"({expected} elements) but holds {len(self.data)}"
            )

    @property
    def rows(self) -> int:
        """Size of the leading dimension."""
        return self.shape[0] if self.shape else 1

    def row(self, index: int) -> List[Any]:
        """One row of the tensor."""
        width = len(self.data) // max(self.rows, 1)
        return self.data[index * width : (index + 1) * width]

    def to_json(self) -> Dict[str, Any]:
        """The tensor as the protocol's JSON object."""
        body: Dict[str, Any] = {
            "name": self.name,
            "shape": list(self.shape),
            "datatype": self.datatype,
        }

        data = self.data
        if self.datatype == "BYTES":
            # Bytes that are not text have no JSON form, so they are declared
            # as base64 rather than guessed at by the other side.
            decoded = []
            binary = False
            for value in data:
                if isinstance(value, str):
                    decoded.append(value)
                    continue
                try:
                    decoded.append(bytes(value).decode())
                except UnicodeDecodeError:
                    binary = True
                    break
            if binary:
                data = [base64.b64encode(bytes(v)).decode() for v in self.data]
                body["parameters"] = {BINARY_ENCODING_PARAMETER: "base64"}
            else:
                data = decoded
        if self.parameters:
            body.setdefault("parameters", {}).update(self.parameters)

        body["data"] = data
        return body

    @classmethod
    def from_json(cls, body: Dict[str, Any]) -> "Tensor":
        """Reads a tensor from the protocol's JSON object."""
        parameters = body.get("parameters") or {}
        data = flatten(body["data"])

        if body["datatype"] == "BYTES" and parameters.get(BINARY_ENCODING_PARAMETER) == "base64":
            data = [base64.b64decode(value) for value in data]

        return cls(
            name=body["name"],
            shape=list(body["shape"]),
            datatype=body["datatype"],
            data=data,
            parameters=parameters or None,
        )
