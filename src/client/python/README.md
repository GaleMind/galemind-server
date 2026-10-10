# GaleMind Python client

A [Predict Protocol v2](https://kserve.github.io/website/latest/modelserving/data_plane/v2_protocol/)
client. The HTTP client needs only the standard library; gRPC and numpy are
optional extras.

## Install

```bash
pip install -e .            # HTTP only
pip install -e '.[grpc]'    # add the gRPC client
pip install -e '.[dev]'     # add pytest and the stub generator
```

## Use

```python
from galemind import HttpClient

client = HttpClient("http://localhost:8000")
assert client.server_ready()

# The model's metadata supplies the input name, datatype and row width,
# so only the numbers are needed here.
response = client.predict("building_energy", [
    [1800, 30, -4, 18, 22, 1, 0, 40],   # a large, poorly insulated office
    [55, 2, 18, 14, 21, 5, 1, 600],     # a small, well insulated flat
])
print(response.output("kwh").data)
```

### What the server can serve

The catalogue endpoints are HTTP extensions, not part of the protocol, so they
exist on `HttpClient` only:

```python
for model in client.model_list():
    print(model.name, model.versions, "latest:", model.latest_version,
          "ready:" , model.ready)

info = client.model_info("occupancy_mlp")          # loads each version
for version in info.versions:
    print(version.version, version.size_bytes, version.digest[:8],
          [(t.name, t.datatype, t.shape) for t in version.inputs])

client.model_info("stackgan_stage2", load=False)   # settings only, no loads
```

A signature comes from a loaded session, so `model_info` loads each version
that is not already in memory; `load=False` returns the versions, their
settings and their digests with empty `inputs` and `outputs`.

### Authentication

The server is open unless it is configured otherwise. Where it requires a
bearer token, give the client one and it goes on every request -- on either
transport, since gRPC metadata is HTTP/2 headers and the server checks both
against the same authenticator:

```python
client = HttpClient("http://localhost:8000", token="s3cret")

with GrpcClient("127.0.0.1:8001", token="s3cret") as grpc_client:
    grpc_client.model_metadata("building_energy")
```

Health and readiness checks need no token, so a probe keeps working when
authentication is switched on.

### Tensors

For full control over tensors, build them directly:

```python
from galemind import HttpClient, Tensor

client = HttpClient()
features = Tensor(
    name="features",
    shape=[1, 8],
    datatype="FP32",
    data=[[1800, 30, -4, 18, 22, 1, 0, 40]],   # nested or flat, both work
)
response = client.infer("building_energy", [features], outputs=["kwh"])
```

Over gRPC, after `make generate`:

```python
from galemind import GrpcClient, Tensor

with GrpcClient("127.0.0.1:8001") as client:
    print(client.model_metadata("building_energy").outputs)
```

## Develop

```bash
make generate   # regenerate the gRPC stubs from ../../grpc_server/proto
make test       # run the client tests
```
