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
