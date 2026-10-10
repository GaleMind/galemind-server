# GaleMind Server

A CPU inference server that speaks the [Predict Protocol
v2](https://github.com/kserve/open-inference-protocol) over HTTP and gRPC, and
nothing else. It loads ONNX models from a repository, merges concurrent
requests into batches, keeps the hot ones in memory, and reports what it is
doing in Prometheus text.

```bash
make rocksdb-lib          # once per clone: stage the prebuilt RocksDB archive
cargo build --release
cargo run --release -p galemind -- serve
gmctl models              # what the server can serve
gmctl infer building_energy --rows 4
```

- **Protocol**: the eight URLs of Predict Protocol v2, plus two catalogue
  endpoints of our own ([below](#the-catalogue-endpoints)).
- **Backends**: ONNX Runtime. Adding one means implementing
  `galemind_core::Backend` and registering a factory.
- **Stores**: RocksDB (default) or a directory tree.
- **Clients**: a Rust library and `gmctl` here, a Python client under
  `src/client/python`.

## Contents

- [Prerequisites](#prerequisites)
- [Building](#building)
- [The model repository](#the-model-repository)
- [Running the server](#running-the-server)
- [The API](#the-api)
- [Authentication](#authentication)
- [gmctl](#gmctl)
- [Load testing](#load-testing)
- [Metrics](#metrics)
- [Using the cores](#using-the-cores)
- [Memory](#memory)
- [Following a request](#following-a-request)
- [Docker](#docker)
- [Project layout](#project-layout)
- [The documentation site](#the-documentation-site)

## Prerequisites

- **Rust** 1.75+ — from [rustup.rs](https://rustup.rs/)
- **libssl-dev** — `reqwest` links against it: `sudo apt install libssl-dev`
- **protobuf-compiler** — the gRPC crate compiles the protocol's `.proto`:
  `sudo apt install protobuf-compiler`
- **libclang** — bindgen needs it for the default RocksDB store:
  `sudo apt install libclang-dev`. `make rocksdb-lib` symlinks an existing
  LLVM libclang if the unversioned `libclang.so` bindgen looks for is missing.
  Not needed with `--no-default-features`.

ONNX Runtime is downloaded by the `ort` crate at build time; nothing to
install.

## Building

```bash
git clone https://github.com/GaleMind/galemind-server.git
cd galemind-server
make rocksdb-lib     # once; see below
cargo build --release
```

That produces two binaries in `target/release`: `galemind`, the server, and
`gmctl`, the client. The examples below assume `target/release` is on your
`PATH`.

### RocksDB

The default `rocksdb-store` feature pulls in `librocksdb-sys`, which compiles
roughly 330 RocksDB C++ translation units from its build script. That takes
minutes, and the `cc` crate keeps no object cache, so anything that
invalidates the build script -- a different feature unification between
`cargo build` and `cargo test`, a moved `LIBCLANG_PATH`, or a single `Ctrl-C`
-- pays the full cost again.

Build the archive once and every later build links it instead:

```bash
make rocksdb-lib
```

This stages an optimized `librocksdb.a` into `.cargo/rocksdb/`, which
`.cargo/config.toml` points `ROCKSDB_LIB_DIR` at. The build script then only
runs bindgen, taking the workspace build from minutes to seconds. The archive
is a build artifact and is not committed, so run this once per clone, and
again after bumping the `rocksdb` crate -- bindgen always generates against
the newly vendored headers, and the staged archive has to match.

To ignore the staged archive and compile the vendored sources, set
`ROCKSDB_COMPILE=1`. To skip RocksDB altogether and fall back to the
filesystem store, build with `--no-default-features`. Docker builds are
unaffected: `.dockerignore` drops `.cargo/`, so they compile from source.

### Make targets

| Command | Description |
|---|---|
| `make all` | Format, test, build |
| `make test` | `cargo test` |
| `make format` | `cargo fmt` |
| `make run` | Start the server, loading `.env` first |
| `make models` | Regenerate the model repository |
| `make models-deps` | Install what `make models` needs |
| `make rocksdb-lib` | Stage the prebuilt RocksDB archive |
| `make docker-build` | Build the server image |
| `make docker-run` | Run it, publishing 8000 and 8001 |
| `cargo e2e` | The end-to-end suite (needs Docker) |
| `npm --prefix website start` | The documentation site, with live reload |

### End-to-end tests

The end-to-end suite drives a real server in a container, so it needs Docker
and a built server image and is not part of `cargo test`:

```bash
docker compose build server
cargo e2e
```

It lives in its own workspace under `src/e2e`, excluded from the root one.
Cargo unifies features across every package selected for a build, so while it
was a member, `cargo test` resolved shared leaf crates differently from
`cargo build`; that moved bindgen's fingerprint, and with it the
librocksdb-sys build script's, recompiling RocksDB on every alternation
between the two. The cost of the split is a separate `src/e2e/target`, and
dependency versions that have to be kept in step with the root
`[workspace.dependencies]` by hand.

## The model repository

A directory of models, each with versioned subdirectories:

```text
models/
  building_energy/
    config.json          # optional
    1/model.onnx
  occupancy_mlp/
    config.json
    1/model.onnx
    2/model.onnx         # an unversioned request resolves to the newest
```

A version directory's name is the version, so versions are whatever you call
them; numeric ones are compared as numbers, so `10` is newer than `9`. A model
directory holding `model.onnx` directly is served as version `1`.

`config.json` is optional, and every field in it is:

```json
{
  "platform": "onnx_onnxv1",
  "max_batch_size": 32,
  "intra_op_threads": 1
}
```

`max_batch_size` caps how many requests the server merges into one call for
this model. `intra_op_threads` overrides the server's `--intra-op-threads`
for this model alone -- raise it for a model with wide operators, and read
[Using the cores](#using-the-cores) before you do. Anything else in the file
is ignored, so a description, the feature names or the class labels can live
there too.

### The committed models

Ten models, eleven versions, all synthetic and all committed. They exist to be
served, not believed: the weights are random, so the shapes, the datatypes
and the cost are real and the predictions are not. Two are trained
scikit-learn ensembles on a synthetic building-energy dataset; the rest are
ONNX graphs built directly by `scripts/synthetic_models.py`.

| Model | Inputs | Outputs | Batch | What it is for |
|---|---|---|---|---|
| `building_energy` | `features[-1,8]` | `kwh[-1,1]` | 32 | Random-forest regressor |
| `building_energy_class` | `features[-1,8]` | `band[-1]`, `probabilities[-1,3]` | 32 | Random-forest classifier; an integer label beside float scores |
| `occupancy_mlp` | `features[-1,8]` | `occupancy[-1,1]` | 32 | Dense net, in **two versions**, so version resolution has something to resolve |
| `energy_autoencoder` | `features[-1,8]` | `reconstruction[-1,8]`, `anomaly_score[-1,1]` | 32 | Two outputs of different widths from one input |
| `feature_scaler` | `features[-1,8]` | `scaled[-1,8]` | 64 | Three operators; the latency floor, with no model to blame |
| `load_forecast_tcn` | `sequence[-1,1,168]` | `forecast[-1,24]` | 32 | Dilated 1-D convolutions over a week of hourly readings |
| `thermal_cnn` | `image[-1,1,32,32]` | `logits[-1,4]`, `probabilities[-1,4]` | 16 | Small CNN; requests are kilobytes |
| `meter_digit_cnn` | `image[-1,1,28,28]` | `digit[-1]`, `probabilities[-1,10]` | 16 | CNN with a label and scores |
| `stackgan_stage1` | `text_embedding[-1,128]`, `noise[-1,100]` | `image_64[-1,3,64,64]` | 8 | StackGAN stage I; **two inputs** |
| `stackgan_stage2` | `image_64[-1,3,64,64]`, `text_embedding[-1,128]` | `image_256[-1,3,256,256]` | 4 | StackGAN stage II; 768 KB per row, tens of ms per row |

The spread is the point. `feature_scaler` answers in microseconds and
`stackgan_stage2` holds a core for tens of milliseconds while returning
three quarters of a megabyte per row, and a server is judged on how it
behaves with both loaded at once. On a twelve-core machine, with one worker
and `--rows 4`:

```text
feature_scaler        55300 requests/s
building_energy       56914 requests/s
load_forecast_tcn     14192 requests/s
thermal_cnn            4562 requests/s
stackgan_stage1         324 requests/s
stackgan_stage2          20 requests/s
```

To regenerate or extend them:

```bash
make models-deps                                    # numpy, onnx, sklearn, skl2onnx
make models                                         # all of them
.venv/bin/python scripts/synthetic_models.py --list
.venv/bin/python scripts/synthetic_models.py --only stackgan_stage2
```

`cargo test -p galemind-onnx --test model_zoo` loads every model in the
repository, runs it, and checks that a batch gives each row the answer it
would have had alone. A generated graph that ONNX's checker accepts is not
necessarily one ONNX Runtime can serve, and that test is the difference.

## Running the server

```bash
cargo run --release -p galemind -- serve
```

Every option has an environment variable, so a container needs no arguments.
`make run` loads `.env` first.

### Addresses and the repository

| Option | Variable | Default |
|---|---|---|
| `--rest-address` | `GALEMIND_REST_ADDRESS` | `0.0.0.0:8000` |
| `--grpc-address` | `GALEMIND_GRPC_ADDRESS` | `0.0.0.0:8001` |
| `--listen-backlog` | `GALEMIND_LISTEN_BACKLOG` | `4096` |
| `--model-repository` | `GALEMIND_MODEL_REPOSITORY` | `models` |
| `--store` | `GALEMIND_STORE` | `rocksdb` |
| `--store-path` | `GALEMIND_STORE_PATH` | `data/models.db` |
| `--preload` | `GALEMIND_PRELOAD` | off |

`--listen-backlog` is the kernel's accept queue for both ports -- how many
completed handshakes may wait to be accepted, not a connection limit. It is
here rather than left at tokio's default of 1024 because a burst of clients
connecting at once fills it and the kernel then drops connections silently;
see [Load testing](#the-accept-queue).

With the default `rocksdb` store the directory tree is ingested at startup and
the server serves from RocksDB, so artifacts, their metadata and usage records
are durable and survive a restart. `--store filesystem` reads the tree
directly, which is simpler and what the examples here use. `--preload` loads
every model before reporting readiness, so a readiness probe passing means no
request will pay for a load.

### Cache, batching, metrics

| Option | Variable | Default | |
|---|---|---|---|
| `--cache-capacity` | `GALEMIND_CACHE_CAPACITY` | `8` | Models kept loaded; the least recently used is dropped first |
| `--cache-ttl-secs` | `GALEMIND_CACHE_TTL_SECS` | `3600` | Unload after this long unused |
| `--cache-sweep-secs` | `GALEMIND_CACHE_SWEEP_SECS` | `60` | How often to look for them |
| `--max-batch-size` | `GALEMIND_MAX_BATCH_SIZE` | `10` | Largest batch formed, whatever a model's config says; a ceiling on the size of one execution, which is what sets resident memory |
| `--min-batch-window-micros` | `GALEMIND_MIN_BATCH_WINDOW_MICROS` | `200` | Shortest wait for a batch to fill |
| `--max-batch-window-micros` | `GALEMIND_MAX_BATCH_WINDOW_MICROS` | `5000` | Longest wait; caps the latency batching adds |
| `--queue-capacity` | `GALEMIND_QUEUE_CAPACITY` | `1024` | Requests that may wait **per model** before the server reports itself overloaded |
| `--queue-timeout-secs` | `GALEMIND_QUEUE_TIMEOUT_SECS` | `10` | Fail a request that waited this long rather than run it late |
| `--max-concurrent-batches` | `GALEMIND_MAX_CONCURRENT_BATCHES` | `1` | Batches of one model that may be executing at once; see [One model, several cores](#one-model-several-cores) |
| `--memory-high-water-mb` | `GALEMIND_MEMORY_HIGH_WATER_MB` | `500` | Resident megabytes above which the server reclaims |
| `--memory-reclaim-cooldown-secs` | `GALEMIND_MEMORY_RECLAIM_COOLDOWN_SECS` | `30` | Wait this long before reclaiming again |
| `--allow-metrics` | `GALEMIND_ALLOW_METRICS` | `true` | Serve `/metrics` |
| `--metrics-interval-ms` | `GALEMIND_METRICS_INTERVAL_MS` | `1000` | How often CPU and memory are re-read |

**The cache capacity must cover the repository** if you intend to serve all of
it. This repository holds eleven versions; with the default capacity of eight,
a round of the whole zoo is a cycle of evictions and reloads. The counters
`gm_model_cache_eviction_count` and `gm_model_cache_miss_count` say when that
is happening.

**The queue capacity is per model**, and it is what the server rejects against:
10 000 requests in flight against one model with the default 1024 is an
`overloaded` response for most of them -- correct back-pressure, but not what
you wanted if you were trying to measure throughput. See [Load
testing](#load-testing).

### Threads and memory

| Option | Variable | Default | |
|---|---|---|---|
| `--worker-threads` | `GALEMIND_WORKER_THREADS` | one per core | Async workers; they decode, queue, encode and reply |
| `--blocking-threads` | `GALEMIND_BLOCKING_THREADS` | twice the cores | Where model code runs; bounds concurrent executions |
| `--intra-op-threads` | `GALEMIND_INTRA_OP_THREADS` | **derived** | Threads inside one operator, per loaded model |
| `--inter-op-threads` | `GALEMIND_INTER_OP_THREADS` | `0` | Threads across independent branches of a graph |
| `--parallel-execution` | `GALEMIND_PARALLEL_EXECUTION` | off | Run independent branches at once; for wide graphs |
| `--cpu-mem-arena` | `GALEMIND_CPU_MEM_ARENA` | off | ONNX Runtime's arena allocator |
| `--memory-high-water-mb` | `GALEMIND_MEMORY_HIGH_WATER_MB` | `500` | Reclaim above this |
| `--memory-reclaim-cooldown-secs` | `GALEMIND_MEMORY_RECLAIM_COOLDOWN_SECS` | `30` | Wait before reclaiming again |

The operator-thread default is derived from the machine and the repository --
the cores divided between the models that can be loaded at once, up to half
the machine -- which on a twelve-core host is six threads for one model and
one for ten. Both ends of that are measured in [Using the cores](#using-the-cores);
[Memory](#memory) covers the rest.

## The API

The eight URLs of the protocol, and nothing else under `/v2`:

```text
GET  /v2                                               server metadata
GET  /v2/health/live                                   200 or 4xx, no body
GET  /v2/health/ready
GET  /v2/models/{model}                                model metadata
GET  /v2/models/{model}/ready
POST /v2/models/{model}/infer
GET  /v2/models/{model}/versions/{version}
GET  /v2/models/{model}/versions/{version}/ready
POST /v2/models/{model}/versions/{version}/infer
```

The version in the path is a literal segment, not a parameter: `/v3/health/live`
is a 404 rather than an answer about a protocol version that does not exist.

An inference request, and its answer:

```bash
curl -s -X POST http://localhost:8000/v2/models/building_energy/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "req-1",
    "inputs": [{
      "name": "features",
      "shape": [2, 8],
      "datatype": "FP32",
      "data": [1800,30,-4,18,22,1,0,40,
                 55, 2,18,14,21,5,1,600]
    }]
  }'
```

```json
{
  "model_name": "building_energy",
  "model_version": "1",
  "id": "req-1",
  "outputs": [
    {"name": "kwh", "shape": [2, 1], "datatype": "FP32", "data": [2633.06, 16.72]}
  ]
}
```

`outputs` may be given in the request to ask for a subset. Errors are the
protocol's error object -- `{"error": "..."}` -- with the status the failure
deserves: 400 for a request the model cannot accept, 404 for a model or
version that does not exist, 503 when the queue is full, 504 on a queue
timeout.

### The catalogue endpoints

The protocol has one metadata call, for one version of one model. It answers
"what does this model take and return" and nothing answers "what can this
server serve" or "which versions of this model are there". Those two are what
you ask first, so they are served as extensions:

```text
GET /v2/models/list
GET /v2/models/info/{model}[?load=false]
```

```bash
curl -s http://localhost:8000/v2/models/list
```

```json
{
  "count": 10,
  "models": [
    {
      "name": "occupancy_mlp",
      "versions": ["1", "2"],
      "latest_version": "2",
      "platform": "onnx_onnxv1",
      "loaded_versions": ["2"],
      "ready": true
    }
  ]
}
```

Listing loads nothing, so a repository larger than the cache can be listed
without evicting a model that requests are using. `ready` says whether a
request naming no version would be served without a load.

```bash
curl -s http://localhost:8000/v2/models/info/occupancy_mlp
```

```json
{
  "name": "occupancy_mlp",
  "latest_version": "2",
  "versions": [
    {
      "version": "1",
      "platform": "onnx_onnxv1",
      "digest": "0f2c…",
      "size_bytes": 4135,
      "max_batch_size": 32,
      "loaded": true,
      "is_latest": false,
      "inputs":  [{"name": "features",  "datatype": "FP32", "shape": [-1, 8]}],
      "outputs": [{"name": "occupancy", "datatype": "FP32", "shape": [-1, 1]}]
    },
    {"version": "2", "is_latest": true, "…": "…"}
  ]
}
```

A signature comes from a loaded session, so `info` loads each version that is
not already in memory. `?load=false` skips that and returns the versions,
their settings and their digests with empty `inputs` and `outputs` -- worth it
against a server whose models are larger than its cache.

The digest is the artifact's SHA-256, which is what says two deployments are
serving the same bytes. It is reported by the RocksDB store, which records it
when ingesting; `--store filesystem` returns it empty rather than reading and
hashing every model to answer a listing.

Because a static path segment wins over a parameter, **`list` and `info` are
reserved model names**: a model called `list` would be unreachable at
`/v2/models/list`. Nothing else is shadowed, and the protocol's own URLs are
untouched.

Neither endpoint exists on the gRPC service, which has no catalogue method;
the Rust client says so rather than quietly listing over HTTP from a gRPC
client.

### gRPC

The same six calls, on port 8001, from the protocol's own
`grpc_predict_v2.proto`: `ServerLive`, `ServerReady`, `ModelReady`,
`ServerMetadata`, `ModelMetadata`, `ModelInfer`. There is no streaming, no
chat endpoint and no protocol-selection header; a previous version of this
server had all three, and they are gone.

Because gRPC is HTTP/2, every call is a stream on a shared connection, which
is why a gRPC run at concurrency 10 000 holds dozens of sockets rather than
ten thousand. A configured bearer token is enforced here too, with
`UNAUTHENTICATED` and `PERMISSION_DENIED` in place of 401 and 403; see
[Authentication](#authentication).

## Authentication

Off by default: every caller is served. It is a seam rather than a decision,
because the identity provider is a deployment's choice.

```bash
GALEMIND_AUTH_TOKEN=s3cret cargo run -p galemind -- serve

curl -H 'Authorization: Bearer s3cret' http://localhost:8000/v2/models/list
gmctl --token s3cret models
gmctl --transport grpc --token s3cret infer building_energy --rows 2
```

**One decision, both transports.** The trait lives in the domain crate rather
than in a transport, because both ports have to make the same decision about
the same token and an identity provider is not an HTTP detail:

```rust
#[async_trait]
pub trait Authenticator: Send + Sync {
    fn name(&self) -> &'static str;
    fn allows_anonymous(&self) -> bool { false }
    async fn authenticate(&self, token: Option<&str>) -> Result<Identity, AuthError>;
}
```

Two implementations ship: `NoAuth`, the default, which accepts everything as
`anonymous` and ignores a token rather than reporting an unverified one as an
identity; and `StaticTokenAuth`, a shared secret compared in constant time. A
provider that issues real identities implements the same trait and is passed
to `rest_server::app_with_auth` and `grpc_server::serve_with_auth`, which is
one line each in the binary.

What each transport adds is the part that is its own:

| | HTTP | gRPC |
|---|---|---|
| Where the check runs | an axum middleware | a tower layer in front of the router |
| Reads | the `Authorization` header | `authorization` metadata, the same header on the wire |
| A bad credential | 401 with `WWW-Authenticate: Bearer` | `UNAUTHENTICATED` |
| A credential that does not reach far enough | 403, and no challenge | `PERMISSION_DENIED` |
| The identity | request extensions | request extensions |

**Why a tower layer and not a tonic interceptor.** tonic's own seam,
`Interceptor`, is a *synchronous* `Request<()> -> Result<Request<()>, Status>`.
That is enough for a shared secret and not enough for anything that fetches: a
real provider has to await -- Keycloak's JWKS is an HTTP call, a cached key
still expires, an introspection endpoint is a round trip -- and doing that from
a sync interceptor means blocking a runtime thread per request, which is a way
to deadlock a server under load rather than a way to authenticate it. A layer's
future is ours, so an await is just an await. A refusal is a trailers-only
response built from a `tonic::Status`, which is what a gRPC client expects; a
bare HTTP 401 would reach it as a transport error with no status code, so the
one thing it needs to know -- that the credential, not the network, was the
problem -- would be the thing it could not see.

**Health and metrics are never authenticated**, on either transport:

| Open | HTTP | gRPC |
|---|---|---|
| Liveness | `/v2/health/live` | `ServerLive` |
| Readiness | `/v2/health/ready` | `ServerReady` |
| One model's readiness | `/v2/models/{model}/ready`, `/v2/models/{model}/versions/{version}/ready` | `ModelReady` |
| Metrics | `/metrics` | -- |

An orchestrator's probe has no token to send and Prometheus has no way to send
one, so requiring one would turn switching authentication on into a rolling
restart of a server that then reports itself unhealthy. Every readiness check
counts, including a model's own: a probe that waits for one named model is how
a deployment that preloads nothing avoids sending traffic into a cold cache.
What that tells an unauthenticated caller is whether a model it already knew
the name of is loaded -- a boolean, from a name it had to supply -- and not
what models exist, what they take or what they answer. Those are the catalogue
and metadata endpoints, which stay closed:

```text
no token    /v2/models/list                               401
bad token   /v2/models/list                               401
good token  /v2/models/list                               200
            /v2/health/live                               200
            /metrics                                      200
            /v2/models/building_energy/ready              200
            /v2/models/building_energy/versions/1/ready   200
no token    /v2/models/building_energy                    401
no token    grpc ModelInfer        this server requires an Authorization: Bearer token
good token  grpc ModelInfer        [0] 39.508022
no token    grpc ServerLive        server: ready
no token    grpc ModelReady        building_energy: ready
```

A deployment that considers even that too much keeps the port off the network;
one that considers the counters sensitive turns the scrape off with
`--allow-metrics false`.

**Keycloak** is the case this was shaped for. Such an implementation holds the
realm's issuer URL, the audience it accepts and a cache of the realm's JWKS;
`authenticate` verifies the token's signature against the cached key for its
`kid`, checks `iss`, `aud` and `exp`, and returns the `sub` with the realm
roles as scopes. Because `authenticate` is async, fetching a key it has not
cached is an await rather than a blocking call on a request thread. It needs a
JWT library and a key cache, which is why it is not written yet -- what exists
is the seam it plugs into, on both ports.

## gmctl

The client ships as a library and as a command line. It defaults to
`http://127.0.0.1:8000` over HTTP and `http://127.0.0.1:8001` over gRPC.

```bash
gmctl live                                   # exits non-zero when it is not
gmctl ready --model building_energy
gmctl info                                   # the server's name and version
gmctl models                                 # the catalogue
gmctl describe occupancy_mlp                 # every version, with metadata
gmctl describe stackgan_stage2 --no-load     # without loading anything
gmctl metadata building_energy
gmctl infer building_energy --row 1800,30,-4,18,22,1,0,40
gmctl infer stackgan_stage1 --rows 2         # generated rows fill every input
gmctl metrics --filter inference --watch 2
gmctl load building_energy -c 64 -n 10000
```

Global flags: `--endpoint`, `--transport http|http2|grpc`, `--json`,
`--token` (or `GALEMIND_TOKEN`, which keeps the secret out of the process
list). `http2` is the same REST port with requests multiplexed as HTTP/2
streams; see [Load testing](#load-testing).
Every command takes `--json`, which prints what a script or a timeseries store
wants rather than a table.

`--row` addresses a model with one input. `--rows N` generates rows for
**every** input the model declares, which is how a generator taking an
embedding and a noise vector is driven as easily as a feature matrix.

## Load testing

```bash
gmctl load building_energy --concurrency 700 --requests 20000 --rows 2 \
  --report reports/c700.json
```

```text
building_energy over http at concurrency 700, 2 rows per request
  20000 completed, 0 failed in 0.35s
  57144 requests/s, 114287 rows/s
  latency ms  min 0.74  p50 11.66  p55 12.11  p90 16.89  p95 18.77  p99 22.97  max 32.99
  at peak 57641 req/s: cpu 0%  memory 22 MiB  pending 352  mean batch 58.4
```

The JSON report carries the summary and one tick per metrics poll, each
pairing what the client measured with what the server reported about itself at
that moment. Client latency alone cannot tell a slow model from a saturated
queue; having both on one timeline can.

Options: `--concurrency`, `--requests` or `--duration`, `--rows`,
`--processes`, `--stagger-ms`, `--max-connections`, `--warmup`, `--seed`,
`--poll-ms`, `--report`, `--samples`.

### One socket per request, or many requests per socket

A request in flight needs a socket, and a socket is a file descriptor --
under HTTP/1.1. Under HTTP/2 a request is a *stream* and many of them share a
connection, so the socket count stops following the concurrency and the
descriptor limit stops being the ceiling. The same run at concurrency 10 000
against the same server, measured by peak sockets on the server side:

| `--transport` | Peak sockets | Throughput | p50 | p99 |
|---|---|---|---|---|
| `http` (HTTP/1.1) | 14 922 | 38 449 req/s | 207 ms | 547 ms |
| `http2` | **2** | 40 826 req/s | 233 ms | 270 ms |
| `grpc` | 40 | 48 684 req/s | 194 ms | 304 ms |

So at the usual soft limit of 1024 descriptors, where HTTP/1.1 cannot run at
all, the other two do:

```bash
ulimit -n 1024

gmctl --transport http  load building_energy -c 10000 -n 20000
# error: invalid request: a run at concurrency 10000 over a transport that opens
# a socket per request needs about 10064 open files, but this process may open
# only 1024: ... or multiplex it instead (`--transport http2` or
# `--transport grpc`), which carries the same run on a few sockets

gmctl --transport http2 load building_energy -c 10000 -n 20000
#   20000 completed, 0 failed in 0.47s   42490 requests/s
gmctl --transport grpc  load building_energy -c 10000 -n 20000
#   20000 completed, 0 failed in 0.41s   48986 requests/s
```

Cleartext HTTP/2 has nothing to negotiate with -- no ALPN without TLS -- so
the client states it and the server answers it; `curl --http2-prior-knowledge`
works against the same URLs.

Two details make that work. The pool is **primed before the burst**: a pool
with nothing in it, handed ten thousand requests at once, has no connection to
put them on and dials one for each until some are ready, which is a connection
storm that peaked at 2 443 sockets before the priming and 28 after. And the
socket count has a **ceiling of 256**, which `GALEMIND_MAX_CONNECTIONS` or
`--max-connections` overrides; measured over gRPC at concurrency 10 000 the
curve is flat enough that the ceiling is about tidiness rather than throughput
-- 157 channels gave 46 274 req/s, 40 channels 49 761, 8 channels 47 192.

### Ten thousand real sockets

Multiplexing answers "can the server take ten thousand requests at once". It
does not answer "can it take ten thousand *clients*", because ten thousand
clients are ten thousand sockets in separate address spaces, not streams
sharing a write loop. For that, use processes -- a descriptor limit is per
process, so ten processes holding a thousand sockets each need no privilege at
all where one process holding ten thousand needs a raised `ulimit`:

```bash
ulimit -n 1200
gmctl load building_energy --processes 10 -c 1000 --duration 12 --rows 2 \
  --stagger-ms 150
```

```text
driving building_energy from 10 processes at concurrency 1000 each: 10000 in flight

building_energy over http at concurrency 10000, 2 rows per request
  587188 completed, 0 failed in 12.27s
  47864 requests/s, 95728 rows/s
  latency ms  min 3.53  p50 216.72  p90 238.98  p99 264.02  max 319.60
  at peak 30306 req/s: cpu 1%  memory 230 MiB  pending 2438  mean batch 45.3
  10 processes, 10000 sockets
```

Ten thousand and twelve sockets on the server, nothing failed, nothing
dropped. Each child is this same binary with the same arguments and a file to
report into; the parent waits and adds up. Sums are sums -- requests, sockets,
failures -- while the rates are computed against the longest wall time, since
the processes overlapped, and the percentiles are recomputed from their pooled
samples, because percentiles cannot be averaged. Use `--duration` rather than
`--requests` with a stagger, or the processes that started first finish before
the last has begun and the aggregate describes a run that never happened.

### The accept queue

A listening socket has a queue of connections the kernel has completed the
handshake for and the server has not yet accepted. When it is full the kernel
drops further attempts -- silently, from the server's side, while the client
sees a connection that failed for no stated reason. This is what ten thousand
simultaneous connects hit first, and it is nowhere near where anyone looks.
The first run of the test above, against a server that had taken tokio's
default backlog of 1024:

```text
  7378 completed, 192622 failed in 0.62s
  failures    internal=192622
  at peak 2795 req/s: cpu 23%        <- the server was idle
```

```bash
nstat -az TcpExtListenOverflows
# TcpExtListenOverflows   139632
```

So the backlog is the server's to choose -- `--listen-backlog`, default 4096 --
and the kernel caps whatever is asked for at `net.core.somaxconn`:

```bash
sysctl -w net.core.somaxconn=16384
galemind serve --listen-backlog 16384
```

With the queue deep enough and the starts staggered, the same run dropped
nothing. `TcpExtListenOverflows` is the counter to watch: a run that fails for
no reason the client can name is usually this.

### The server's queue

`--queue-capacity` is per model and defaults to 1024, so 10 000 requests in
flight against a single model are mostly rejected as `overloaded` -- the right
answer, and not the one you want when measuring:

```bash
# 10k against one model: 87 792 of 100 000 rejected
gmctl load building_energy -c 10000 -n 100000

# the same run against a server sized for it
cargo run --release -p galemind -- serve --queue-capacity 16384 --max-batch-size 64
gmctl load building_energy -c 10000 -n 100000
#   99836 completed, 164 failed in 31.36s   (peak 54 704 req/s)
```

### In the order things break

1. **The client's shape.** A pool that cannot hold a connection per worker
   spends the run reconnecting; `gmctl load` sizes it from `--concurrency`.
   The warmup runs at the full concurrency for the same reason -- sent one at
   a time it established exactly one connection, and the measured phase then
   paid for the other 699 handshakes inside its own latencies. `--warmup 0`
   turns it off.
2. **The descriptor limit**, checked before the run with the command that
   raises it, or the suggestion to multiplex instead.
3. **The accept queue**: `--listen-backlog`, and `net.core.somaxconn` above it.
4. **The server's per-model queue**: `--queue-capacity`.
5. **The cores**: [Using the cores](#using-the-cores).

## Metrics

Prometheus text at `/metrics` on the REST port. Names follow NVIDIA Triton's
with `gm_` where Triton uses `nv_`, so dashboards and alerts port across with
a prefix change. Metrics that cannot mean anything here are left out rather
than reported as zero: no GPU metrics on a CPU server, no pinned memory
without a GPU, no response-cache metrics without a response cache.

**Build and platform**

```text
gm_server_info{version="0.2.0",platform="amd64",os="linux"} 1
gm_server_version{version="0.2.0"} 1
gm_server_platform{platform="amd64",os="linux"} 1
```

The version and the platform are published as series of their own as well as
labels on `gm_server_info`: a join works in a query but is awkward in an
alert and impossible in a Grafana variable, and those two are what a
deployment is grouped by. The architecture is named as container images name
it -- `amd64`, `arm64` -- not as Rust spells it.

**Per model**, labelled `model` and `version`

| Metric | |
|---|---|
| `gm_inference_request_success` | Requests answered |
| `gm_inference_request_failure` | Failures, by `reason`: `rejected`, `timeout`, `backend`, `invalid`, `other` |
| `gm_inference_count` | Rows inferred (a batch of n counts n) |
| `gm_inference_exec_count` | Batch executions; `count / exec_count` is the mean batch size |
| `gm_inference_pending_request_count` | Queued, not yet executing |
| `gm_inference_request_duration_us` | Cumulative end-to-end microseconds |
| `gm_inference_queue_duration_us` | Cumulative microseconds waiting in the queue |
| `gm_inference_compute_infer_duration_us` | Cumulative microseconds in the backend |
| `gm_inference_request_duration_seconds` | Histogram, twelve buckets from 100µs to 1s |
| `gm_batch_merged_count` | Executions that combined more than one request |

A failure that never resolved to a model is counted with empty labels.

**Cache**: `gm_model_cache_entries`, `gm_model_cache_resident_bytes`,
`gm_model_cache_hit_count`, `gm_model_cache_miss_count`,
`gm_model_cache_eviction_count`, `gm_model_cache_expiration_count`,
`gm_models_available`.

**Process and machine**

| Metric | |
|---|---|
| `gm_cpu_utilization` | Share of the machine used since the last sample, where 1.0 is every core |
| `gm_cpu_count` | Cores available |
| `gm_memory_usage_bytes` | Resident memory now |
| `gm_memory_peak_bytes` | The largest it has ever been |
| `gm_memory_total_bytes`, `gm_memory_utilization` | The machine's memory |
| `gm_server_uptime_seconds` | Seconds since the server started |
| `gm_server_worker_threads` | Async workers serving requests |
| `gm_server_blocking_threads` | Threads available for blocking work; bounds concurrent inferences |
| `gm_server_intra_op_threads` | Operator threads per model |
| `gm_server_loadable_models` | Versions that can be in memory at once; what operator threads multiply against |
| `gm_server_compute_threads_max` | Executions in flight × operator threads; above `gm_cpu_count` the machine is oversubscribed |
| `gm_server_cpu_mem_arena` | 1 when ONNX Runtime's arena allocator is on |
| `gm_malloc_arena_max` | `MALLOC_ARENA_MAX`, or 0 when unset |
| `gm_server_allocator` | `system` or `jemalloc`; whether an idle server returns memory |
| `gm_memory_high_water_bytes` | The mark above which the server reclaims, or 0 when off |
| `gm_memory_reclaim_count` | Times it has gone over and reclaimed |
| `gm_memory_reclaim_unload_count` | Times a reclaim was not enough and the models went too |

Utilisation is a rate, so it is sampled on `--metrics-interval-ms` rather than
computed during a scrape. Counters are cumulative and never reset.

## Using the cores

There are two thread pools and they multiply:

- **Async workers** (`--worker-threads`, default one per core) decode, queue
  and answer requests.
- **Blocking threads** (`--blocking-threads`, default twice the core count)
  run model code: a batch is handed to one of them, so this bounds how many
  inferences can execute at once. Tokio's own default is 512, which would let
  a burst across ten models start 512 inferences at once on a twelve-core
  machine; that is why it is bounded here.
- **Operator threads** (`--intra-op-threads`) are what ONNX Runtime may
  use *inside* one operator, **per loaded model**. Derived from the cores
  and the repository rather than fixed; see below.

The parallelism for a repository of models comes from serving them side by
side, which is why the derived operator-thread count falls to one as the
repository grows: ten models each taking a thread per core would be 120
threads fighting over 12. For a server with one model it is the other way
round, and the derivation says so.

A server with one large model and spare cores is the opposite case and wants
`--intra-op-threads` raised, or `intra_op_threads` set in that model's
`config.json`. `--parallel-execution` additionally runs independent branches
of a graph at once, which is worth it for a wide graph such as a large
ensemble and pointless for a deep chain.

### Operator threads are derived from the machine

`--intra-op-threads` is not a fixed default. Left unset, the server divides
the cores between the models that can be loaded at once:

```text
intra_op = clamp(cores / loadable_models, 1, cores / 2)
loadable_models = min(--cache-capacity, versions in the repository)
```

On a twelve-core machine that is **six** threads for a repository of one
model, three for four models, and **one** for the ten in this repository. On
a two-core machine it is one whatever is loaded. The plan and its input are
logged at startup and published as `gm_server_loadable_models`:

```text
INFO galemind: thread plan cpu_count=12 worker_threads=12 blocking_threads=24
     intra_op_threads=6 loadable_models=1 compute_threads_max=6
```

Why that shape, measured at both ends. A single heavy model wants the
threads, because its one execution is the only work there is:

| `--intra-op-threads` | Cores used | Throughput | p50 |
|---|---|---|---|
| 1 | 1.3 | 168 req/s | 23.4 ms |
| 6 (derived) | 2.7 | 289 req/s | 13.5 ms |

Six models under load at once want the opposite, because their executions
already fill the machine and operator threads only add contention:

```text
six models driven together, combined throughput
--intra-op-threads 1    26 337 requests/s
--intra-op-threads 3    23 747 requests/s
--intra-op-threads 6    24 045 requests/s
```

Hence dividing rather than choosing: the derived value is the first table's
answer for a server with one model and the second's for a server with ten.

The ceiling of half the machine is there because the other half is not idle
-- serving requests, encoding responses and answering scrapes took a quarter
to a third of this server's CPU time in every measurement above. And the
division rounds down, because an operator thread that has to share a core
with another session's costs more in contention than it returns.

Two values are still yours to set. An explicit `--intra-op-threads N`
overrides the derivation entirely, and `0` keeps its ONNX Runtime meaning:
let the runtime choose, which is one thread per core for *every* loaded
session -- 120 threads on a twelve-core machine with ten models. A single
model's `config.json` may also raise it for that model alone, which is how
one heavy model gets more without the rest of the repository taking it too.

### One model, several cores

The dispatcher pipelines: it forms the next batch while the current one is
executing, rather than waiting for the execution to finish and then opening a
batch window. The window is therefore paid in parallel with the work instead
of after it, which measured about 5% on a small CNN (13 180 against 12 350
requests a second, repeatable).

What it does not do is run two batches of the same model at once, and
`--max-concurrent-batches` defaults to 1 for a reason worth knowing:
**ONNX Runtime's `Run` is not thread-safe.** The `ort` crate makes
`Session::run` take `&mut self` precisely to prevent it, with a comment in
its source that is unusually direct on the subject, so a session sits behind
a mutex and a second execution of the same model only waits on it -- while
holding a blocking thread another model could have used. Measured, raising it
to four cost about a fifth of the throughput and nothing else.

So the parallelism for one model is **inside** one execution, which is
`--intra-op-threads`. On the 256x256 generator at concurrency 4, over gRPC so
that JSON encoding is not what is being measured:

| `--intra-op-threads` | Cores used | Throughput |
|---|---|---|
| 1 | 1.2 | 200 req/s |
| 2 | 2.0 | 319 req/s |
| 4 | 2.6 | 341 req/s |
| 8 | 3.3 | 354 req/s |

Cores used is the process' own CPU time over the wall time of the run, not a
sampled utilisation. The gain flattens because the graph has limited
parallelism and is partly memory-bandwidth bound, which is the normal shape
for convolutions on a CPU.

The rule that follows: **many models, one operator thread each; one model,
several.** A repository of ten small models wants the default, where the
parallelism comes from serving them side by side. A server with one heavy
model wants `--intra-op-threads` raised, or `intra_op_threads` set in that
model's `config.json` so only it pays.

`--max-concurrent-batches` stays because the serialisation belongs to the
backend rather than to the dispatcher: a backend that can run a model
concurrently -- a pool of sessions, or a runtime whose calls are thread-safe
-- makes it the knob that uses it, and executions are already spawned to
allow for it.

The plan is logged at startup and published in the metrics:

```text
INFO galemind: thread plan cpu_count=12 worker_threads=12 blocking_threads=24
     intra_op_threads=1 parallel_execution=false compute_threads_max=24
     cpu_mem_arena=false platform="amd64"
```

If `compute_threads_max` is more than twice `gm_cpu_count` the server warns:
throughput that stops climbing with concurrency is either a saturated machine
or an oversubscribed one, and those two numbers say which.

## Memory

A server that starts at 18 MB and sits at 200 MB after a few hundred requests
looks like a leak and usually is not one. Two caches are *meant* to grow --
the model cache holds loaded sessions, and an inference needs room for its
intermediates -- and underneath them two allocators retain what they have
freed. Measured on this repository, driving all ten models twice over:

| Configuration | start → pass 1 → pass 2 |
|---|---|
| `MALLOC_ARENA_MAX=2`, ONNX arena off (**the default**) | 17 → 71 → 69 MB |
| `MALLOC_ARENA_MAX=2`, ONNX arena on | 17 → 143 → 136 MB |
| unset, ONNX arena off | 17 → 172 → 196 MB |
| unset, ONNX arena on | 17 → 169 → 231 MB |

Two conclusions, and the second was the surprise:

**`MALLOC_ARENA_MAX` is the larger lever.** glibc keeps freed memory in
per-thread arenas and allows up to eight per core by default -- 96 on a
twelve-core host. Inference frees in large blocks from many threads, so those
arenas retain most of what a batch allocated, and resident memory settles two
to three times above what the work in flight needs. It has to be set in the
environment, because glibc reads it before `main` runs; the image and
`docker-compose.yml` set it, and a bare-metal deployment should too:

```bash
MALLOC_ARENA_MAX=2 galemind serve
```

**ONNX Runtime's CPU arena is off by default**, which is
`session_options.enable_cpu_mem_arena = False` in the Python API and
`--cpu-mem-arena` here. The arena speeds up repeated runs of the same shape by
never giving memory back: resident memory settles at the high-water mark of
the largest batch *each loaded session* has ever seen. With ten models and a
batch size that varies, that is 136 MB against 69. For a server with one model
under steady load, where the high-water mark is reached quickly and stays, it
is free throughput -- turn it on and measure.

Also true, and worth knowing before reaching for a profiler:

- **One session per model version, built once and reused.** Building a session
  parses and optimizes the graph and allocates its weights; building one per
  request would cost more than the inference and leave the arena to grow
  behind it. Sessions are built when the cache loads a model and dropped when
  it unloads one.
- **Memory pattern is off.** It reuses one allocation plan across runs, which
  assumes the shapes stay the same; batches here vary in size by design.
- **The working set of an inference is the dominant term, and it is set by
  rows per execution rather than by threads.** Measured below.

### Where the megabytes are

Loading is cheap and running is not. Loading all ten models, one at a time, on
a fresh server:

```text
after loading              rss MiB cache MiB rss delta cache delta
(nothing)                     16.5       0.0         -         -
building_energy               36.4       0.2     +19.9      +0.2
building_energy_class         37.6       0.4      +1.2      +0.2
energy_autoencoder            37.9       0.4      +0.3      +0.0
feature_scaler                38.0       0.4      +0.1      +0.0
load_forecast_tcn             38.0       0.5      +0.0      +0.1
meter_digit_cnn               38.2       0.6      +0.2      +0.1
occupancy_mlp                 38.2       0.6      +0.0      +0.0
stackgan_stage1               39.4       1.4      +1.2      +0.8
stackgan_stage2               41.9       1.5      +2.5      +0.1
thermal_cnn                   41.9       1.5      +0.0      +0.0
```

Three things to read off it:

1. **ONNX Runtime costs about 20 MB to start, once.** That is the whole of the
   first model's +19.9 MB -- its own weights are 0.2 MB. Nothing else pays it.
2. **Sessions are small.** Ten loaded models are 1.5 MB of weights and 25 MB of
   process, most of which is the runtime, not the models.
3. **So the climb to 170 MB is not the models being loaded.** It is the
   requests being answered.

Driving one model -- `stackgan_stage2`, which returns `3×256×256` floats, 0.75
MiB per row -- shows where it comes from:

| | resident | peak |
|---|---|---|
| idle, ten models loaded | 42 MB | 47 MB |
| concurrency 1, 1 row | 57 MB | 57 MB |
| concurrency 10, 1 row | 54 MB | 90 MB |
| concurrency 10, 4 rows | 173 MB | 237 MB |
| concurrency 32, 4 rows | 163 MB | 308 MB |

One model, nothing else running, and the whole 17-to-170 climb reproduced. A
graph that upsamples to 256×256 holds activations far larger than its output
while it runs: the intermediates of one execution, not the weights, are what
the allocator has to find room for, and the high-water mark of that is what
resident memory settles at.

#### What actually lowers it

Rows per execution. Not threads -- which surprised me, and is worth saying
plainly because the obvious guess is wrong. The same load, varying
`--blocking-threads`:

```text
blocking_threads=24   rss 120.0 MiB   peak 225.4 MiB
blocking_threads=8    rss 179.0 MiB   peak 222.1 MiB
blocking_threads=2    rss 125.1 MiB   peak 240.1 MiB
```

Twelve times fewer threads, same plateau. The reason is the dispatcher: there
is one per model and it runs one execution at a time, so concurrency against a
single model becomes a *bigger batch*, not more simultaneous inferences. The
thread count bounds how many *models* can be executing at once, which is the
term that matters for a repository of heavy models and not for one.

What does move it is the size of an execution, which is rows per request times
requests merged:

```text
--max-batch-size=1    4.0 rows per execution   peak 180.2 MiB
--max-batch-size=4    6.3 rows per execution   peak 222.8 MiB
--max-batch-size=16   5.3 rows per execution   peak 254.0 MiB
```

So a heavy model's plateau comes down by capping `max_batch_size` in its
`config.json` -- which costs throughput, because batching is what buys it -- or
by callers sending fewer rows per request. For a model whose output is
measured in kilobytes, none of this is worth thinking about.

### Giving memory back

Everything above explains why resident memory is what it is. The other
question is whether it ever comes down, and by default most of it does not.
Measured after a pass over all ten models, then standing idle:

```text
                    after load   idle +5s   +15s   +30s
glibc, models loaded     118.4      118.4   118.4  118.4
glibc, models unloaded    98.7       70.8    70.8   70.8
jemalloc, models loaded  248.7      231.6   114.5   95.8
```

Three mechanisms, and they do different things.

**Unloading a model** returns the session. `--cache-ttl-secs 60` unloads a
model nobody has asked for in a minute, and `--cache-sweep-secs` is how often
that is checked. Above, unloading all ten returned 28 MB -- more than the
1.5 MB of weights the cache accounts for, because a session is its graph, its
plan and its allocators too. What it does not return is the rest: with every
model unloaded and nothing in flight, glibc still held 70.8 MB against
16.2 MB at startup.

**Returning pages to the kernel** is the other 54 MB, and glibc will not do
it. Freed memory sits in per-thread arenas for the life of the process, and
the only way out is `malloc_trim`, which is an FFI call these crates cannot
make under `forbid(unsafe_code)`. So the answer is a different allocator:

```bash
cargo build --release --features jemalloc
_RJEM_MALLOC_CONF=background_thread:true,dirty_decay_ms:60000,muzzy_decay_ms:60000 \
  galemind serve
```

jemalloc returns pages that have gone unused for `dirty_decay_ms`, and the
background thread is what notices while the server is idle -- which is exactly
when nothing else would. With a sixty-second window:

```text
after load:    rss 213.6 MiB  peak 221.2 MiB
idle +15s:     rss 213.9 MiB
idle +30s:     rss 177.0 MiB
idle +45s:     rss 134.8 MiB
idle +60s:     rss 113.6 MiB
idle +75s:      rss 96.0 MiB
idle +105s:     rss 87.2 MiB
```

Models stayed loaded throughout, so the next request is answered as fast as
the last. It is a window and not a deadline: pages go back gradually over it.
The decay is read before `main` runs, like `MALLOC_ARENA_MAX`, so it is set in
the environment; the image and `docker-compose.yml` set both, and the image is
built with the feature. `gm_server_allocator` says which allocator a running
server has, because a memory graph cannot be read without knowing.

It is not a trade: on the same loads jemalloc was as fast or faster
(building_energy 54 548 against 46 875 requests a second, stackgan_stage1 280
against 254), at the cost of a higher transient peak -- it is less eager to
reuse a page and more willing to give it up.

**Capping the size of an execution** is the third, and it lowers the peak
rather than recovering from it: `--max-batch-size`, default 10, and a model's
own `max_batch_size`. See [What actually lowers it](#what-actually-lowers-it).

There is deliberately no "free everything now" call. Returning pages costs
page faults when the work comes back, which is why both allocators are built
to wait a while first, and a server that does it after every execution pays
that cost on every request.

### The high-water mark

Everything above is about why memory is what it is, and how it comes back on
its own. A deployment with a memory limit wants neither explanation: it wants
a number not to be exceeded. So there is one, `--memory-high-water-mb`,
default 500.

Above the mark the server reclaims, cheapest first:

1. **Asks the allocator for its retained pages.** Nothing is lost -- these
   are pages already freed -- and the models stay loaded, so the next request
   is answered as fast as the last.
2. **Unloads the models**, only if that was not enough. This does lose
   something: each model's next request pays its load again.

```bash
galemind serve --memory-high-water-mb 500 --memory-reclaim-cooldown-secs 30
```

Driving a 256x256 generator at concurrency 64 with sixteen rows a request,
against a mark of 500 MB:

```text
t= 4s  rss  424 MiB  reclaims 0  unloads 0
t=12s  rss  442 MiB  reclaims 0  unloads 0
t=14s  rss  503 MiB  reclaims 1  unloads 0
t=16s  rss  452 MiB  reclaims 1  unloads 0
t=20s  rss  307 MiB  reclaims 1  unloads 0
```

```text
INFO galemind_service::memory: reclaimed memory allocator="jemalloc"
     before_mb=503 after_mb=332 returned_mb=170 mark_mb=500
```

170 MB handed back at the mark, with every model still loaded and nothing
refused. The cooldown is what keeps a server whose steady state is near the
mark from purging on every sample and faulting the same pages straight back
in; `gm_memory_high_water_bytes`, `gm_memory_reclaim_count` and
`gm_memory_reclaim_unload_count` are the series to watch, and a rising unload
count means the mark is below this server's working set rather than that the
guard is doing its job.

Two honest limits. The guard samples every two seconds, so a spike that comes
and goes between samples is recorded by `gm_memory_peak_bytes` and not acted
on -- it is a limit on sustained memory, not a ceiling the kernel enforces.
And a purge can only return memory that is *free*: under sustained load
jemalloc sometimes returns nothing because the memory is in use, and the log
says so in those words rather than claiming success. The lever then is the
size of an execution, not the allocator.

Set `--memory-high-water-mb 0` to turn it off, which is right for a server
whose memory is already bounded by its cache and its batch size.

### How to tell a leak from retention

Drive the server in rounds and watch what the deltas do. Growth that shrinks
and turns negative is retention; growth that adds the same amount every round
is a leak. One small model, rounds of 100, 200 and 300 requests at concurrency
10:

```text
round    requests      rss MiB   peak MiB   delta MiB
0        0                16.8       16.8           -
1        100              16.8       16.8        +0.0
2        200              16.8       37.2        +0.0
3        300              37.2       37.2       +20.4
4..14    100/200/300      37.2       37.2        +0.0   (every round)
total growth after 2 700 further requests: +0.0 MiB
```

It reached a high-water mark and stopped dead. The whole repository -- each
round driving all ten models -- reaches a higher one and still stops:

```text
round    req x model    rss MiB   peak MiB   delta MiB
0        0                 17.0       17.0          -
1        100              114.0      154.8      +97.0
2        200              140.5      167.5      +26.5
3        300              146.4      172.8       +5.9
4        100              178.1      189.2      +31.7
5        200              151.5      210.1      -26.6
6        300              162.1      210.1      +10.6

# four more identical passes, 12 000 further requests
1        300              148.3      221.1      -27.0
2        300              159.5      221.1      +11.2
3        300              170.7      221.1      +11.2
4        300              170.7      221.1       +0.0
total growth: -4.6 MiB
```

Twelve thousand requests later it is 4.6 MB *lower* than it started, with the
peak pinned at 221 MB and not moving. A leak never gives memory back.

```bash
# 1. Is it bounded? Drive one model in rounds and watch the plateau.
for i in 1 2 3 4 5; do
  gmctl load building_energy -n 2000 -c 32 --rows 8 --poll-ms 0 > /dev/null
  curl -s localhost:8000/metrics | grep -E '^gm_memory_(usage|peak)_bytes'
done
```

Flat across rounds is retention, and a small model is flat: 12 000 requests
against `building_energy` moves resident memory from 17.1 MB to 17.3 MB and
stops. Still climbing in the fifth round, on a model whose batch size and
concurrency are not changing, is a leak worth chasing.

```bash
# 2. Is it the model cache? Compare the resident bytes it admits to
#    with the process' own.
curl -s localhost:8000/metrics | grep -E 'gm_model_cache_(entries|resident_bytes)'
```

```bash
# 3. Is it eviction churn? A capacity below the repository's size reloads
#    models continuously, and each reload is a fresh session.
curl -s localhost:8000/metrics | grep -E 'gm_model_cache_(miss|eviction)_count'
```

```bash
# 4. Does it come back? Unload on a short TTL and watch.
galemind serve --cache-ttl-secs 30 --cache-sweep-secs 5
```

`gm_memory_peak_bytes` beside `gm_memory_usage_bytes` is the quickest read of
all: the kernel never lowers the peak, so a process that peaked at 200 MB and
sits at 70 MB gave its memory back, while one whose current reading tracks its
peak upwards did not.

## Following a request

Every request gets an id and a span, and everything logged while handling it
is nested inside that span. The id comes from `X-Request-Id` when the caller
sends one -- so a trace can be followed from a gateway's log into this one --
and is echoed on the response either way, so a caller that sent none can still
quote the id of a request that went wrong.

```bash
GALEMIND_LOG=galemind=debug,rest_server=debug,galemind_service=debug \
  cargo run --release -p galemind -- serve

curl -H 'X-Request-Id: demo-42' -X POST \
  http://localhost:8000/v2/models/thermal_cnn/infer -d @request.json
```

```text
DEBUG request{id=demo-42 method=POST path=/v2/models/thermal_cnn/infer}:
      rest_server::trace: request received
DEBUG request{id=demo-42 …}: galemind_service::service: request received
      request=infer-1 model=thermal_cnn inputs=1 rows=1
 INFO request{id=demo-42 …}: galemind_cache::lru: loaded model into cache
      model=thermal_cnn/1 entries=1
 INFO request{id=demo-42 …}: galemind_service::service: starting batching
      dispatcher model=thermal_cnn/1
DEBUG request{id=demo-42 …}: galemind_service::service: request answered
      request=infer-1 model=thermal_cnn/1 rows=1 elapsed_us=18528
DEBUG request{id=demo-42 … status=200}: rest_server::trace: request answered
      elapsed_us=18739
```

Reading that: 18.7 ms end to end, of which 18.5 ms was the model load this
request happened to trigger. `--preload` or a warmup takes that out. At
`trace` the service also logs version resolution and validation separately,
and the dispatcher logs each batch it forms.

At the default `info` filter a healthy server says nothing per request -- one
answering 50 000 requests a second cannot log two lines each -- so the flow is
off until asked for. A 5xx is logged as a warning whatever the filter.

gRPC calls are traced the same way, under a `grpc` span, taking the id from
`x-request-id` metadata or from the request's own `id` field.

## Docker

```bash
docker compose build server
docker compose up server
```

The image is multi-stage, runs as a non-root user, carries `gmctl` so the
container can check its own health, and sets `MALLOC_ARENA_MAX=2`. The
repository is baked in at `/app/models`; mount over it to serve your own.
`docker-compose.yml` also defines a `load` service that runs `gmctl load` against
the server and writes its report to `./reports`:

```bash
LOAD_MODEL=stackgan_stage1 LOAD_CONCURRENCY=32 docker compose run --rm load
```

## Project layout

A workspace of small crates, each depending on behaviour rather than on a
runtime, a framework or a storage engine.

| Crate | |
|---|---|
| `src/core` | The domain: datatypes, tensors, model identity, requests, the JSON encoding, the authentication trait, and the traits every replaceable part implements. No I/O. |
| `src/onnx` | The ONNX Runtime backend. The only crate that knows ONNX Runtime exists. |
| `src/store` | Model repositories: a directory tree, RocksDB, memory. |
| `src/cache` | The bounded, time-to-live model cache with single-flight loading. |
| `src/batching` | The per-model dispatcher and the adaptive batch policy. |
| `src/service` | Resolution, caching and batching behind one trait; metrics; the listening socket. |
| `src/rest_server` | The HTTP transport, the tracing and auth middleware. |
| `src/grpc_server` | The gRPC transport and its authentication layer. |
| `src/galemind` | The server binary and its command line. |
| `src/client/rust` | The client library, the load generator, and `gmctl`. |
| `src/client/python` | The Python client. |
| `src/e2e` | The end-to-end suite, in its own workspace. |
| `scripts/` | Model generators, the RocksDB staging script. |
| `website/` | The documentation site. |

## The documentation site

Everything above, and more room to explain it, as a [Docusaurus](https://docusaurus.io/)
site in `website/`:

```bash
cd website
npm ci
npm start          # live reload on http://localhost:3000
npm run build      # the static site, into website/build
```

Node 20 or newer. The sidebar is written out in `sidebars.ts` rather than
generated, because the order to read the pages in is not the order a directory
listing happens to be in, and `onBrokenLinks` is `throw`, so a link to a page
that does not exist fails the build instead of shipping.

## License

Apache 2.0. See [LICENSE](LICENSE).
