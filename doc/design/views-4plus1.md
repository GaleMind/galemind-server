# The 4+1 views

Five diagrams, in Kruchten's arrangement: four views of the same server, and
the scenarios that tie them together. Each answers a different question, and
none of them is the architecture by itself.

| View | The question it answers |
|---|---|
| [Logical](#logical-view--what-the-pieces-are) | What are the pieces, and which abstractions do they meet at? |
| [Process](#process-view--what-runs-at-once) | What runs at once, and what bounds it? |
| [Development](#development-view--the-workspace) | Which crate may depend on which? |
| [Physical](#physical-view--where-it-runs) | Where does it run, and who talks to it? |
| [Scenarios](#scenarios--the-1) | Does the whole thing actually work, end to end? |

The same five diagrams are kept as standalone Mermaid sources in
[`website/diagrams/`](../../website/diagrams) and rendered to SVG for the
documentation site, so a change to the architecture is a change to a text file
and a re-render rather than a redrawing:

```bash
cd website
npx @mermaid-js/mermaid-cli -i diagrams/logical-view.mmd \
  -o static/img/views/logical-view.svg -b transparent
```

## Logical view — what the pieces are

The arrows all point inwards, at traits in a crate that does no I/O. That is
the whole structural claim of this server: a transport depends on `Inference`,
not on a service; the cache depends on `Backend`, not on ONNX Runtime; and
every box outside the domain is replaceable without touching the ones beside
it.

```mermaid
%% The logical view: what the pieces are, and which abstractions they meet
%% at. Every arrow into the domain is a trait, so the arrows point inwards.
graph TB
    subgraph transports["Transports"]
        direction LR
        REST["rest_server<br/>axum"]
        GRPC["grpc_server<br/>tonic"]
    end

    subgraph domain["galemind_core — the domain, no I/O"]
        subgraph served["what a transport asks for"]
            direction LR
            INFER(["Inference"])
            AUTH(["Authenticator"])
            METRICS(["MetricsSource"])
        end
        subgraph replaceable["what an adapter provides"]
            direction LR
            CACHET(["ModelCache"])
            STORE(["ModelStore"])
            BACKEND(["Backend"])
        end
    end

    subgraph orchestration["Orchestration"]
        direction LR
        SVC["galemind_service"]
        BATCH["galemind_batching"]
        CACHE["galemind_cache"]
    end

    subgraph adapters["Adapters"]
        direction LR
        ONNX["galemind_onnx"]
        FS["FsModelStore"]
        ROCKS["RocksModelStore"]
    end

    REST --> INFER
    GRPC --> INFER
    REST --> AUTH
    GRPC --> AUTH
    REST --> METRICS
    SVC -.implements.-> INFER
    SVC -.implements.-> METRICS
    SVC --> BATCH
    SVC --> CACHE
    BATCH --> CACHET
    CACHE -.implements.-> CACHET
    CACHE --> BACKEND
    CACHE --> STORE
    ONNX -.implements.-> BACKEND
    FS -.implements.-> STORE
    ROCKS -.implements.-> STORE

    classDef iface fill:#e8f0f8,stroke:#2a6fa8,stroke-width:2px,color:#1d4d75;
    classDef unit fill:#ffffff,stroke:#8aa4bb,color:#20303d;
    class INFER,AUTH,BACKEND,STORE,CACHET,METRICS iface;
    class REST,GRPC,SVC,BATCH,CACHE,ONNX,FS,ROCKS unit;
```

## Process view — what runs at once

The same picture in motion, and the one that explains the flags. Requests are
decoded on async workers, wait in a per-model queue, are merged by that
model's dispatcher, and execute on a blocking thread inside a session that
runs one batch at a time. Three background tasks watch the rest: the cache
sweeper, the resource sampler and the memory guard.

```mermaid
%% The process view: the threads and tasks a running server has, and what
%% each may do at once. The numbers are the flags that bound them.
graph TB
    CLIENT(["Clients"]) --> ACCEPT["Accept queue<br/>--listen-backlog 4096"]

    subgraph runtime["One process, one tokio runtime"]
        ACCEPT --> WORKERS["Async workers — one per core<br/>decode, validate, encode, reply"]
        WORKERS --> QUEUE["Per-model queue — 1024<br/>full ⇒ 503 overloaded"]
        QUEUE --> DISP["Dispatcher per model<br/>forms the next batch while<br/>the current one executes"]
        DISP -->|spawn_blocking| POOL["Blocking pool — 2 × cores"]
        POOL --> SESSION["Session per model version<br/>one Run at a time, behind a mutex<br/>--intra-op-threads inside it"]
        SESSION -->|rows| WORKERS

        SWEEP["Cache sweeper<br/>every --cache-sweep-secs"]
        MGUARD["Memory guard<br/>every 2s"]
        SAMPLER["Resource sampler<br/>every --metrics-interval-ms"]
    end

    SWEEP -.unloads idle models.-> SESSION
    MGUARD -.purges, then unloads.-> SESSION
    SAMPLER -.reads /proc.-> MGUARD

    classDef task fill:#ffffff,stroke:#8aa4bb,color:#20303d;
    classDef hot fill:#e8f0f8,stroke:#2a6fa8,stroke-width:2px,color:#1d4d75;
    class WORKERS,DISP,POOL,SESSION hot;
    class QUEUE,SWEEP,SAMPLER,MGUARD,ACCEPT,CLIENT task;
```

Details, and the measurements behind the numbers, in
[concurrency.md](concurrency.md).

## Development view — the workspace

Which crate may depend on which. `src/core` has no dependency on anything in
the workspace, which is what makes it testable without a runtime; `src/e2e`
is outside the workspace entirely, for the feature-unification reason in
[overview.md](overview.md#why-the-end-to-end-suite-is-a-separate-workspace).

```mermaid
%% The development view: the workspace, and which crate may depend on which.
%% Everything points inwards, so the domain depends on nothing.
graph TB
    BIN["src/galemind<br/>the binary: CLI, runtime, allocator"]
    REST["src/rest_server"]
    GRPC["src/grpc_server"]
    SVC["src/service"]
    BATCH["src/batching"]
    CACHE["src/cache"]
    STORE["src/store"]
    ONNX["src/onnx"]
    CORE["src/core<br/>the domain, no I/O"]
    CLIENT["src/client/rust<br/>library + gmctl"]
    PY["src/client/python"]
    E2E["src/e2e<br/>its own workspace"]

    BIN --> REST
    BIN --> GRPC
    BIN --> SVC
    BIN --> ONNX
    BIN --> STORE
    REST --> SVC
    GRPC --> SVC
    SVC --> BATCH
    SVC --> CACHE
    SVC --> STORE
    BATCH --> CORE
    CACHE --> CORE
    STORE --> CORE
    ONNX --> CORE
    REST --> CORE
    GRPC --> CORE
    SVC --> CORE
    CLIENT --> CORE
    CLIENT --> GRPC
    E2E --> CLIENT

    classDef domain fill:#e8f0f8,stroke:#2a6fa8,stroke-width:2px,color:#1d4d75;
    classDef crate fill:#ffffff,stroke:#8aa4bb,color:#20303d;
    classDef outside fill:#f6f7f9,stroke:#b8c3cf,stroke-dasharray:4 3,color:#20303d;
    class CORE domain;
    class BIN,REST,GRPC,SVC,BATCH,CACHE,STORE,ONNX,CLIENT crate;
    class PY,E2E outside;
```

## Physical view — where it runs

One container, one process, two ports. Everything an operator needs on the
HTTP port — the protocol, the catalogue, health and `/metrics` — and the
protocol again on the gRPC port. Health and metrics are reachable without a
credential by design, so probes and scrapes survive turning authentication
on; see [authentication.md](authentication.md).

```mermaid
%% The physical view: one container, two ports, and who talks to them.
graph TB
    subgraph clients["Callers"]
        APP["Applications<br/>Rust / Python clients"]
        CLI["gmctl"]
        PROM["Prometheus"]
        ORCH["Orchestrator probes"]
    end

    subgraph host["Host or node"]
        subgraph container["Container — non-root, MALLOC_ARENA_MAX=2"]
            P8000["8000/tcp — HTTP/1.1 and HTTP/2<br/>/v2/... and /metrics"]
            P8001["8001/tcp — gRPC"]
            SERVER["galemind serve"]
            MODELS[("/app/models<br/>baked in, mountable")]
            DATA[("/app/data<br/>RocksDB volume")]
        end
    end

    APP -->|"Predict Protocol v2"| P8000
    APP -->|"Predict Protocol v2"| P8001
    CLI --> P8000
    PROM -->|"scrape /metrics, no token"| P8000
    ORCH -->|"health, no token"| P8000
    P8000 --> SERVER
    P8001 --> SERVER
    SERVER --> MODELS
    SERVER --> DATA

    classDef port fill:#e8f0f8,stroke:#2a6fa8,stroke-width:2px,color:#1d4d75;
    classDef unit fill:#ffffff,stroke:#8aa4bb,color:#20303d;
    classDef store fill:#f6f7f9,stroke:#b8c3cf,color:#20303d;
    class P8000,P8001 port;
    class SERVER,APP,CLI,PROM,ORCH unit;
    class MODELS,DATA store;
```

## Scenarios — the +1

Two walks through the whole stack, which is what the four views are for.

### One inference

Including every way it can end early: an open endpoint, a refused credential,
a full queue, a timed-out wait.

```mermaid
%% The +1: one inference, from the socket to the answer, with the decisions
%% that can end it early.
sequenceDiagram
    autonumber
    participant C as Caller
    participant T as trace middleware
    participant A as Authenticator
    participant S as InferenceService
    participant K as Model cache
    participant D as Dispatcher
    participant B as ONNX session

    C->>T: POST /v2/models/thermal_cnn/infer
    T->>T: id from X-Request-Id, or minted
    T->>A: Authorization header
    alt health or metrics
        A-->>T: open, no credential needed
    else bad or missing token
        A-->>C: 401 + WWW-Authenticate
    end
    T->>S: InferRequest, inside the request span
    S->>S: resolve version, validate against metadata
    S->>K: get(thermal_cnn/1)
    alt not loaded
        K->>B: build session once, reuse after
    end
    S->>D: submit
    alt queue full
        D-->>C: 503 overloaded
    else waited past the timeout
        D-->>C: 504
    end
    D->>D: merge requests up to --max-batch-size
    D->>B: run on a blocking thread
    B-->>D: rows
    D->>D: split rows back per caller
    D-->>S: this caller's rows only
    S-->>T: InferResponse
    T-->>C: 200, X-Request-Id echoed
```

Narrated step by step in [inference-path.md](inference-path.md).

### Resident memory crosses the mark

The memory guard touches the allocator, the cache and the metrics, and is the
clearest example of a decision split between a component that can measure and
a caller that can act.

```mermaid
%% The +1, second scenario: what happens when resident memory crosses the
%% high-water mark.
sequenceDiagram
    autonumber
    participant G as Memory guard
    participant P as /proc/self/status
    participant AL as Allocator
    participant SV as InferenceService
    participant K as Model cache

    loop every 2 seconds
        G->>P: resident bytes
    end
    G->>G: over --memory-high-water-mb?
    alt below the mark
        G-->>G: nothing to do
    else within the cooldown
        G-->>G: a reclaim ran recently
    else over the mark
        G->>AL: return retained pages
        alt jemalloc
            AL-->>G: purged, pages back to the kernel
        else platform allocator
            AL-->>G: unsupported, log how to fix it
        end
        G->>P: resident bytes again
        alt now under the mark
            G-->>G: done, models still loaded
        else still over
            G->>SV: unload_all
            SV->>K: invalidate every loaded version
            K-->>SV: dropped sessions
            SV-->>G: count, recorded for the metrics
        end
    end
```

Narrated in [memory.md](memory.md).

## Described by

[`website/diagrams/*.mmd`](../../website/diagrams),
[`website/docs/architecture.mdx`](../../website/docs/architecture.mdx).
