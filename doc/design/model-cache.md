# The model cache

> **The problem.** A loaded model is a live session that cannot be serialized
> and costs tens of milliseconds to build, so it has to be reused across
> requests. Keeping every model loaded is unbounded memory; and two requests
> for the same cold model would each build it, at twice the memory while both
> are in flight.
>
> **How it is solved.** A bounded, time-to-live cache of `Arc<dyn Backend>`
> with a per-key gate, so concurrent misses load once, and a re-check after the
> gate so the waiter does not reload what it just waited for. `loaded_keys`
> sits beside `get` so that reporting what is loaded never loads anything.

A loaded model is a live runtime session: a parsed, optimized graph with its
weights allocated. It cannot be serialized, so the cache that matters at
request time is in memory. That makes the repository a two-tier affair.

```mermaid
graph TB
    REQ(["infer / metadata"]) --> HOT
    subgraph hot["Hot tier — LruTtlCache, in memory"]
        HOT["bounded map of<br/>Arc&lt;dyn Backend&gt;"]
    end
    subgraph cold["Cold tier — ModelStore"]
        ROCKS[("RocksDB<br/>artifacts · metadata · usage")]
        FS[("Directory tree")]
    end
    HOT -->|"miss: fetch bytes"| ROCKS
    FS -->|"ingest at startup"| ROCKS
    HOT -->|"load via BackendFactory"| SESS["Session, built once"]
    HOT -->|"touch: last used, count, size"| ROCKS

    classDef hotc fill:#e8f0f8,stroke:#2a6fa8,stroke-width:2px,color:#1d4d75;
    classDef unit fill:#ffffff,stroke:#8aa4bb,color:#20303d;
    classDef store fill:#f6f7f9,stroke:#b8c3cf,color:#20303d;
    class HOT,SESS hotc;
    class REQ unit;
    class ROCKS,FS store;
```

## The types

```mermaid
classDiagram
    class ModelCache {
        <<trait>>
        +get(key) async Result~Backend~
        +invalidate(key) async
        +sweep() async usize
        +loaded_keys() Vec~ModelKey~
        +stats() CacheStats
    }
    class LruTtlCache {
        -store: Arc~ModelStore~
        -registry: Arc~BackendRegistry~
        -usage: Option~UsageIndex~
        -clock: Arc~Clock~
        -entries: Mutex~HashMap~
        -gates: Mutex~HashMap~
        -counters: Counters
    }
    class LruTtlConfig {
        +capacity: usize
        +ttl: Duration
    }
    class CacheStats {
        +entries: usize
        +hits: u64
        +misses: u64
        +evictions: u64
        +expirations: u64
        +resident_bytes: usize
    }
    class Clock {
        <<trait>>
        +now_millis() u64
        +unix_millis() u64
    }

    ModelCache <|.. LruTtlCache
    LruTtlCache *-- LruTtlConfig
    LruTtlCache o-- Clock
    ModelCache ..> CacheStats : stats
```

`gates` is the single-flight machinery below, kept in its own map rather than
inside the entries: a gate has to exist for a key that is *not* in the cache
yet, which is precisely the case it guards. And `Clock` is injected rather
than read from the system, so the time-to-live can be tested without sleeping.

## How it is bounded

Two bounds, for two different problems:

| Bound | Flag | Default | Problem it solves |
|---|---|---|---|
| Capacity | `--cache-capacity` | 8 | Too many models loaded at once; the least recently used is dropped to make room |
| Time-to-live | `--cache-ttl-secs` | 3600 | A model nobody is asking for any more, holding memory |
| Sweep interval | `--cache-sweep-secs` | 60 | How often the time-to-live is actually enforced |

Eviction and expiration are counted separately in `CacheStats`, because they
mean different things about a deployment: evictions say the capacity is too
small for the working set, expirations say models are going quiet.

## Single-flight loading

Two requests for the same cold model must not each build a session. Building
one parses and optimizes the graph and allocates its weights — tens of
milliseconds for a heavy model, and twice the memory while both are in flight.

```mermaid
sequenceDiagram
    autonumber
    participant A as Request A
    participant B as Request B
    participant K as LruTtlCache
    participant G as Per-key gate
    participant S as Store + factory

    A->>K: get(model/1)
    B->>K: get(model/1)
    K-->>A: miss
    K-->>B: miss
    A->>G: take the gate for model/1
    B->>G: wait on the same gate
    A->>S: fetch bytes, build the session
    S-->>A: Arc<dyn Backend>
    A->>K: insert, evicting the LRU entry if full
    A->>G: release
    G-->>B: acquired
    B->>K: re-check under the gate
    K-->>B: hit — the same Arc, nothing built twice
```

The re-check after acquiring the gate is the whole trick: without it, the
second waiter would load the model it just waited for someone else to load.

## Listing must not load

`ModelCache::loaded_keys` exists alongside `get` for one reason. Reporting
whether a model is loaded must not load it — otherwise listing a repository
larger than the cache would evict the models that requests are actively using,
and a dashboard refresh would be a load-test.

The same reasoning runs up through `Inference::catalogue`, which loads
nothing, and `Inference::model_info`, where loading is an explicit opt-in
because a signature can only be reported by a loaded model.

## Surviving a restart

An in-memory cache starts cold, and starting cold is unavoidable. Starting
*blind* is not: the optional `UsageIndex` — implemented by RocksDB's `usage`
column family — records last-used time, load count and resident size for every
model, so a restarted server knows which models were hot and can order its
evictions sensibly rather than by accident of arrival.

## Who drives it

```mermaid
flowchart LR
    SW["Cache sweeper task<br/>every --cache-sweep-secs"] -->|"sweep()"| C["LruTtlCache"]
    MG["Memory guard<br/>over the high-water mark"] -->|"unload_all → invalidate()"| C
    RQ["A request for a cold model"] -->|"get()"| C
    AD["Admin: a changed artifact"] -->|"invalidate()"| C

    classDef unit fill:#ffffff,stroke:#8aa4bb,color:#20303d;
    classDef hotc fill:#e8f0f8,stroke:#2a6fa8,stroke-width:2px,color:#1d4d75;
    class C hotc;
    class SW,MG,RQ,AD unit;
```

The memory guard's `unload_all` is not an eviction policy — the capacity and
the time-to-live are that — but the last thing a server over a memory limit
can give up. It leaves the dispatchers running: they are cheap, and a reloaded
model finds its queue where it left it. See [memory.md](memory.md).

## Described by

`src/cache/src/lru.rs`, `src/cache/src/lib.rs`,
`src/core/src/traits.rs` (`ModelCache`, `UsageIndex`, `CacheStats`),
`src/service/src/service.rs`.
