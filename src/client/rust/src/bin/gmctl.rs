//! `gmctl`: a command-line client for a Predict Protocol v2 server.

#![forbid(unsafe_code)]

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use galemind_client::{
    connect_as, connect_with, parse_row, rows_to_tensor, run_load, scrape, sole_input,
    synthesize_inputs, ClientConfig, LoadConfig, PredictV2Client, Scrape, Transport,
};
use galemind_core::{InferRequest, ModelMetadata, ModelRef, Parameters, RequestedOutput};
use serde_json::json;
use std::process::ExitCode;
use std::sync::Arc;

/// Talk to a GaleMind inference server.
#[derive(Debug, Parser)]
#[command(name = "gmctl", version, about, long_about = None)]
struct Cli {
    /// Server address. Defaults to the local server's port for the chosen
    /// transport: 8000 for HTTP, 8001 for gRPC.
    #[arg(long, short, global = true, env = "GALEMIND_ENDPOINT")]
    endpoint: Option<String>,

    /// Which transport to speak.
    #[arg(
        long,
        short,
        global = true,
        env = "GALEMIND_TRANSPORT",
        default_value = "http"
    )]
    transport: TransportArg,

    /// Print machine-readable JSON instead of a table.
    #[arg(long, global = true)]
    json: bool,

    /// Bearer token for a server that requires one, on either transport.
    ///
    /// Prefer the environment variable: an argument is visible in the process
    /// list to every user on the machine.
    #[arg(long, global = true, env = "GALEMIND_TOKEN", value_name = "TOKEN")]
    token: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum TransportArg {
    /// HTTP/1.1: a connection per request in flight.
    Http,
    /// HTTP/2 on the same port: requests are streams, so a run at any
    /// concurrency holds a handful of sockets.
    Http2,
    /// gRPC, which is HTTP/2 and multiplexed by construction.
    Grpc,
}

impl Cli {
    /// The transport as the child's command line spells it.
    fn transport_name(&self) -> &'static str {
        match self.transport {
            TransportArg::Http => "http",
            TransportArg::Http2 => "http2",
            TransportArg::Grpc => "grpc",
        }
    }
}

impl From<TransportArg> for Transport {
    fn from(value: TransportArg) -> Self {
        match value {
            TransportArg::Http => Transport::Http,
            TransportArg::Http2 => Transport::Http2,
            TransportArg::Grpc => Transport::Grpc,
        }
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Ask whether the server is live. Exits non-zero when it is not.
    Live,
    /// Ask whether the server, or one model, is ready.
    Ready(ReadyArgs),
    /// Show the server's name, version and extensions.
    Info,
    /// List every model the server can serve.
    Models,
    /// Show every version of one model, with its metadata.
    Describe(DescribeArgs),
    /// Show a model's inputs and outputs.
    Metadata(ModelArgs),
    /// Run an inference.
    Infer(InferArgs),
    /// Read the server's Prometheus metrics.
    Metrics(MetricsArgs),
    /// Drive load at the server and report what happened.
    Load(LoadArgs),
}

#[derive(Debug, Clone, Args)]
struct LoadArgs {
    /// Model to drive.
    #[arg(default_value = "building_energy")]
    model: String,

    /// Model version; the server chooses when omitted.
    #[arg(long, short = 'V')]
    version: Option<String>,

    /// Requests in flight at once. Ten thousand is the classic C10K figure.
    #[arg(long, short, default_value_t = 64)]
    concurrency: usize,

    /// Run the load from this many operating-system processes.
    ///
    /// Each process runs the whole run at `--concurrency`, so ten processes
    /// at a thousand is ten thousand requests in flight over ten thousand
    /// sockets -- and, under HTTP/1.1, ten thousand real connections rather
    /// than ten thousand streams on a few. Each process gets its own
    /// open-file limit, which is what makes that possible at all: one
    /// process cannot hold ten thousand sockets under the usual limit of
    /// 1024, and ten holding a thousand each can.
    ///
    /// The children's reports are merged into one, with the percentiles
    /// recomputed from their pooled samples.
    #[arg(long, default_value_t = 1, value_name = "N")]
    processes: usize,

    /// Milliseconds between starting one process and the next.
    ///
    /// Each process opens its connections as it starts, so starting ten at
    /// once asks the server's kernel to complete ten thousand handshakes at
    /// the same instant. The accept queue is capped at
    /// `net.core.somaxconn` -- 4096 on a current kernel -- and what does not
    /// fit is dropped, which arrives at the client as a connection that
    /// failed for no stated reason. Staggering the starts keeps each burst to
    /// one process' worth while the connections, once open, stay open.
    ///
    /// Use `--duration` rather than `--requests` with a stagger, so that a
    /// short run does not finish in the processes that started first before
    /// the last one has begun.
    #[arg(long, default_value_t = 100, value_name = "MS")]
    stagger_ms: u64,

    /// Keep every latency in the JSON report, not just the percentiles.
    ///
    /// Implied by `--processes`, which needs them to merge exactly.
    #[arg(long)]
    samples: bool,

    /// Internal: write this child's report here and print nothing.
    #[arg(long, hide = true, value_name = "FILE")]
    child_report: Option<std::path::PathBuf>,

    /// The most sockets a multiplexed run may open. Defaults to 256.
    ///
    /// A ceiling on a count that otherwise follows the concurrency: on
    /// `--transport http2` and `--transport grpc` the requests are streams,
    /// so this is how many connections they are spread over. It does nothing
    /// under `--transport http`, where a request owns its connection until it
    /// answers.
    #[arg(
        long,
        env = "GALEMIND_MAX_CONNECTIONS",
        value_name = "N",
        default_value_t = galemind_client::limits::DEFAULT_MAX_CONNECTIONS,
    )]
    max_connections: usize,

    /// Stop after this many requests.
    #[arg(
        long,
        short = 'n',
        default_value_t = 10_000,
        conflicts_with = "duration"
    )]
    requests: usize,

    /// Stop after this many seconds instead of after a request count.
    #[arg(long, short)]
    duration: Option<f64>,

    /// Rows per request, to vary how much the client batches for itself.
    #[arg(long, default_value_t = 1)]
    rows: usize,

    /// Requests to run and discard first, so model loading and connection
    /// setup stay out of the percentiles.
    #[arg(long, default_value_t = 64)]
    warmup: usize,

    /// Seed for the generated feature rows.
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// How often to poll the server's metrics, in milliseconds. Zero records
    /// no timeseries.
    #[arg(long, default_value_t = 500)]
    poll_ms: u64,

    /// Write the full JSON report here, ready for a timeseries store.
    #[arg(long, short = 'o', value_name = "FILE")]
    report: Option<std::path::PathBuf>,
}

#[derive(Debug, Args)]
struct DescribeArgs {
    /// Model name.
    model: String,

    /// Do not load versions that are not already in memory.
    ///
    /// Without a load a version cannot report its signature, so this trades
    /// the inputs and outputs for a call that costs the server nothing --
    /// worth it against a server whose models are larger than its cache.
    #[arg(long)]
    no_load: bool,
}

#[derive(Debug, Args)]
struct MetricsArgs {
    /// Show only metrics whose name contains this text.
    #[arg(long, short, value_name = "TEXT")]
    filter: Option<String>,

    /// Keep printing every this many seconds instead of once.
    #[arg(long, short, value_name = "SECONDS")]
    watch: Option<f64>,

    /// Show the raw exposition text as the server sent it.
    #[arg(long)]
    raw: bool,
}

#[derive(Debug, Args)]
struct ReadyArgs {
    /// Check this model instead of the whole server.
    #[arg(long, short)]
    model: Option<String>,
    /// Model version; the server chooses when omitted.
    #[arg(long, short = 'V')]
    version: Option<String>,
}

#[derive(Debug, Args)]
struct ModelArgs {
    /// Model name.
    model: String,
    /// Model version; the server chooses when omitted.
    #[arg(long, short = 'V')]
    version: Option<String>,
}

#[derive(Debug, Args)]
struct InferArgs {
    /// Model name.
    model: String,

    /// Model version; the server chooses when omitted.
    #[arg(long, short = 'V')]
    version: Option<String>,

    /// One row of comma-separated feature values. Repeat for a batch.
    #[arg(long, short, value_name = "A,B,C")]
    row: Vec<String>,

    /// Generate this many rows instead of giving them, for a quick trial.
    #[arg(long, conflicts_with = "row", value_name = "N")]
    rows: Option<usize>,

    /// Seed for generated rows, so a run can be repeated.
    #[arg(long, default_value_t = 1, requires = "rows")]
    seed: u64,

    /// Return only these outputs. Repeat for several.
    #[arg(long, short)]
    output: Vec<String>,

    /// Request identifier, returned in the response.
    #[arg(long)]
    id: Option<String>,
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli).await {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

/// The address to use when none was given.
fn default_endpoint(transport: Transport) -> &'static str {
    match transport {
        // HTTP/2 is the same server on the same port, only framed differently.
        Transport::Http | Transport::Http2 => "http://127.0.0.1:8000",
        Transport::Grpc => "http://127.0.0.1:8001",
    }
}

async fn run(cli: Cli) -> Result<ExitCode> {
    let transport = Transport::from(cli.transport);
    let endpoint = cli
        .endpoint
        .clone()
        .unwrap_or_else(|| default_endpoint(transport).to_string());

    let client = connect_with(transport, &endpoint, 1, cli.token.as_deref())
        .await
        .with_context(|| format!("cannot reach {endpoint} over {transport}"))?;

    match cli.command {
        Command::Live => {
            let live = client.server_live().await?;
            report_health(cli.json, "server", live);
            Ok(exit_code(live))
        }
        Command::Ready(args) => {
            let (what, ready) = match &args.model {
                Some(model) => {
                    let reference = ModelRef::new(model.clone(), args.version.clone());
                    (reference.to_string(), client.model_ready(&reference).await?)
                }
                None => ("server".to_string(), client.server_ready().await?),
            };
            report_health(cli.json, &what, ready);
            Ok(exit_code(ready))
        }
        Command::Info => {
            let metadata = client.server_metadata().await?;
            if cli.json {
                println!(
                    "{}",
                    json!({
                        "name": metadata.name,
                        "version": metadata.version,
                        "extensions": metadata.extensions,
                    })
                );
            } else {
                println!("{} {}", metadata.name, metadata.version);
                println!(
                    "extensions: {}",
                    if metadata.extensions.is_empty() {
                        "none".to_string()
                    } else {
                        metadata.extensions.join(", ")
                    }
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Models => {
            let catalogue = client.model_list().await?;
            print_catalogue(cli.json, &catalogue);
            Ok(ExitCode::SUCCESS)
        }
        Command::Describe(args) => {
            let info = client.model_info(&args.model, !args.no_load).await?;
            print_model_info(cli.json, &info);
            Ok(ExitCode::SUCCESS)
        }
        Command::Metadata(args) => {
            let reference = ModelRef::new(args.model, args.version);
            let metadata = client.model_metadata(&reference).await?;
            print_metadata(cli.json, &metadata);
            Ok(ExitCode::SUCCESS)
        }
        Command::Infer(args) => infer(client.as_ref(), args, cli.json).await,
        // Metrics are served over HTTP whichever transport is in use, since the
        // Predict Protocol has no metrics call of its own.
        Command::Metrics(args) => {
            let http = cli
                .endpoint
                .clone()
                .unwrap_or_else(|| default_endpoint(Transport::Http).to_string());
            show_metrics(&http, args, cli.json).await
        }
        Command::Load(ref args) if args.processes > 1 && args.child_report.is_none() => {
            drop(client);
            let args = args.clone();
            run_in_processes(&cli, args).await
        }
        Command::Load(args) => {
            // Metrics always come over HTTP, even when the load runs on gRPC,
            // so a gRPC run still records the server's own view of itself.
            let metrics_endpoint = match transport {
                Transport::Http | Transport::Http2 => endpoint.clone(),
                Transport::Grpc => default_endpoint(Transport::Http).to_string(),
            };

            // The client built above is sized for one request at a time. A
            // load run needs one built for its concurrency -- a pool that can
            // hold a connection per worker, and more than one HTTP/2
            // connection -- or the run measures the client.
            drop(client);
            let plan = ClientConfig::for_concurrency(args.concurrency)
                .with_bearer_token(cli.token.as_deref())
                .with_max_connections(Some(args.max_connections));
            let client = connect_as(transport, &endpoint, &plan)
                .await
                .with_context(|| {
                    format!(
                        "cannot reach {endpoint} over {transport} at concurrency {}",
                        args.concurrency
                    )
                })?;
            load(client, args, metrics_endpoint, cli.json).await
        }
    }
}

async fn load(
    client: Box<dyn PredictV2Client>,
    args: LoadArgs,
    metrics_endpoint: String,
    as_json: bool,
) -> Result<ExitCode> {
    let config = LoadConfig {
        model: args.model.clone(),
        version: args.version,
        concurrency: args.concurrency,
        total_requests: args.duration.is_none().then_some(args.requests),
        duration: args.duration.map(std::time::Duration::from_secs_f64),
        rows_per_request: args.rows,
        warmup_requests: args.warmup,
        seed: args.seed,
        metrics_interval: (args.poll_ms > 0)
            .then(|| std::time::Duration::from_millis(args.poll_ms)),
        metrics_endpoint: Some(metrics_endpoint),
        record_samples: args.samples || args.child_report.is_some(),
    };

    // Opened before the run, not after it: a child of a multi-process run
    // spends the whole run at the edge of its descriptor limit, and the
    // connection pool does not give its sockets back the instant the last
    // request lands. Creating the file afterwards is how a finished run
    // reports `Too many open files` instead of its results.
    let child = match &args.child_report {
        Some(path) => Some((
            std::fs::File::create(path)
                .with_context(|| format!("cannot create {}", path.display()))?,
            path.clone(),
        )),
        None => None,
    };

    if !as_json && child.is_none() {
        println!(
            "driving {} at concurrency {} ({})",
            args.model,
            args.concurrency,
            match (config.total_requests, args.duration) {
                (Some(n), _) => format!("{n} requests"),
                (_, Some(secs)) => format!("{secs}s"),
                _ => "unbounded".to_string(),
            }
        );
    }

    let report = run_load(Arc::from(client), config).await?;

    // A child of a multi-process run reports to its parent through a file
    // and says nothing itself: ten processes printing ten summaries is not a
    // result, it is ten results.
    if let Some((mut file, path)) = child {
        use std::io::Write;
        file.write_all(report.to_json().as_bytes())
            .with_context(|| format!("cannot write {}", path.display()))?;
        return Ok(exit_code(report.summary.completed > 0));
    }

    if let Some(path) = &args.report {
        std::fs::write(path, report.to_json())
            .with_context(|| format!("cannot write {}", path.display()))?;
        if !as_json {
            println!("report written to {}", path.display());
        }
    }

    if as_json {
        println!("{}", report.to_json());
    } else {
        print_load_report(&report);
    }

    // A run where nothing succeeded is a failure, whatever the report says.
    Ok(exit_code(report.summary.completed > 0))
}

/// Runs the load from several operating-system processes and merges what
/// they report.
///
/// Why processes rather than more tasks in this one: a socket is a file
/// descriptor, and a descriptor limit is per process. Ten thousand real
/// connections cannot be held by one process under the usual soft limit of
/// 1024, however many tasks it runs, and ten processes holding a thousand
/// each need no privilege and no `ulimit` at all. It is also the more honest
/// simulation of ten thousand clients, which are ten thousand sockets in
/// separate address spaces rather than streams sharing one write loop.
///
/// Each child is this same binary with the same arguments, `--processes 1`
/// and a file to report into. The parent only waits and adds up.
async fn run_in_processes(cli: &Cli, args: LoadArgs) -> Result<ExitCode> {
    let processes = args.processes.max(1);
    let directory = std::env::temp_dir().join(format!("gmctl-load-{}", std::process::id()));
    std::fs::create_dir_all(&directory)
        .with_context(|| format!("cannot create {}", directory.display()))?;

    let binary = std::env::current_exe().context("cannot find this executable")?;
    if !cli.json {
        println!(
            "driving {} from {processes} processes at concurrency {} each: {} in flight, {} requests each",
            args.model,
            args.concurrency,
            processes * args.concurrency,
            args.requests,
        );
    }

    let mut children = Vec::with_capacity(processes);
    for index in 0..processes {
        let report_path = directory.join(format!("{index}.json"));
        let mut command = std::process::Command::new(&binary);
        command.arg("--transport").arg(cli.transport_name());
        if let Some(endpoint) = &cli.endpoint {
            command.arg("--endpoint").arg(endpoint);
        }
        // The token goes through the environment rather than the command
        // line: an argument is visible in the process list to every user on
        // the machine, and this would put it there once per process.
        if let Some(token) = &cli.token {
            command.env("GALEMIND_TOKEN", token);
        }
        command
            .arg("load")
            .arg(&args.model)
            .arg("--concurrency")
            .arg(args.concurrency.to_string())
            .arg("--rows")
            .arg(args.rows.to_string())
            .arg("--warmup")
            .arg(args.warmup.to_string())
            // A different seed per process, so the processes are not all
            // sending byte-identical requests.
            .arg("--seed")
            .arg((args.seed + index as u64).to_string())
            .arg("--max-connections")
            .arg(args.max_connections.to_string())
            .arg("--child-report")
            .arg(&report_path);

        if let Some(version) = &args.version {
            command.arg("--version").arg(version);
        }
        match args.duration {
            Some(seconds) => {
                command.arg("--duration").arg(seconds.to_string());
            }
            None => {
                command.arg("--requests").arg(args.requests.to_string());
            }
        }
        // One process polls the server's metrics; the rest would be scraping
        // the same numbers N times a second to no purpose.
        command
            .arg("--poll-ms")
            .arg(if index == 0 { args.poll_ms } else { 0 }.to_string());

        let child = command
            .spawn()
            .with_context(|| format!("cannot start {}", binary.display()))?;
        children.push((child, report_path));

        if args.stagger_ms > 0 && index + 1 < processes {
            tokio::time::sleep(std::time::Duration::from_millis(args.stagger_ms)).await;
        }
    }

    let mut reports = Vec::with_capacity(processes);
    let mut failed_children = 0;
    for (mut child, path) in children {
        let status = child.wait().context("cannot wait for a load process")?;
        match std::fs::read_to_string(&path) {
            Ok(json) => match serde_json::from_str::<galemind_client::LoadReport>(&json) {
                Ok(report) => reports.push(report),
                Err(e) => {
                    failed_children += 1;
                    eprintln!("warning: a load process wrote a report that cannot be read: {e}");
                }
            },
            Err(_) => {
                // No report at all: the child died before finishing, and its
                // exit status is the only thing left to say so.
                failed_children += 1;
                eprintln!("warning: a load process exited without a report ({status})");
            }
        }
        let _ = std::fs::remove_file(&path);
    }
    let _ = std::fs::remove_dir(&directory);

    let Some(report) = galemind_client::LoadReport::merge(reports) else {
        anyhow::bail!("no load process produced a report");
    };

    if let Some(path) = &args.report {
        std::fs::write(path, report.to_json())
            .with_context(|| format!("cannot write {}", path.display()))?;
        if !cli.json {
            println!("report written to {}", path.display());
        }
    }

    if cli.json {
        println!("{}", report.to_json());
    } else {
        print_load_report(&report);
        println!(
            "  {} processes, {} sockets",
            report.processes,
            report
                .sockets
                .map(|count| count.to_string())
                .unwrap_or_else(|| "?".to_string())
        );
        if failed_children > 0 {
            println!("  {failed_children} of {processes} processes reported nothing");
        }
    }

    Ok(exit_code(
        report.summary.completed > 0 && failed_children == 0,
    ))
}

fn print_load_report(report: &galemind_client::LoadReport) {
    let summary = &report.summary;
    println!(
        "\n{} over {} at concurrency {}, {} rows per request",
        report.model, report.transport, report.concurrency, report.rows_per_request
    );
    println!(
        "  {} completed, {} failed in {:.2}s",
        summary.completed, summary.failed, summary.wall_secs
    );
    println!(
        "  {:.0} requests/s, {:.0} rows/s",
        summary.throughput_rps, summary.rows_per_sec
    );
    let l = &summary.latency;
    println!(
        "  latency ms  min {:.2}  p50 {:.2}  p55 {:.2}  p90 {:.2}  p95 {:.2}  p99 {:.2}  max {:.2}",
        l.min_ms, l.p50_ms, l.p55_ms, l.p90_ms, l.p95_ms, l.p99_ms, l.max_ms
    );
    if !summary.errors.is_empty() {
        let kinds: Vec<_> = summary
            .errors
            .iter()
            .map(|(kind, count)| format!("{kind}={count}"))
            .collect();
        println!("  failures    {}", kinds.join("  "));
    }

    // The server's own view at the busiest moment, which is where saturation
    // shows up.
    if let Some(peak) = report
        .timeseries
        .iter()
        .max_by(|a, b| a.throughput_rps.total_cmp(&b.throughput_rps))
    {
        println!(
            "  at peak {:.0} req/s: cpu {:.0}%  memory {:.0} MiB  pending {}  mean batch {:.1}",
            peak.throughput_rps,
            peak.cpu_utilization.unwrap_or(0.0) * 100.0,
            peak.memory_bytes.unwrap_or(0.0) / (1024.0 * 1024.0),
            peak.server_pending.unwrap_or(0.0),
            peak.mean_batch_size.unwrap_or(0.0)
        );
    }
}

/// Prints the catalogue as a table, or as the JSON the server sent.
fn print_catalogue(as_json: bool, catalogue: &galemind_core::json::ModelListBody) {
    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(catalogue)
                .unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
        );
        return;
    }

    if catalogue.models.is_empty() {
        println!("the server serves no models");
        return;
    }

    println!(
        "{:<28} {:<10} {:<14} {:<16} LOADED",
        "MODEL", "LATEST", "PLATFORM", "VERSIONS"
    );
    for model in &catalogue.models {
        println!(
            "{:<28} {:<10} {:<14} {:<16} {}",
            model.name,
            model.latest_version,
            model.platform,
            model.versions.join(","),
            if model.loaded_versions.is_empty() {
                "-".to_string()
            } else {
                model.loaded_versions.join(",")
            }
        );
    }
    println!(
        "\n{} model{}, {} loaded",
        catalogue.count,
        if catalogue.count == 1 { "" } else { "s" },
        catalogue.models.iter().filter(|model| model.ready).count()
    );
}

/// Prints every version of one model.
fn print_model_info(as_json: bool, info: &galemind_core::json::ModelInfoBody) {
    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(info).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
        );
        return;
    }

    println!("{} (latest version {})", info.name, info.latest_version);
    for version in &info.versions {
        let flags = [
            version.is_latest.then_some("latest"),
            version.loaded.then_some("loaded"),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

        println!(
            "\n  version {:<10} {:<14} {}",
            version.version,
            version.platform,
            if flags.is_empty() {
                String::new()
            } else {
                format!("[{}]", flags.join(" "))
            }
        );
        if let Some(bytes) = version.size_bytes {
            println!("    size           {bytes} bytes");
        }
        if let Some(size) = version.max_batch_size {
            println!("    max batch      {size}");
        }
        if let Some(threads) = version.intra_op_threads {
            println!("    intra-op       {threads} threads");
        }
        if !version.digest.is_empty() {
            // The digest is what says two deployments are serving the same
            // artifact; the first eight characters are enough to compare.
            println!(
                "    digest         {}",
                &version.digest[..8.min(version.digest.len())]
            );
        }
        for (label, tensors) in [("input", &version.inputs), ("output", &version.outputs)] {
            for tensor in tensors {
                println!(
                    "    {label:<7}        {:<16} {:<6} {:?}",
                    tensor.name, tensor.datatype, tensor.shape
                );
            }
        }
        if version.inputs.is_empty() && version.outputs.is_empty() {
            println!("    signature      unknown; the version is not loaded");
        }
    }
}

async fn show_metrics(endpoint: &str, args: MetricsArgs, as_json: bool) -> Result<ExitCode> {
    loop {
        if args.raw {
            let url = format!("{}/metrics", endpoint.trim_end_matches('/'));
            println!("{}", reqwest::get(&url).await?.text().await?);
        } else {
            let scraped = scrape(endpoint)
                .await
                .with_context(|| format!("cannot read metrics from {endpoint}"))?;
            print_scrape(&scraped, args.filter.as_deref(), as_json);
        }

        match args.watch {
            Some(seconds) if seconds > 0.0 => {
                tokio::time::sleep(std::time::Duration::from_secs_f64(seconds)).await;
                if !as_json {
                    println!();
                }
            }
            _ => return Ok(ExitCode::SUCCESS),
        }
    }
}

fn print_scrape(scraped: &Scrape, filter: Option<&str>, as_json: bool) {
    let wanted: Vec<_> = scraped
        .samples
        .iter()
        .filter(|sample| filter.is_none_or(|text| sample.name.contains(text)))
        .collect();

    if as_json {
        // One object per sample, which is what a timeseries store wants.
        let samples: Vec<_> = wanted
            .iter()
            .map(|sample| {
                json!({
                    "name": sample.name,
                    "labels": sample.labels,
                    "value": sample.value,
                })
            })
            .collect();
        println!(
            "{}",
            json!({
                "scraped_at_unix_ms": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or_default(),
                "samples": samples,
            })
        );
        return;
    }

    for sample in wanted {
        let labels = if sample.labels.is_empty() {
            String::new()
        } else {
            let pairs: Vec<_> = sample
                .labels
                .iter()
                .map(|(name, value)| format!("{name}={value}"))
                .collect();
            format!("  {{{}}}", pairs.join(" "))
        };
        println!(
            "{:<48}{:>14}{labels}",
            sample.name,
            format_value(sample.value)
        );
    }
}

/// Formats a metric value without turning a count into a float.
fn format_value(value: f64) -> String {
    if value.is_finite() && value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        format!("{value:.6}")
    }
}

fn exit_code(ok: bool) -> ExitCode {
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn report_health(as_json: bool, what: &str, ok: bool) {
    if as_json {
        println!("{}", json!({"target": what, "ready": ok}));
    } else {
        println!("{what}: {}", if ok { "ready" } else { "not ready" });
    }
}

fn print_metadata(as_json: bool, metadata: &ModelMetadata) {
    if as_json {
        let tensors = |tensors: &[galemind_core::TensorMetadata]| {
            tensors
                .iter()
                .map(|t| {
                    json!({"name": t.name, "datatype": t.datatype.to_string(), "shape": t.shape})
                })
                .collect::<Vec<_>>()
        };
        println!(
            "{}",
            json!({
                "name": metadata.name,
                "versions": metadata.versions,
                "platform": metadata.platform,
                "inputs": tensors(&metadata.inputs),
                "outputs": tensors(&metadata.outputs),
            })
        );
        return;
    }

    println!("{} ({})", metadata.name, metadata.platform);
    if !metadata.versions.is_empty() {
        println!("versions: {}", metadata.versions.join(", "));
    }
    for (label, tensors) in [("input", &metadata.inputs), ("output", &metadata.outputs)] {
        for tensor in tensors.iter() {
            println!(
                "  {label:<7} {:<16} {:<6} {:?}",
                tensor.name,
                tensor.datatype.to_string(),
                tensor.shape
            );
        }
    }
}

async fn infer(client: &dyn PredictV2Client, args: InferArgs, as_json: bool) -> Result<ExitCode> {
    let reference = ModelRef::new(args.model, args.version);

    // The model's own metadata says what an input is called, how wide a row is
    // and which datatype to send, so the caller only supplies numbers.
    let metadata = client
        .model_metadata(&reference)
        .await
        .with_context(|| format!("cannot read metadata for {reference}"))?;

    let inputs = match args.rows {
        // Generated rows fill every input the model declares, so a model
        // taking an image and an embedding can be tried as easily as one
        // taking a feature matrix.
        Some(count) => synthesize_inputs(&metadata, count, args.seed)?,
        None => {
            // Given rows go to the one input they can belong to.
            let input = sole_input(&metadata)?;
            let rows = args
                .row
                .iter()
                .map(|row| parse_row(row))
                .collect::<Result<Vec<_>, _>>()?;
            if rows.is_empty() {
                anyhow::bail!("give at least one --row, or --rows N to generate some");
            }
            vec![rows_to_tensor(input, &rows)?]
        }
    };

    let request = InferRequest {
        model: reference,
        id: args.id,
        parameters: Parameters::new(),
        inputs,
        requested_outputs: args
            .output
            .iter()
            .map(|name| RequestedOutput {
                name: name.clone(),
                parameters: Parameters::new(),
            })
            .collect(),
    };

    let response = client.infer(request).await?;

    if as_json {
        let outputs = response
            .outputs
            .iter()
            .map(|output| {
                let (data, _) = galemind_core::json::encode(&output.data);
                json!({
                    "name": output.name,
                    "datatype": output.data.datatype().to_string(),
                    "shape": output.shape,
                    "data": data,
                })
            })
            .collect::<Vec<_>>();
        println!(
            "{}",
            json!({
                "model_name": response.model_name,
                "model_version": response.model_version,
                "id": response.id,
                "outputs": outputs,
            })
        );
        return Ok(ExitCode::SUCCESS);
    }

    println!(
        "{}/{}{}",
        response.model_name,
        response.model_version,
        response
            .id
            .as_ref()
            .map(|id| format!("  id={id}"))
            .unwrap_or_default()
    );
    for output in &response.outputs {
        println!(
            "  {:<16} {:<6} {:?}",
            output.name,
            output.data.datatype().to_string(),
            output.shape
        );
        for (index, row) in format_rows(output).iter().enumerate() {
            println!("    [{index}] {row}");
        }
    }
    Ok(ExitCode::SUCCESS)
}

/// Formats an output one row per line, so a batch stays readable.
fn format_rows(output: &galemind_core::Tensor) -> Vec<String> {
    let rows = output.rows().unwrap_or(1).max(1);
    let (data, _) = galemind_core::json::encode(&output.data);
    let elements = match data {
        serde_json::Value::Array(elements) => elements,
        other => vec![other],
    };
    let width = elements.len().div_ceil(rows);

    elements
        .chunks(width.max(1))
        .map(|chunk| {
            chunk
                .iter()
                .map(|value| match value {
                    serde_json::Value::Number(n) => match n.as_f64() {
                        // Keep long float lists readable without hiding
                        // meaningful digits.
                        Some(f) if f.fract() != 0.0 => format!("{f:.6}"),
                        _ => n.to_string(),
                    },
                    other => other.to_string(),
                })
                .collect::<Vec<_>>()
                .join("  ")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn the_command_line_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn a_multiplexed_transport_answers_on_the_rest_port() {
        assert_eq!(default_endpoint(Transport::Http2), "http://127.0.0.1:8000");
        assert_eq!(
            default_endpoint(Transport::Http2),
            default_endpoint(Transport::Http),
            "http2 is the same server, framed differently"
        );
    }

    /// A load run defaults to one process, and the multi-process form is
    /// what asks for real sockets rather than streams.
    #[test]
    fn a_load_run_is_one_process_unless_asked_for_more() {
        let cli = Cli::try_parse_from(["gmctl", "load", "building_energy"]).unwrap();
        let Command::Load(args) = cli.command else {
            panic!("expected load");
        };
        assert_eq!(args.processes, 1);
        assert!(args.child_report.is_none());
        assert_eq!(
            args.max_connections,
            galemind_client::limits::DEFAULT_MAX_CONNECTIONS
        );

        let cli = Cli::try_parse_from([
            "gmctl",
            "--transport",
            "http2",
            "load",
            "thermal_cnn",
            "--processes",
            "10",
            "-c",
            "1000",
            "--stagger-ms",
            "150",
            "--max-connections",
            "64",
        ])
        .unwrap();
        assert_eq!(cli.transport_name(), "http2");
        let Command::Load(args) = cli.command else {
            panic!("expected load");
        };
        assert_eq!(args.processes, 10);
        assert_eq!(args.concurrency, 1_000);
        assert_eq!(args.stagger_ms, 150);
        assert_eq!(args.max_connections, 64);
    }

    #[test]
    fn the_default_endpoint_follows_the_transport() {
        assert_eq!(default_endpoint(Transport::Http), "http://127.0.0.1:8000");
        assert_eq!(default_endpoint(Transport::Grpc), "http://127.0.0.1:8001");
    }

    #[test]
    fn rows_are_collected_from_repeated_flags() {
        let cli = Cli::try_parse_from([
            "gmctl",
            "infer",
            "building_energy",
            "--row",
            "1,2",
            "--row",
            "3,4",
        ])
        .unwrap();
        let Command::Infer(args) = cli.command else {
            panic!("expected infer");
        };
        assert_eq!(args.row, vec!["1,2", "3,4"]);
    }

    #[test]
    fn generated_rows_and_given_rows_cannot_be_mixed() {
        assert!(
            Cli::try_parse_from(["gmctl", "infer", "m", "--row", "1,2", "--rows", "4"]).is_err()
        );
    }

    #[test]
    fn a_seed_without_generated_rows_is_refused() {
        assert!(Cli::try_parse_from(["gmctl", "infer", "m", "--seed", "7"]).is_err());
    }

    #[test]
    fn a_load_run_defaults_to_the_energy_model_and_a_bounded_request_count() {
        let cli = Cli::try_parse_from(["gmctl", "load"]).unwrap();
        let Command::Load(args) = cli.command else {
            panic!("expected load");
        };
        assert_eq!(args.model, "building_energy");
        assert_eq!(args.concurrency, 64);
        assert_eq!(args.requests, 10_000);
        assert_eq!(args.duration, None);
        assert_eq!(args.poll_ms, 500);
    }

    #[test]
    fn a_load_run_is_bounded_by_requests_or_time_but_not_both() {
        assert!(
            Cli::try_parse_from(["gmctl", "load", "--requests", "100", "--duration", "5"]).is_err()
        );

        let cli = Cli::try_parse_from(["gmctl", "load", "--duration", "2.5"]).unwrap();
        let Command::Load(args) = cli.command else {
            panic!("expected load");
        };
        assert_eq!(args.duration, Some(2.5));
    }

    #[test]
    fn a_c10k_run_is_expressible_on_the_command_line() {
        let cli = Cli::try_parse_from([
            "gmctl",
            "load",
            "building_energy",
            "--concurrency",
            "10000",
            "--requests",
            "200000",
            "--report",
            "/tmp/c10k.json",
        ])
        .unwrap();
        let Command::Load(args) = cli.command else {
            panic!("expected load");
        };
        assert_eq!(args.concurrency, 10_000);
        assert_eq!(args.requests, 200_000);
        assert_eq!(args.report.unwrap().to_str(), Some("/tmp/c10k.json"));
    }

    #[test]
    fn the_catalogue_commands_parse() {
        let cli = Cli::try_parse_from(["gmctl", "models"]).unwrap();
        assert!(matches!(cli.command, Command::Models));

        let cli = Cli::try_parse_from(["gmctl", "describe", "building_energy"]).unwrap();
        let Command::Describe(args) = cli.command else {
            panic!("expected describe");
        };
        assert_eq!(args.model, "building_energy");
        // Loading is the default, so the signature is reported.
        assert!(!args.no_load);

        let cli =
            Cli::try_parse_from(["gmctl", "describe", "building_energy", "--no-load"]).unwrap();
        let Command::Describe(args) = cli.command else {
            panic!("expected describe");
        };
        assert!(args.no_load);
    }

    #[test]
    fn metrics_can_be_watched_and_filtered() {
        let cli =
            Cli::try_parse_from(["gmctl", "metrics", "--filter", "inference", "--watch", "2"])
                .unwrap();
        let Command::Metrics(args) = cli.command else {
            panic!("expected metrics");
        };
        assert_eq!(args.filter.as_deref(), Some("inference"));
        assert_eq!(args.watch, Some(2.0));
        assert!(!args.raw);
    }

    #[test]
    fn a_whole_number_metric_is_not_printed_as_a_float() {
        assert_eq!(format_value(12.0), "12");
        assert_eq!(format_value(0.25), "0.250000");
        assert_eq!(format_value(f64::INFINITY), "inf");
    }

    #[test]
    fn the_transport_is_chosen_by_name() {
        let cli = Cli::try_parse_from(["gmctl", "--transport", "grpc", "live"]).unwrap();
        assert_eq!(cli.transport, TransportArg::Grpc);
        assert!(Cli::try_parse_from(["gmctl", "--transport", "carrier-pigeon", "live"]).is_err());
    }

    #[test]
    fn a_not_ready_answer_exits_non_zero() {
        assert_eq!(
            format!("{:?}", exit_code(false)),
            format!("{:?}", ExitCode::FAILURE)
        );
    }
}
