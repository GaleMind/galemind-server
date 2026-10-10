//! `gmctl`: a command-line client for a Predict Protocol v2 server.

#![forbid(unsafe_code)]

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use galemind_client::{
    connect, parse_row, rows_to_tensor, scrape, sole_input, synthesize_rows, PredictV2Client,
    Scrape, Transport,
};
use galemind_core::{InferRequest, ModelMetadata, ModelRef, Parameters, RequestedOutput};
use serde_json::json;
use std::process::ExitCode;

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

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum TransportArg {
    Http,
    Grpc,
}

impl From<TransportArg> for Transport {
    fn from(value: TransportArg) -> Self {
        match value {
            TransportArg::Http => Transport::Http,
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
    /// Show a model's inputs and outputs.
    Metadata(ModelArgs),
    /// Run an inference.
    Infer(InferArgs),
    /// Read the server's Prometheus metrics.
    Metrics(MetricsArgs),
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
        Transport::Http => "http://127.0.0.1:8000",
        Transport::Grpc => "http://127.0.0.1:8001",
    }
}

async fn run(cli: Cli) -> Result<ExitCode> {
    let transport = Transport::from(cli.transport);
    let endpoint = cli
        .endpoint
        .clone()
        .unwrap_or_else(|| default_endpoint(transport).to_string());

    let client = connect(transport, &endpoint)
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
                .unwrap_or_else(|| default_endpoint(Transport::Http).to_string());
            show_metrics(&http, args, cli.json).await
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
    let input = sole_input(&metadata)?;

    let rows = match args.rows {
        Some(count) => synthesize_rows(count, galemind_client::row_width(input)?, args.seed),
        None => args
            .row
            .iter()
            .map(|row| parse_row(row))
            .collect::<Result<Vec<_>, _>>()?,
    };
    if rows.is_empty() {
        anyhow::bail!("give at least one --row, or --rows N to generate some");
    }

    let request = InferRequest {
        model: reference,
        id: args.id,
        parameters: Parameters::new(),
        inputs: vec![rows_to_tensor(input, &rows)?],
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
