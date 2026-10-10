//! Classifying a building's energy band over gRPC.
//!
//! Start a server first:
//!
//! ```text
//! cargo run -p galemind -- serve
//! cargo run -p galemind-client --example grpc
//! ```
//!
//! The endpoint can be overridden with `GALEMIND_GRPC_ENDPOINT`.
//!
//! gRPC sends tensor contents as raw bytes rather than as JSON numbers, which
//! avoids per-element encoding and is the only form that carries 16-bit floats
//! exactly. The client API is otherwise identical to the HTTP one: both
//! implement [`PredictV2Client`], so a program can switch transport by
//! configuration.

use galemind_client::{GrpcClient, PredictV2Client};
use galemind_core::{InferRequest, ModelRef, Parameters, Tensor, TensorData};

/// Three buildings of decreasing energy intensity.
const BUILDINGS: [[f32; 8]; 3] = [
    [1800.0, 30.0, -4.0, 18.0, 22.0, 1.0, 0.0, 40.0],
    [400.0, 12.0, 8.0, 11.0, 21.5, 3.0, 0.0, 250.0],
    [55.0, 2.0, 18.0, 14.0, 21.0, 5.0, 1.0, 600.0],
];

/// The bands the classifier was trained on, in label order.
const BANDS: [&str; 3] = ["efficient", "typical", "wasteful"];

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let endpoint = std::env::var("GALEMIND_GRPC_ENDPOINT")
        .unwrap_or_else(|_| "http://127.0.0.1:8001".to_string());
    let client = GrpcClient::connect(&endpoint).await?;

    if !client.server_ready().await? {
        return Err(format!("{endpoint} is not ready to serve models").into());
    }
    println!("connected to {endpoint} over gRPC");

    let model = ModelRef::latest("building_energy_class");
    let metadata = client.model_metadata(&model).await?;
    println!(
        "{} returns {}",
        metadata.name,
        metadata
            .outputs
            .iter()
            .map(|t| format!("{} {:?}", t.name, t.shape))
            .collect::<Vec<_>>()
            .join(" and ")
    );

    // Only the band is asked for here; the probabilities are left behind rather
    // than sent over the wire for nothing.
    let features: Vec<f32> = BUILDINGS.iter().flatten().copied().collect();
    let response = client
        .infer(InferRequest {
            model,
            id: Some("grpc-example".to_string()),
            parameters: Parameters::new(),
            inputs: vec![Tensor::new(
                "features",
                vec![BUILDINGS.len() as i64, 8],
                TensorData::Fp32(features),
            )],
            requested_outputs: vec![galemind_core::RequestedOutput {
                name: "band".to_string(),
                parameters: Parameters::new(),
            }],
        })
        .await?;

    let TensorData::Int64(bands) = &response.outputs[0].data else {
        return Err("expected an INT64 band output".into());
    };
    for (building, band) in BUILDINGS.iter().zip(bands) {
        let name = BANDS.get(*band as usize).copied().unwrap_or("unknown");
        println!("  {:>6.0} m^2 -> {name}", building[0]);
    }

    Ok(())
}
