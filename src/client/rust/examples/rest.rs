//! Predicting building energy use over HTTP/REST.
//!
//! Start a server first:
//!
//! ```text
//! cargo run -p galemind -- serve
//! cargo run -p galemind-client --example rest
//! ```
//!
//! The endpoint can be overridden with `GALEMIND_ENDPOINT`.

use galemind_client::{HttpClient, PredictV2Client};
use galemind_core::{InferRequest, ModelRef, Parameters, Tensor, TensorData};

/// Two buildings: a large, poorly insulated office on a cold evening, and a
/// small, well insulated flat on a mild sunny afternoon. The features are, in
/// order: floor area, occupancy, outdoor temperature, hour of day, heating
/// setpoint, insulation rating, whether it is a weekend, solar irradiance.
const BUILDINGS: [[f32; 8]; 2] = [
    [1800.0, 30.0, -4.0, 18.0, 22.0, 1.0, 0.0, 40.0],
    [55.0, 2.0, 18.0, 14.0, 21.0, 5.0, 1.0, 600.0],
];

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let endpoint =
        std::env::var("GALEMIND_ENDPOINT").unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());
    let client = HttpClient::new(&endpoint)?;

    // 1. Is the server up, and is it ready to serve models?
    if !client.server_live().await? {
        return Err(format!("{endpoint} is not live").into());
    }
    let server = client.server_metadata().await?;
    println!("connected to {} {} over HTTP", server.name, server.version);

    // 2. Ask the model what it takes and what it returns. A client that reads
    //    the metadata instead of hardcoding shapes keeps working when the model
    //    is retrained with different inputs.
    let model = ModelRef::latest("building_energy");
    if !client.model_ready(&model).await? {
        return Err(format!("{model} is not ready").into());
    }
    let metadata = client.model_metadata(&model).await?;
    println!(
        "{} ({}) takes {} {:?} and returns {} {:?}",
        metadata.name,
        metadata.platform,
        metadata.inputs[0].name,
        metadata.inputs[0].shape,
        metadata.outputs[0].name,
        metadata.outputs[0].shape,
    );

    // 3. Send both buildings as one request. A batched request is one round
    //    trip and one model execution, which is far cheaper than two.
    let features: Vec<f32> = BUILDINGS.iter().flatten().copied().collect();
    let response = client
        .infer(InferRequest {
            model: model.clone(),
            id: Some("rest-example".to_string()),
            parameters: Parameters::new(),
            inputs: vec![Tensor::new(
                &metadata.inputs[0].name,
                vec![BUILDINGS.len() as i64, 8],
                TensorData::Fp32(features),
            )],
            // Empty means every output the model produces.
            requested_outputs: Vec::new(),
        })
        .await?;

    // 4. Read the answer. The response says which version served it, which
    //    matters once more than one version is deployed.
    println!(
        "served by {}/{}",
        response.model_name, response.model_version
    );
    let TensorData::Fp32(kwh) = &response.outputs[0].data else {
        return Err("expected an FP32 output".into());
    };
    for (building, predicted) in BUILDINGS.iter().zip(kwh) {
        println!(
            "  {:>6.0} m^2, {:>3.0} people, {:>5.1} C outside -> {predicted:>8.1} kWh/day",
            building[0], building[1], building[2]
        );
    }

    Ok(())
}
