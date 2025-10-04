use std::time::Duration;
use tokio::time::sleep;

pub const DEFAULT_REST_ENDPOINT: &str = "http://localhost:8080";
pub const DEFAULT_GRPC_ENDPOINT: &str = "http://localhost:50051";
pub const MLFLOW_ENDPOINT: &str = "http://localhost:5000";

pub async fn wait_for_service(url: &str, max_attempts: u32) -> anyhow::Result<()> {
    let client = reqwest::Client::new();

    for attempt in 1..=max_attempts {
        match client.get(url).send().await {
            Ok(response) if response.status().is_success() => {
                println!("Service at {} is ready (attempt {})", url, attempt);
                return Ok(());
            }
            Ok(response) => {
                println!("Service at {} returned status {} (attempt {})", url, response.status(), attempt);
            }
            Err(e) => {
                println!("Failed to connect to {} (attempt {}): {}", url, attempt, e);
            }
        }

        if attempt < max_attempts {
            sleep(Duration::from_secs(2)).await;
        }
    }

    anyhow::bail!("Service at {} did not become available after {} attempts", url, max_attempts)
}

pub async fn setup_test_environment() -> anyhow::Result<()> {
    println!("Waiting for services to be ready...");

    // Wait for MLflow
    wait_for_service(&format!("{}/health", MLFLOW_ENDPOINT), 30).await
        .or_else(|_| wait_for_service(MLFLOW_ENDPOINT, 5))?;

    // Wait for GaleMind server
    wait_for_service(&format!("{}/health", DEFAULT_REST_ENDPOINT), 30).await?;

    println!("All services are ready!");
    Ok(())
}