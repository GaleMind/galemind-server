use e2e_tests::{setup_test_environment, RestClient, GrpcClient, DEFAULT_REST_ENDPOINT, DEFAULT_GRPC_ENDPOINT, MLFLOW_ENDPOINT};
use serde_json::json;

#[tokio::test]
async fn test_full_integration_flow() {
    setup_test_environment().await.expect("Failed to setup test environment");

    // Test both REST and gRPC endpoints are accessible
    let rest_client = RestClient::new(DEFAULT_REST_ENDPOINT);
    let grpc_client = GrpcClient::new(DEFAULT_GRPC_ENDPOINT);

    // Check REST API health
    let rest_health = rest_client.health_check().await.expect("REST health check failed");
    assert_eq!(rest_health.status, "OK");

    // Check gRPC connection
    let _grpc_channel = grpc_client.connect().await.expect("gRPC connection failed");

    // Test MLflow is accessible (basic connectivity test)
    let mlflow_client = reqwest::Client::new();
    let mlflow_response = mlflow_client
        .get(MLFLOW_ENDPOINT)
        .send()
        .await
        .expect("Failed to connect to MLflow");

    assert!(
        mlflow_response.status().is_success() || mlflow_response.status().is_redirection(),
        "MLflow should be accessible"
    );

    println!("Full integration test passed - all services are communicating");
}

#[tokio::test]
async fn test_service_dependencies() {
    setup_test_environment().await.expect("Failed to setup test environment");

    let rest_client = RestClient::new(DEFAULT_REST_ENDPOINT);

    // Test that the server can list models (basic functionality)
    let models = rest_client.list_models().await.expect("Failed to list models");
    println!("Available models: {:?}", models);

    // If models are available, test a simple inference flow
    if !models.is_empty() {
        let model_name = &models[0];

        // Get model info
        let model_info = rest_client.model_info(model_name).await;
        println!("Model info for {}: {:?}", model_name, model_info);

        // Try inference (may fail if no actual model is loaded, but should not crash)
        let inference_request = e2e_tests::rest_client::InferenceRequest {
            model_name: model_name.clone(),
            input_data: json!({"features": [1.0, 2.0, 3.0]}),
            parameters: Some(json!({"output_format": "json"})),
        };

        let inference_result = rest_client.inference(inference_request).await;
        println!("Inference result: {:?}", inference_result);
    }
}

#[tokio::test]
async fn test_error_handling() {
    setup_test_environment().await.expect("Failed to setup test environment");

    let rest_client = RestClient::new(DEFAULT_REST_ENDPOINT);

    // Test various error scenarios

    // 1. Non-existent model
    let bad_request = e2e_tests::rest_client::InferenceRequest {
        model_name: "definitely_not_a_real_model_12345".to_string(),
        input_data: json!({"data": []}),
        parameters: None,
    };

    let result = rest_client.inference(bad_request).await;
    assert!(result.is_err(), "Should fail with non-existent model");

    // 2. Invalid model info request
    let model_info_result = rest_client.model_info("invalid_model_name").await;
    assert!(result.is_err(), "Should fail with invalid model name");

    println!("Error handling tests passed");
}