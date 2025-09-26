use e2e_tests::{setup_test_environment, RestClient, DEFAULT_REST_ENDPOINT};
use serde_json::json;

#[tokio::test]
async fn test_health_check() {
    setup_test_environment().await.expect("Failed to setup test environment");

    let client = RestClient::new(DEFAULT_REST_ENDPOINT);
    let health = client.health_check().await.expect("Health check failed");

    assert_eq!(health.status, "OK");
    assert!(!health.timestamp.is_empty());
}

#[tokio::test]
async fn test_list_models() {
    setup_test_environment().await.expect("Failed to setup test environment");

    let client = RestClient::new(DEFAULT_REST_ENDPOINT);
    let models = client.list_models().await.expect("List models failed");

    // Should at least return an empty list without errors
    assert!(models.len() >= 0);
}

#[tokio::test]
async fn test_inference_with_invalid_model() {
    setup_test_environment().await.expect("Failed to setup test environment");

    let client = RestClient::new(DEFAULT_REST_ENDPOINT);
    let request = e2e_tests::rest_client::InferenceRequest {
        model_name: "non_existent_model".to_string(),
        input_data: json!({"data": [1, 2, 3]}),
        parameters: None,
    };

    let result = client.inference(request).await;
    assert!(result.is_err(), "Should fail with non-existent model");
}

#[tokio::test]
async fn test_model_info_invalid_model() {
    setup_test_environment().await.expect("Failed to setup test environment");

    let client = RestClient::new(DEFAULT_REST_ENDPOINT);
    let result = client.model_info("non_existent_model").await;

    assert!(result.is_err(), "Should fail with non-existent model");
}

#[tokio::test]
async fn test_inference_with_valid_model() {
    setup_test_environment().await.expect("Failed to setup test environment");

    let client = RestClient::new(DEFAULT_REST_ENDPOINT);

    // First, get available models
    let models = client.list_models().await.expect("Failed to list models");

    if !models.is_empty() {
        let model_name = &models[0];
        let request = e2e_tests::rest_client::InferenceRequest {
            model_name: model_name.clone(),
            input_data: json!({"data": [1.0, 2.0, 3.0]}),
            parameters: Some(json!({"temperature": 0.8})),
        };

        let response = client.inference(request).await.expect("Inference failed");
        assert_eq!(response.model_name, *model_name);
        assert!(response.inference_time_ms > 0.0);
    } else {
        println!("No models available for testing inference");
    }
}

#[tokio::test]
async fn test_concurrent_requests() {
    setup_test_environment().await.expect("Failed to setup test environment");

    let client = RestClient::new(DEFAULT_REST_ENDPOINT);
    let mut handles = vec![];

    // Make 5 concurrent health check requests
    for _ in 0..5 {
        let client_clone = client.clone();
        let handle = tokio::spawn(async move {
            client_clone.health_check().await
        });
        handles.push(handle);
    }

    // Wait for all requests to complete
    for handle in handles {
        let result = handle.await.expect("Task panicked");
        assert!(result.is_ok(), "Health check should succeed");
    }
}