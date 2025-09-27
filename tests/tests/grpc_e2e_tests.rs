use e2e_tests::{setup_test_environment, GrpcClient, DEFAULT_GRPC_ENDPOINT};
use serde_json::json;

#[tokio::test]
async fn test_grpc_connection() {
    setup_test_environment().await.expect("Failed to setup test environment");

    let client = GrpcClient::new(DEFAULT_GRPC_ENDPOINT);
    let result = client.connect().await;

    assert!(result.is_ok(), "Should be able to connect to gRPC server");
}

#[tokio::test]
async fn test_grpc_health_check() {
    setup_test_environment().await.expect("Failed to setup test environment");

    let client = GrpcClient::new(DEFAULT_GRPC_ENDPOINT);
    let health = client.health_check().await.expect("gRPC health check failed");

    assert_eq!(health, "OK");
}

#[tokio::test]
async fn test_grpc_inference() {
    setup_test_environment().await.expect("Failed to setup test environment");

    let client = GrpcClient::new(DEFAULT_GRPC_ENDPOINT);
    let input_data = json!({"data": [1.0, 2.0, 3.0]});

    let result = client.inference("test_model", input_data).await;

    // Since this is a placeholder implementation, we just check it doesn't panic
    assert!(result.is_ok(), "gRPC inference should not fail");
}

#[tokio::test]
async fn test_concurrent_grpc_requests() {
    setup_test_environment().await.expect("Failed to setup test environment");

    let client = GrpcClient::new(DEFAULT_GRPC_ENDPOINT);
    let mut handles = vec![];

    // Make 3 concurrent gRPC requests
    for _ in 0..3 {
        let client_clone = client.clone();
        let handle = tokio::spawn(async move {
            client_clone.health_check().await
        });
        handles.push(handle);
    }

    // Wait for all requests to complete
    for handle in handles {
        let result = handle.await.expect("Task panicked");
        assert!(result.is_ok(), "gRPC health check should succeed");
    }
}