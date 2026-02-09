# E2E Tests for GaleMind Server

This directory contains end-to-end tests for the GaleMind Server that test the complete system including MLflow integration.

## Overview

The E2E tests are designed to run against the full system deployed via Docker Compose, including:
- GaleMind Server (REST and gRPC APIs)
- MLflow Tracking Server
- All dependencies and services

## Test Structure

- `src/lib.rs` - Main library exports
- `src/common.rs` - Common utilities and service setup
- `src/rest_client.rs` - REST API client for testing
- `src/grpc_client.rs` - gRPC client for testing
- `tests/rest_e2e_tests.rs` - REST API end-to-end tests
- `tests/grpc_e2e_tests.rs` - gRPC API end-to-end tests
- `tests/integration_tests.rs` - Full system integration tests
- `run_e2e_tests.sh` - Test runner script

## Running Tests

### Prerequisites

1. Docker and Docker Compose installed
2. Rust toolchain installed

### Using the Test Runner (Recommended)

```bash
# From the project root
./tests/run_e2e_tests.sh
```

This script will:
1. Start all services using docker-compose
2. Wait for services to be healthy
3. Run all E2E tests
4. Clean up services afterwards

### Manual Testing

1. Start services:
```bash
docker-compose -f docker-compose.dev.yml up -d
```

2. Wait for services to be ready (you can check with `docker-compose ps`)

3. Run tests:
```bash
cd tests
cargo test
```

4. Clean up:
```bash
docker-compose -f docker-compose.dev.yml down
```

### Running Specific Tests

```bash
cd tests

# Run only REST API tests
cargo test rest_e2e_tests

# Run only gRPC tests
cargo test grpc_e2e_tests

# Run only integration tests
cargo test integration_tests

# Run with verbose output
cargo test -- --nocapture
```

## Test Categories

### REST API Tests (`rest_e2e_tests.rs`)
- Health check endpoint
- Model listing
- Model information retrieval
- Inference requests (valid and invalid)
- Concurrent request handling

### gRPC Tests (`grpc_e2e_tests.rs`)
- gRPC connection establishment
- Health check via gRPC
- Inference via gRPC
- Concurrent gRPC requests

### Integration Tests (`integration_tests.rs`)
- Full system integration flow
- Service dependency verification
- MLflow connectivity
- Error handling across services

## Test Configuration

Tests use the following default endpoints:
- REST API: `http://localhost:8080`
- gRPC API: `http://localhost:50051`
- MLflow: `http://localhost:5000`

These can be modified in `src/common.rs` if needed.

## Troubleshooting

### Services Not Starting
- Check Docker Compose logs: `docker-compose -f docker-compose.dev.yml logs`
- Ensure ports 8080, 50051, and 5000 are not in use

### Tests Failing
- Verify all services are healthy: `docker-compose -f docker-compose.dev.yml ps`
- Check if models directory exists and is accessible
- Run tests with verbose output: `cargo test -- --nocapture`

### MLflow Connection Issues
- MLflow may take longer to start up
- Check MLflow logs: `docker-compose -f docker-compose.dev.yml logs mlflow`
- Verify MLflow is accessible at http://localhost:5000

## Development

To add new tests:
1. Add test functions to the appropriate test file
2. Use the existing client classes for making requests
3. Follow the pattern of calling `setup_test_environment()` at the start of each test
4. Test both success and failure scenarios

To modify the test clients:
1. Update `src/rest_client.rs` for REST API changes
2. Update `src/grpc_client.rs` for gRPC changes (requires protobuf definitions)
3. Update `src/common.rs` for shared utilities