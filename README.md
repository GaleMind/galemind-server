# GaleMind Server

GaleMind ML Inference Server v0.1 - A high-performance machine learning inference server providing both REST and gRPC APIs.

## Prerequisites

- **Rust** (1.70+): Install from [rustup.rs](https://rustup.rs/)
- **Make**: Required for using the Makefile commands

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd galemind-server
```

2. Install Rust dependencies:
```bash
cargo build
```

## Compilation

### Using Makefile (Recommended)

```bash
# Build the entire project (includes format and test)
make all

# Run tests only
make test

# Format code
make format

# Run the server
make run
```

### Using Cargo directly

```bash
# Build the project
cargo build

# Build for production (optimized)
cargo build --release

# Run tests
cargo test

# Format code
cargo fmt
```

## Usage

### Environment Variables

Set the required environment variables in the `.env` file (recommended):
```bash
export MODELS_DIR=/path/to/your/models
```

### Starting the Server

Using Makefile (automatically loads environment variables from `.env`):
```bash
make run
```

Or using cargo directly:
```bash
cargo run -p galemind start
```

### Server Configuration

The server supports the following command-line options:

```bash
cargo run -p galemind start \
  --rest-host 0.0.0.0 \
  --rest-port 8080 \
  --grpc-host 0.0.0.0 \
  --grpc-port 50051
```

- **REST API**: Available at `http://localhost:8080` (default)
- **gRPC API**: Available at `localhost:50051` (default)

### Available Make Commands

| Command | Description |
|---------|-------------|
| `make all` | Format code, run tests, and build the project |
| `make test` | Run all tests |
| `make format` | Format code using cargo fmt |
| `make run` | Start the GaleMind server |

## Project Structure

This is a Rust workspace containing:
- `src/galemind/` - Main server application
- `src/grpc_server/` - gRPC server implementation
- `src/rest_server/` - REST API server implementation

## License

See the LICENSE file for details.