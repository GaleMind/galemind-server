# GaleMind Server

GaleMind ML Inference Server v0.1 - A high-performance machine learning inference server providing both REST and gRPC APIs.

## Prerequisites

- **Rust** (1.70+): Install from [rustup.rs](https://rustup.rs/)
- **Make**: Required for using the Makefile commands

# Github deployment notes
- pushing on main or develop will trigger 
  - Docker imange building, 
  - and (upon success) pushing to `galemindzen`'s _Docker hub_ private repo
- pushing a `v*` tag onto any commit will trigger its docker image building and (upon success) pushing to `galemindzen`'s _Docker hub_ private repo ; if tag ends with `+k8s` then also get the same effect as `k8s_v*` tag
- pushing a `k8s_v*` tag tagging onto any commit will trigger its k8s deployment (upon successful last Docker image building and pushing) onto Galemind's Kubernetes cluster on Linode

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd galemind-server
```

2. Install dependencies:  
Make sure you have installed `libssl-dev`! Rust openSSL crate depends on it.  
For Debian derivatives:
```bash
sudo apt install libssl-dev
```

Make sure you have installed `protobuf-compiler`! Rust grpc_server crate depends on it.  
For Debian derivatives:  
```bash
sudo apt install protobuf-compiler
```

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

## API Usage

The REST server supports both the native Galemind protocol and OpenAI-compatible API through the `X-Protocol-Inference` header.

### OpenAI Protocol

Use `X-Protocol-Inference: openai` header to interact with OpenAI-compatible endpoints:

#### Chat Completions
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Protocol-Inference: openai" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 150
  }'
```

#### List Models
```bash
curl -X GET http://localhost:8080/v1/models \
  -H "X-Protocol-Inference: openai"
```

#### Model Ready Check
```bash
curl -X GET http://localhost:8080/v1/models/gpt-3.5-turbo/ready \
  -H "X-Protocol-Inference: openai"
```

### Galemind Protocol

Use `X-Protocol-Inference: galemind` header (or omit header for default) to use the native Galemind protocol:

#### Inference Request
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Protocol-Inference: galemind" \
  -d '{
    "id": "test-request-1",
    "inputs": [
      {
        "name": "input_text",
        "shape": [1],
        "datatype": "string",
        "data": ["Hello, how are you?"]
      }
    ]
  }'
```

#### List Models
```bash
curl -X GET http://localhost:8080/v1/models \
  -H "X-Protocol-Inference: galemind"
```

#### Model Ready Check
```bash
curl -X GET http://localhost:8080/v1/models/my-model/ready \
  -H "X-Protocol-Inference: galemind"
```

## gRPC Unified Interface

The gRPC server now supports an enhanced unified interface that provides:

- **Protocol Selection**: Choose between Galemind and OpenAI protocols
- **Multiple Content Types**: Text, Binary, and Base64 content support
- **Streaming Support**: Advanced streaming with chunk management and end-of-stream detection
- **Backward Compatibility**: Full compatibility with existing ModelInfer methods

### New gRPC Service Methods

#### UnifiedInfer (Single Request/Response)
```protobuf
rpc UnifiedInfer(UnifiedInferRequest) returns (UnifiedInferResponse)
```

#### UnifiedInferStream (Bidirectional Streaming)
```protobuf
rpc UnifiedInferStream(stream UnifiedInferRequest) returns (stream UnifiedInferResponse)
```

### Message Structure

#### UnifiedInferRequest
```protobuf
message UnifiedInferRequest {
  InferenceProtocol protocol = 1;           // PROTOCOL_GALEMIND or PROTOCOL_OPENAI
  optional ModelInferRequest legacy_request = 2;  // For backward compatibility
  MessageContent content = 3;               // Enhanced content with type support
  optional StreamMetadata stream_metadata = 4;    // Streaming metadata
  string model_name = 5;
  string model_version = 6;
  string request_id = 7;
  map<string, InferParameter> parameters = 8;
  map<string, string> metadata = 9;
}
```

#### Content Types
- **CONTENT_TYPE_TEXT**: Plain text content
- **CONTENT_TYPE_BINARY**: Raw binary data
- **CONTENT_TYPE_BASE64**: Base64-encoded content

#### Streaming Features
- **Stream ID**: Unique identifier for stream sessions
- **Chunk Sequencing**: Ordered chunk processing with sequence numbers
- **End-of-Stream Detection**: Automatic stream completion handling
- **Stream Reconstruction**: Automatic combining of chunked content

### Backward Compatibility

The enhanced interface maintains full backward compatibility:

1. **Legacy Support**: Original `ModelInfer` and `ModelInferAsync` methods continue to work
2. **Legacy Request Field**: Use `legacy_request` field in `UnifiedInferRequest` to wrap existing requests
3. **Protocol Fallback**: Defaults to Galemind protocol when not specified

### Example Usage Patterns

#### Single Request with OpenAI Protocol
```protobuf
UnifiedInferRequest {
  protocol: PROTOCOL_OPENAI
  content: {
    content_type: CONTENT_TYPE_TEXT
    text_content: "Hello, how are you?"
  }
  model_name: "gpt-3.5-turbo"
  request_id: "req_123"
}
```

#### Streaming with Chunks
```protobuf
// Chunk 1
UnifiedInferRequest {
  protocol: PROTOCOL_GALEMIND
  content: {
    content_type: CONTENT_TYPE_TEXT
    text_content: "First part of message"
  }
  stream_metadata: {
    stream_id: "stream_456"
    chunk_sequence: 1
    is_streaming: true
    end_of_stream: false
    total_chunks: 3
  }
  model_name: "my-model"
  request_id: "req_456"
}

// Final Chunk
UnifiedInferRequest {
  protocol: PROTOCOL_GALEMIND
  content: {
    content_type: CONTENT_TYPE_TEXT
    text_content: "Final part of message"
  }
  stream_metadata: {
    stream_id: "stream_456"
    chunk_sequence: 3
    is_streaming: true
    end_of_stream: true
    total_chunks: 3
  }
  model_name: "my-model"
  request_id: "req_456"
}
```

#### Binary Content Processing
```protobuf
UnifiedInferRequest {
  protocol: PROTOCOL_GALEMIND
  content: {
    content_type: CONTENT_TYPE_BINARY
    binary_content: [raw_bytes_here]
  }
  model_name: "image-processor"
  request_id: "req_789"
}
```

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