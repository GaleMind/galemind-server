mod translator;

use async_trait::async_trait;
use foundation::api::inference::InferParameter;
use foundation::{
    InferenceRequest, InferenceServerBuilder, InferenceServerConfig, ModelDiscoveryService, ModelId,
};
use futures::Stream;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, transport::Server};

// Include the generated protobuf code
pub mod grpc_server {
    tonic::include_proto!("grpc_server");
}

use grpc_server::{
    ContentType, InferenceProtocol, MessageContent, ModelInferRequest, ModelInferResponse,
    ModelMetadataRequest, ModelMetadataResponse, ModelReadyRequest, ModelReadyResponse,
    PerformanceMetrics, ResponseStatus, ServerLiveRequest, ServerLiveResponse,
    ServerMetadataRequest, ServerMetadataResponse, ServerReadyRequest, ServerReadyResponse,
    StreamMetadata, TokenUsage, UnifiedInferRequest, UnifiedInferResponse,
    model_metadata_response::TensorMetadata,
    prediction_service_server::{PredictionService, PredictionServiceServer},
};

pub struct PredictionServiceImpl {
    model_manager: Arc<ModelDiscoveryService>,
}

impl PredictionServiceImpl {
    pub fn new(model_manager: Arc<ModelDiscoveryService>) -> Self {
        Self { model_manager }
    }
}

#[tonic::async_trait]
impl PredictionService for PredictionServiceImpl {
    type ModelInferAsyncStream =
        Pin<Box<dyn Stream<Item = Result<ModelInferResponse, Status>> + Send>>;
    type UnifiedInferStreamStream =
        Pin<Box<dyn Stream<Item = Result<UnifiedInferResponse, Status>> + Send>>;

    async fn server_live(
        &self,
        request: Request<ServerLiveRequest>,
    ) -> Result<Response<ServerLiveResponse>, Status> {
        println!("Got a request: {:?}", request);

        let reply = ServerLiveResponse { live: true };

        Ok(Response::new(reply))
    }

    async fn server_ready(
        &self,
        request: Request<ServerReadyRequest>,
    ) -> Result<Response<ServerReadyResponse>, Status> {
        println!("Got a request: {:?}", request);

        let reply = ServerReadyResponse { ready: true };

        Ok(Response::new(reply))
    }

    async fn model_ready(
        &self,
        request: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        println!("Got a request: {:?}", request);

        let reply = ModelReadyResponse { ready: true };

        Ok(Response::new(reply))
    }

    async fn server_metadata(
        &self,
        request: Request<ServerMetadataRequest>,
    ) -> Result<Response<ServerMetadataResponse>, Status> {
        println!("Got a request: {:?}", request);

        let reply = ServerMetadataResponse {
            name: "server_metadata".to_string(),
            version: "v1.0.0".to_string(),
            extensions: vec!["extension1".to_string(), "extension2".to_string()],
        };

        Ok(Response::new(reply))
    }

    async fn model_metadata(
        &self,
        request: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        println!("Got a request: {:?}", request);

        let reply = ModelMetadataResponse {
            name: "model_metadata".to_string(),
            versions: vec!["v1.0.0".to_string(), "v2.0.0".to_string()],
            platform: "platform".to_string(),
            inputs: vec![
                TensorMetadata {
                    name: "tensor_metadata_input1".to_string(),
                    datatype: "float32".to_string(),
                    shape: vec![1, 224, 224, 3],
                },
                TensorMetadata {
                    name: "tensor_metadata_input2".to_string(),
                    datatype: "int64".to_string(),
                    shape: vec![1],
                },
            ],
            outputs: vec![
                TensorMetadata {
                    name: "tensor_metadata_output1".to_string(),
                    datatype: "float32".to_string(),
                    shape: vec![1, 1000],
                },
                TensorMetadata {
                    name: "tensor_metadata_output2".to_string(),
                    datatype: "int64".to_string(),
                    shape: vec![1],
                },
            ],
        };

        Ok(Response::new(reply))
    }

    async fn model_infer_async(
        &self,
        request: Request<tonic::Streaming<ModelInferRequest>>,
    ) -> Result<Response<Self::ModelInferAsyncStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = mpsc::channel(4);

        let model_manager = self.model_manager.clone();

        tokio::spawn(async move {
            while let Some(message) = stream.message().await.transpose() {
                match message {
                    Ok(req) => {
                        let model_id = ModelId(req.id.clone());

                        let parameters = req
                            .parameters
                            .into_iter()
                            .map(|(k, v)| (k, InferParameter::from(v)))
                            .collect::<HashMap<_, _>>();

                        let inference_request = InferenceRequest {
                            model_name: req.model_name.clone(),
                            model_version: Some(req.model_version.clone()),
                            id: req.id.clone(),
                            parameters: Some(parameters),
                            outputs: None,
                        };

                        model_manager.add_request(model_id, inference_request);

                        // ACK/dummy responses if needed
                        let response = ModelInferResponse {
                            model_name: req.model_name,
                            model_version: req.model_version,
                            id: req.id,
                            parameters: HashMap::new(),
                            outputs: vec![],
                            raw_output_contents: vec![],
                        };
                        if let Err(e) = tx.send(Ok(response)).await {
                            eprintln!("Error sending response: {:?}", e);
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("Error reading stream: {:?}", e);
                        break;
                    }
                }
            }
        });

        Ok(Response::new(
            Box::pin(ReceiverStream::new(rx)) as Self::ModelInferAsyncStream
        ))
    }

    async fn model_infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        println!("Got a request: {:?}", request);

        let req = request.into_inner();
        let model_id = ModelId(req.id.clone());

        let domain_params = req
            .parameters
            .into_iter()
            .map(|(k, v)| (k, InferParameter::from(v)))
            .collect::<HashMap<_, _>>();

        let inference_request = InferenceRequest {
            model_name: req.model_name.clone(),
            model_version: Some(req.model_version.clone()),
            id: req.id.clone(),
            parameters: Some(domain_params),
            outputs: None, // or map req.outputs if needed
        };

        // Enqueue into ModelManager
        self.model_manager.add_request(model_id, inference_request);

        let reply = ModelInferResponse {
            model_name: req.model_name,
            model_version: req.model_version,
            id: req.id,
            parameters: HashMap::new(),
            outputs: vec![],
            raw_output_contents: vec![],
        };

        Ok(Response::new(reply))
    }

    async fn unified_infer(
        &self,
        request: Request<UnifiedInferRequest>,
    ) -> Result<Response<UnifiedInferResponse>, Status> {
        let req = request.into_inner();
        let start_time = std::time::Instant::now();

        // Handle backward compatibility
        if let Some(legacy_req) = &req.legacy_request {
            // Use legacy processing for backward compatibility
            let legacy_response = self.process_legacy_request(legacy_req.clone()).await?;

            let response = UnifiedInferResponse {
                protocol: req.protocol,
                legacy_response: Some(legacy_response.clone()),
                content: Some(MessageContent {
                    content_type: ContentType::Text as i32,
                    content: Some(grpc_server::message_content::Content::TextContent(format!(
                        "Legacy response for model: {}",
                        legacy_response.model_name
                    ))),
                }),
                stream_metadata: None,
                model_name: req.model_name,
                model_version: req.model_version,
                request_id: req.request_id,
                status: Some(ResponseStatus {
                    code: grpc_server::response_status::StatusCode::StatusSuccess as i32,
                    message: "Success".to_string(),
                    details: std::collections::HashMap::new(),
                }),
                parameters: std::collections::HashMap::new(),
                metadata: std::collections::HashMap::new(),
                metrics: Some(PerformanceMetrics {
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                    queue_time_ms: 0,
                    token_usage: None,
                    memory_usage_bytes: None,
                    gpu_utilization: None,
                }),
            };

            return Ok(Response::new(response));
        }

        // Process based on protocol
        let response_content = match req.protocol() {
            InferenceProtocol::ProtocolOpenai => self.process_openai_request(&req).await?,
            InferenceProtocol::ProtocolGalemind | InferenceProtocol::ProtocolUnspecified => {
                self.process_galemind_request(&req).await?
            }
        };

        let response = UnifiedInferResponse {
            protocol: req.protocol,
            legacy_response: None,
            content: Some(response_content),
            stream_metadata: req.stream_metadata,
            model_name: req.model_name,
            model_version: req.model_version,
            request_id: req.request_id,
            status: Some(ResponseStatus {
                code: grpc_server::response_status::StatusCode::StatusSuccess as i32,
                message: "Success".to_string(),
                details: std::collections::HashMap::new(),
            }),
            parameters: std::collections::HashMap::new(),
            metadata: std::collections::HashMap::new(),
            metrics: Some(PerformanceMetrics {
                processing_time_ms: start_time.elapsed().as_millis() as u64,
                queue_time_ms: 0,
                token_usage: Some(TokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: 20,
                    total_tokens: 30,
                }),
                memory_usage_bytes: Some(1024 * 1024),
                gpu_utilization: Some(75.5),
            }),
        };

        Ok(Response::new(response))
    }

    async fn unified_infer_stream(
        &self,
        request: Request<tonic::Streaming<UnifiedInferRequest>>,
    ) -> Result<Response<Self::UnifiedInferStreamStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = mpsc::channel(4);
        let _model_manager = self.model_manager.clone();

        tokio::spawn(async move {
            let mut stream_sessions: std::collections::HashMap<String, Vec<UnifiedInferRequest>> =
                std::collections::HashMap::new();

            while let Some(message) = stream.message().await.transpose() {
                match message {
                    Ok(req) => {
                        let start_time = std::time::Instant::now();

                        // Handle streaming logic
                        if let Some(stream_meta) = &req.stream_metadata {
                            let stream_id = stream_meta.stream_id.clone();

                            // Store chunk in session
                            stream_sessions
                                .entry(stream_id.clone())
                                .or_insert_with(Vec::new)
                                .push(req.clone());

                            // If end of stream, process complete message
                            if stream_meta.end_of_stream {
                                if let Some(chunks) = stream_sessions.remove(&stream_id) {
                                    let combined_content = Self::combine_stream_chunks(&chunks);

                                    let response = UnifiedInferResponse {
                                        protocol: req.protocol,
                                        legacy_response: None,
                                        content: Some(combined_content),
                                        stream_metadata: Some(StreamMetadata {
                                            stream_id: stream_id.clone(),
                                            chunk_sequence: stream_meta.chunk_sequence,
                                            is_streaming: true,
                                            end_of_stream: true,
                                            total_chunks: stream_meta.total_chunks,
                                        }),
                                        model_name: req.model_name,
                                        model_version: req.model_version,
                                        request_id: req.request_id,
                                        status: Some(ResponseStatus {
                                            code: grpc_server::response_status::StatusCode::StatusSuccess as i32,
                                            message: "Stream completed".to_string(),
                                            details: std::collections::HashMap::new(),
                                        }),
                                        parameters: std::collections::HashMap::new(),
                                        metadata: std::collections::HashMap::new(),
                                        metrics: Some(PerformanceMetrics {
                                            processing_time_ms: start_time.elapsed().as_millis() as u64,
                                            queue_time_ms: 0,
                                            token_usage: Some(TokenUsage {
                                                prompt_tokens: 15,
                                                completion_tokens: 25,
                                                total_tokens: 40,
                                            }),
                                            memory_usage_bytes: Some(2048 * 1024),
                                            gpu_utilization: Some(80.0),
                                        }),
                                    };

                                    if let Err(e) = tx.send(Ok(response)).await {
                                        eprintln!("Error sending stream response: {:?}", e);
                                        break;
                                    }
                                }
                            } else {
                                // Send acknowledgment for chunk received
                                let ack_response = UnifiedInferResponse {
                                    protocol: req.protocol,
                                    legacy_response: None,
                                    content: Some(MessageContent {
                                        content_type: ContentType::Text as i32,
                                        content: Some(
                                            grpc_server::message_content::Content::TextContent(
                                                format!(
                                                    "Chunk {} received",
                                                    stream_meta.chunk_sequence
                                                ),
                                            ),
                                        ),
                                    }),
                                    stream_metadata: Some(StreamMetadata {
                                        stream_id: stream_id.clone(),
                                        chunk_sequence: stream_meta.chunk_sequence,
                                        is_streaming: true,
                                        end_of_stream: false,
                                        total_chunks: stream_meta.total_chunks,
                                    }),
                                    model_name: req.model_name,
                                    model_version: req.model_version,
                                    request_id: req.request_id,
                                    status: Some(ResponseStatus {
                                        code:
                                            grpc_server::response_status::StatusCode::StatusSuccess
                                                as i32,
                                        message: "Chunk received".to_string(),
                                        details: std::collections::HashMap::new(),
                                    }),
                                    parameters: std::collections::HashMap::new(),
                                    metadata: std::collections::HashMap::new(),
                                    metrics: Some(PerformanceMetrics {
                                        processing_time_ms: start_time.elapsed().as_millis() as u64,
                                        queue_time_ms: 0,
                                        token_usage: None,
                                        memory_usage_bytes: None,
                                        gpu_utilization: None,
                                    }),
                                };

                                if let Err(e) = tx.send(Ok(ack_response)).await {
                                    eprintln!("Error sending chunk ack: {:?}", e);
                                    break;
                                }
                            }
                        } else {
                            // Non-streaming request
                            let response_content = match req.protocol() {
                                InferenceProtocol::ProtocolOpenai => MessageContent {
                                    content_type: ContentType::Text as i32,
                                    content: Some(
                                        grpc_server::message_content::Content::TextContent(
                                            "OpenAI protocol response".to_string(),
                                        ),
                                    ),
                                },
                                _ => MessageContent {
                                    content_type: ContentType::Text as i32,
                                    content: Some(
                                        grpc_server::message_content::Content::TextContent(
                                            "Galemind protocol response".to_string(),
                                        ),
                                    ),
                                },
                            };

                            let response = UnifiedInferResponse {
                                protocol: req.protocol,
                                legacy_response: None,
                                content: Some(response_content),
                                stream_metadata: None,
                                model_name: req.model_name,
                                model_version: req.model_version,
                                request_id: req.request_id,
                                status: Some(ResponseStatus {
                                    code: grpc_server::response_status::StatusCode::StatusSuccess
                                        as i32,
                                    message: "Success".to_string(),
                                    details: std::collections::HashMap::new(),
                                }),
                                parameters: std::collections::HashMap::new(),
                                metadata: std::collections::HashMap::new(),
                                metrics: Some(PerformanceMetrics {
                                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                                    queue_time_ms: 0,
                                    token_usage: Some(TokenUsage {
                                        prompt_tokens: 12,
                                        completion_tokens: 18,
                                        total_tokens: 30,
                                    }),
                                    memory_usage_bytes: Some(1536 * 1024),
                                    gpu_utilization: Some(70.0),
                                }),
                            };

                            if let Err(e) = tx.send(Ok(response)).await {
                                eprintln!("Error sending response: {:?}", e);
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Error reading stream: {:?}", e);
                        break;
                    }
                }
            }
        });

        Ok(Response::new(
            Box::pin(ReceiverStream::new(rx)) as Self::UnifiedInferStreamStream
        ))
    }
}

impl PredictionServiceImpl {
    async fn process_legacy_request(
        &self,
        req: ModelInferRequest,
    ) -> Result<ModelInferResponse, Status> {
        // Convert to domain request and process
        let model_id = ModelId(req.id.clone());
        let domain_params = req
            .parameters
            .into_iter()
            .map(|(k, v)| (k, InferParameter::from(v)))
            .collect::<HashMap<_, _>>();

        let inference_request = InferenceRequest {
            model_name: req.model_name.clone(),
            model_version: Some(req.model_version.clone()),
            id: req.id.clone(),
            parameters: Some(domain_params),
            outputs: None,
        };

        self.model_manager.add_request(model_id, inference_request);

        Ok(ModelInferResponse {
            model_name: req.model_name,
            model_version: req.model_version,
            id: req.id,
            parameters: HashMap::new(),
            outputs: vec![],
            raw_output_contents: vec![],
        })
    }

    async fn process_openai_request(
        &self,
        req: &UnifiedInferRequest,
    ) -> Result<MessageContent, Status> {
        // Process OpenAI-style request
        if let Some(content) = &req.content {
            match content.content_type() {
                ContentType::Text => {
                    if let Some(grpc_server::message_content::Content::TextContent(text)) =
                        &content.content
                    {
                        return Ok(MessageContent {
                            content_type: ContentType::Text as i32,
                            content: Some(grpc_server::message_content::Content::TextContent(
                                format!("OpenAI response to: {}", text),
                            )),
                        });
                    }
                }
                ContentType::Base64 => {
                    if let Some(grpc_server::message_content::Content::Base64Content(data)) =
                        &content.content
                    {
                        return Ok(MessageContent {
                            content_type: ContentType::Base64 as i32,
                            content: Some(grpc_server::message_content::Content::Base64Content(
                                format!("processed_{}", data),
                            )),
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(MessageContent {
            content_type: ContentType::Text as i32,
            content: Some(grpc_server::message_content::Content::TextContent(
                "Default OpenAI response".to_string(),
            )),
        })
    }

    async fn process_galemind_request(
        &self,
        req: &UnifiedInferRequest,
    ) -> Result<MessageContent, Status> {
        // Process Galemind-style request
        if let Some(content) = &req.content {
            match content.content_type() {
                ContentType::Binary => {
                    if let Some(grpc_server::message_content::Content::BinaryContent(data)) =
                        &content.content
                    {
                        return Ok(MessageContent {
                            content_type: ContentType::Binary as i32,
                            content: Some(grpc_server::message_content::Content::BinaryContent(
                                [b"processed_", data.as_slice()].concat(),
                            )),
                        });
                    }
                }
                ContentType::Text => {
                    if let Some(grpc_server::message_content::Content::TextContent(text)) =
                        &content.content
                    {
                        return Ok(MessageContent {
                            content_type: ContentType::Text as i32,
                            content: Some(grpc_server::message_content::Content::TextContent(
                                format!("Galemind response to: {}", text),
                            )),
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(MessageContent {
            content_type: ContentType::Text as i32,
            content: Some(grpc_server::message_content::Content::TextContent(
                "Default Galemind response".to_string(),
            )),
        })
    }

    fn combine_stream_chunks(chunks: &[UnifiedInferRequest]) -> MessageContent {
        let mut combined_text = String::new();
        let mut combined_binary = Vec::new();
        let mut content_type = ContentType::Text;

        for chunk in chunks {
            if let Some(content) = &chunk.content {
                match &content.content {
                    Some(grpc_server::message_content::Content::TextContent(text)) => {
                        combined_text.push_str(text);
                        content_type = ContentType::Text;
                    }
                    Some(grpc_server::message_content::Content::BinaryContent(data)) => {
                        combined_binary.extend_from_slice(data);
                        content_type = ContentType::Binary;
                    }
                    Some(grpc_server::message_content::Content::Base64Content(data)) => {
                        combined_text.push_str(data);
                        content_type = ContentType::Base64;
                    }
                    None => {}
                }
            }
        }

        match content_type {
            ContentType::Binary => MessageContent {
                content_type: ContentType::Binary as i32,
                content: Some(grpc_server::message_content::Content::BinaryContent(
                    combined_binary,
                )),
            },
            ContentType::Base64 => MessageContent {
                content_type: ContentType::Base64 as i32,
                content: Some(grpc_server::message_content::Content::Base64Content(
                    combined_text,
                )),
            },
            _ => MessageContent {
                content_type: ContentType::Text as i32,
                content: Some(grpc_server::message_content::Content::TextContent(
                    combined_text,
                )),
            },
        }
    }
}

/// Builder for setting up the gRPC server
pub struct GrpcServerBuilder {
    address: String,
    service_impl: PredictionServiceImpl,
}
/// async trait should applied also to the implementation.
#[async_trait]
impl InferenceServerBuilder for GrpcServerBuilder {
    fn configure(
        context: InferenceServerConfig,
        model_manager: Arc<ModelDiscoveryService>,
    ) -> Self {
        let addr = format!("{}:{}", context.grpc_hostname, context.grpc_port);
        Self {
            address: addr,
            service_impl: PredictionServiceImpl::new(model_manager),
        }
    }
    async fn start(self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let addr = self.address.parse()?;

        println!("gRPC PredictionService server listening on {}", addr);

        Server::builder()
            .add_service(PredictionServiceServer::new(self.service_impl))
            .serve(addr)
            .await?;
        Ok(())
    }
}
