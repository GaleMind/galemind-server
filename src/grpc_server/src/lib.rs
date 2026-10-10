//! The Predict Protocol v2 gRPC transport.
//!
//! Six calls, no extensions: liveness, readiness, model readiness, server
//! metadata, model metadata and inference. The transport decodes a request,
//! hands it to [`galemind_core::Inference`] and encodes the answer; all
//! resolution, caching and batching live behind that trait.

#![forbid(unsafe_code)]

mod convert;

use convert::{metadata_to_proto, request_from_proto, response_to_proto, to_status};
use galemind_core::{Inference, ModelRef, SERVER_NAME, SERVER_VERSION};
use std::net::SocketAddr;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{debug, info};

/// The generated Predict Protocol v2 messages and service stubs.
pub mod pb {
    tonic::include_proto!("inference");
}

pub use pb::grpc_inference_service_client::GrpcInferenceServiceClient;
pub use pb::grpc_inference_service_server::{GrpcInferenceService, GrpcInferenceServiceServer};

/// Serves the protocol over gRPC.
pub struct GrpcPredictV2Service {
    inference: Arc<dyn Inference>,
}

impl GrpcPredictV2Service {
    pub fn new(inference: Arc<dyn Inference>) -> Self {
        Self { inference }
    }

    /// Wraps this service in its generated server, ready to add to a tonic
    /// server builder.
    pub fn into_server(self) -> GrpcInferenceServiceServer<Self> {
        GrpcInferenceServiceServer::new(self)
    }
}

#[tonic::async_trait]
impl GrpcInferenceService for GrpcPredictV2Service {
    async fn server_live(
        &self,
        _request: Request<pb::ServerLiveRequest>,
    ) -> Result<Response<pb::ServerLiveResponse>, Status> {
        // Reaching this point is itself the proof of liveness: the server is
        // receiving and answering requests.
        Ok(Response::new(pb::ServerLiveResponse { live: true }))
    }

    async fn server_ready(
        &self,
        _request: Request<pb::ServerReadyRequest>,
    ) -> Result<Response<pb::ServerReadyResponse>, Status> {
        Ok(Response::new(pb::ServerReadyResponse {
            ready: self.inference.is_ready().await,
        }))
    }

    async fn model_ready(
        &self,
        request: Request<pb::ModelReadyRequest>,
    ) -> Result<Response<pb::ModelReadyResponse>, Status> {
        let request = request.into_inner();
        let model = ModelRef::new(request.name, Some(request.version));
        Ok(Response::new(pb::ModelReadyResponse {
            ready: self.inference.is_model_ready(&model).await,
        }))
    }

    async fn server_metadata(
        &self,
        _request: Request<pb::ServerMetadataRequest>,
    ) -> Result<Response<pb::ServerMetadataResponse>, Status> {
        Ok(Response::new(pb::ServerMetadataResponse {
            name: SERVER_NAME.to_string(),
            version: SERVER_VERSION.to_string(),
            // No protocol extensions are implemented.
            extensions: Vec::new(),
        }))
    }

    async fn model_metadata(
        &self,
        request: Request<pb::ModelMetadataRequest>,
    ) -> Result<Response<pb::ModelMetadataResponse>, Status> {
        let request = request.into_inner();
        let model = ModelRef::new(request.name, Some(request.version));
        let metadata = self.inference.metadata(&model).await.map_err(to_status)?;
        Ok(Response::new(metadata_to_proto(metadata)))
    }

    async fn model_infer(
        &self,
        request: Request<pb::ModelInferRequest>,
    ) -> Result<Response<pb::ModelInferResponse>, Status> {
        let request = request.into_inner();
        debug!(
            model = %request.model_name,
            inputs = request.inputs.len(),
            "gRPC inference request"
        );

        let decoded = request_from_proto(request).map_err(to_status)?;
        let response = self.inference.infer(decoded).await.map_err(to_status)?;
        Ok(Response::new(response_to_proto(response)))
    }
}

/// Runs the gRPC server until `shutdown` resolves.
pub async fn serve(
    address: SocketAddr,
    inference: Arc<dyn Inference>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!(%address, "gRPC server listening");
    tonic::transport::Server::builder()
        .add_service(GrpcPredictV2Service::new(inference).into_server())
        .serve_with_shutdown(address, shutdown)
        .await?;
    info!("gRPC server stopped");
    Ok(())
}
