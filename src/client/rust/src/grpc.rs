//! The gRPC client.

use crate::PredictV2Client;
use async_trait::async_trait;
use galemind_core::{
    InferError, InferRequest, InferResponse, ModelMetadata, ModelRef, ServerMetadata,
};
use grpc_server::convert::{
    from_status, metadata_from_proto, request_to_proto, response_from_proto,
};
use grpc_server::{pb, GrpcInferenceServiceClient};
use tonic::transport::Channel;

type Result<T> = std::result::Result<T, InferError>;

/// Talks the protocol over gRPC.
///
/// The generated client is cheap to clone and shares one connection, so each
/// call clones it rather than holding a lock; concurrent requests then travel
/// as separate HTTP/2 streams on the same channel.
pub struct GrpcClient {
    client: GrpcInferenceServiceClient<Channel>,
}

impl GrpcClient {
    /// `endpoint` is the server's address, for example `http://localhost:8001`.
    pub async fn connect(endpoint: &str) -> Result<Self> {
        let endpoint = if endpoint.contains("://") {
            endpoint.to_string()
        } else {
            format!("http://{endpoint}")
        };

        let client = GrpcInferenceServiceClient::connect(endpoint.clone())
            .await
            .map_err(|e| InferError::Internal(format!("cannot connect to {endpoint}: {e}")))?;
        Ok(Self { client })
    }

    fn clone_client(&self) -> GrpcInferenceServiceClient<Channel> {
        self.client.clone()
    }
}

#[async_trait]
impl PredictV2Client for GrpcClient {
    async fn server_live(&self) -> Result<bool> {
        Ok(self
            .clone_client()
            .server_live(pb::ServerLiveRequest {})
            .await
            .map_err(from_status)?
            .into_inner()
            .live)
    }

    async fn server_ready(&self) -> Result<bool> {
        Ok(self
            .clone_client()
            .server_ready(pb::ServerReadyRequest {})
            .await
            .map_err(from_status)?
            .into_inner()
            .ready)
    }

    async fn model_ready(&self, model: &ModelRef) -> Result<bool> {
        Ok(self
            .clone_client()
            .model_ready(pb::ModelReadyRequest {
                name: model.name.clone(),
                version: model.version.clone().unwrap_or_default(),
            })
            .await
            .map_err(from_status)?
            .into_inner()
            .ready)
    }

    async fn server_metadata(&self) -> Result<ServerMetadata> {
        let response = self
            .clone_client()
            .server_metadata(pb::ServerMetadataRequest {})
            .await
            .map_err(from_status)?
            .into_inner();

        Ok(ServerMetadata {
            name: response.name,
            version: response.version,
            extensions: response.extensions,
        })
    }

    async fn model_metadata(&self, model: &ModelRef) -> Result<ModelMetadata> {
        let response = self
            .clone_client()
            .model_metadata(pb::ModelMetadataRequest {
                name: model.name.clone(),
                version: model.version.clone().unwrap_or_default(),
            })
            .await
            .map_err(from_status)?
            .into_inner();

        metadata_from_proto(response)
    }

    async fn infer(&self, request: InferRequest) -> Result<InferResponse> {
        let response = self
            .clone_client()
            .model_infer(request_to_proto(&request))
            .await
            .map_err(from_status)?
            .into_inner();

        response_from_proto(response)
    }

    fn transport(&self) -> &'static str {
        "grpc"
    }
}
