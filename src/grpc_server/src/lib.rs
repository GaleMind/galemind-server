//! The Predict Protocol v2 gRPC transport.
//!
//! Six calls, no extensions: liveness, readiness, model readiness, server
//! metadata, model metadata and inference. The transport decodes a request,
//! hands it to [`galemind_core::Inference`] and encodes the answer; all
//! resolution, caching and batching live behind that trait.

#![forbid(unsafe_code)]

pub mod auth;
pub mod convert;

use auth::AuthLayer;
use convert::{metadata_to_proto, request_from_proto, response_to_proto, to_status};
use galemind_core::auth::{Authenticator, NoAuth};
use galemind_core::{request_id, Inference, ModelRef, SERVER_NAME, SERVER_VERSION};
use galemind_service::{bind_with_backlog, DEFAULT_LISTEN_BACKLOG};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tonic::metadata::MetadataMap;
use tonic::transport::server::TcpIncoming;
use tonic::{Request, Response, Status};
use tracing::{debug, field::Empty, info, warn, Instrument};

/// The metadata key a request id is read from, matching the HTTP header the
/// REST transport uses, so one id follows a request whichever transport it
/// arrived on.
pub const REQUEST_ID_KEY: &str = "x-request-id";

/// The id to log this call under: the caller's, or a new one.
///
/// A gRPC caller can supply an id two ways -- request metadata, which a
/// gateway or an interceptor sets, or the protocol's own `id` field on an
/// inference request. Metadata wins, because it is the one that spans every
/// call rather than only inference.
fn request_id_of(metadata: &MetadataMap, protocol_id: Option<&str>) -> String {
    let from_metadata = metadata
        .get(REQUEST_ID_KEY)
        .and_then(|value| value.to_str().ok());
    match from_metadata {
        Some(id) if request_id::is_usable(id) => id.to_string(),
        _ => request_id::request_id_or_new(protocol_id),
    }
}

/// A span naming one gRPC call, so that everything logged while handling it
/// is attributed to it.
fn call_span(call: &'static str, id: &str) -> tracing::Span {
    tracing::info_span!("grpc", call, id = %id, status = Empty)
}

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
        let span = call_span("model_ready", &request_id_of(request.metadata(), None));
        let request = request.into_inner();

        async move {
            let model = ModelRef::new(request.name, Some(request.version));
            let ready = self.inference.is_model_ready(&model).await;
            debug!(%model, ready, "model readiness checked");
            Ok(Response::new(pb::ModelReadyResponse { ready }))
        }
        .instrument(span)
        .await
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
        let span = call_span("model_metadata", &request_id_of(request.metadata(), None));
        let request = request.into_inner();

        async move {
            let model = ModelRef::new(request.name, Some(request.version));
            debug!(%model, "metadata requested");
            let metadata = self.inference.metadata(&model).await.map_err(to_status)?;
            Ok(Response::new(metadata_to_proto(metadata)))
        }
        .instrument(span)
        .await
    }

    async fn model_infer(
        &self,
        request: Request<pb::ModelInferRequest>,
    ) -> Result<Response<pb::ModelInferResponse>, Status> {
        let id = request_id_of(request.metadata(), Some(request.get_ref().id.as_str()));
        let span = call_span("model_infer", &id);
        let request = request.into_inner();

        async move {
            let at = Instant::now();
            debug!(
                model = %request.model_name,
                version = %request.model_version,
                inputs = request.inputs.len(),
                "request received"
            );

            let decoded = request_from_proto(request).map_err(to_status)?;
            let outcome = self.inference.infer(decoded).await;
            let elapsed_us = at.elapsed().as_micros();

            match outcome {
                Ok(response) => {
                    debug!(
                        elapsed_us,
                        outputs = response.outputs.len(),
                        "request answered"
                    );
                    Ok(Response::new(response_to_proto(response)))
                }
                Err(e) => {
                    let status = to_status(e);
                    tracing::Span::current().record("status", status.code() as i32);
                    // A client error is the caller's to fix and is already in
                    // the metrics; only a fault of the server's own is worth a
                    // line at the default filter.
                    if status.code() == tonic::Code::Internal {
                        warn!(elapsed_us, error = %status.message(), "request failed");
                    } else {
                        debug!(elapsed_us, error = %status.message(), "request refused");
                    }
                    Err(status)
                }
            }
        }
        .instrument(span)
        .await
    }
}

/// Runs the gRPC server until `shutdown` resolves, authenticating nothing.
///
/// Equivalent to [`serve_with_auth`] with [`NoAuth`], which is what the
/// server runs unless a deployment configures otherwise.
pub async fn serve(
    address: SocketAddr,
    inference: Arc<dyn Inference>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    serve_with_auth(
        address,
        inference,
        Arc::new(NoAuth),
        DEFAULT_LISTEN_BACKLOG,
        shutdown,
    )
    .await
}

/// Runs the gRPC server until `shutdown` resolves, checking every call
/// against `auth`.
///
/// The check is a layer in front of the router, so it covers every method of
/// every service on the port, and it is the same [`Authenticator`] the HTTP
/// port is given: one configured provider, both transports. Health and model
/// readiness are exempt, as they are over HTTP; see
/// [`galemind_core::auth::is_open_grpc_method`].
pub async fn serve_with_auth(
    address: SocketAddr,
    inference: Arc<dyn Inference>,
    auth: Arc<dyn Authenticator>,
    backlog: u32,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Bound here rather than by tonic, for the accept queue: tonic's own
    // bind asks the kernel for 1024, which a burst of connections fills
    // before the server sees any of them. See
    // `galemind_service::listener`.
    let listener = bind_with_backlog(address, backlog).await?;
    info!(
        address = %listener.local_addr()?,
        authentication = auth.name(),
        backlog,
        "gRPC server listening"
    );
    tonic::transport::Server::builder()
        .layer(AuthLayer::new(auth))
        .add_service(GrpcPredictV2Service::new(inference).into_server())
        .serve_with_incoming_shutdown(TcpIncoming::from(listener), shutdown)
        .await?;
    info!("gRPC server stopped");
    Ok(())
}
