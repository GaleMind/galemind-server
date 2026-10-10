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
use std::sync::atomic::{AtomicUsize, Ordering};
use tonic::metadata::{Ascii, MetadataValue};
use tonic::transport::Channel;
use tonic::Request;

type Result<T> = std::result::Result<T, InferError>;

/// Requests one channel is expected to carry well.
///
/// A channel multiplexes every request onto one connection, and that
/// connection has one flow-control window, one write loop and one
/// head-of-line to be blocked at -- so a run spreads its requests over
/// several. How many each should take is the same question the multiplexed
/// HTTP client answers, and with the same measured answer, so it is the same
/// constant.
use crate::limits::STREAMS_PER_CONNECTION as REQUESTS_PER_CONNECTION;

// How many connections a client will open however high the concurrency goes
// is [`crate::limits::max_connections`]: a channel per worker would be an
// HTTP/1.1 client with extra steps, and the point of HTTP/2 is that one
// connection carries many requests. It is shared with the multiplexed HTTP
// client, and configured the same way, because it is the same trade-off.

/// Talks the protocol over gRPC.
///
/// The generated client is cheap to clone and shares one connection, so each
/// call clones it rather than holding a lock; concurrent requests then travel
/// as separate HTTP/2 streams on the same channel.
///
/// For a load run the client holds several channels and spreads calls across
/// them, because one connection stops scaling long before the server does;
/// see [`GrpcClient::connect_for_concurrency`].
pub struct GrpcClient {
    /// At least one. Calls are spread across them.
    clients: Vec<GrpcInferenceServiceClient<Channel>>,
    /// Which channel the next call uses. Round-robin rather than
    /// least-loaded: it needs no bookkeeping on the hot path, and every
    /// request here costs about the same.
    next: AtomicUsize,
    /// Sent as `authorization` metadata on every call, for a server that
    /// requires one. Built once: a header value that cannot be made is a
    /// mistake in the token, which belongs at construction rather than on
    /// every request.
    token: Option<MetadataValue<Ascii>>,
}

impl GrpcClient {
    /// `endpoint` is the server's address, for example `http://localhost:8001`.
    pub async fn connect(endpoint: &str) -> Result<Self> {
        Self::connect_channels(endpoint, 1).await
    }

    /// Connects enough channels to carry `concurrency` requests at once.
    pub async fn connect_for_concurrency(endpoint: &str, concurrency: usize) -> Result<Self> {
        Self::connect_channels(endpoint, connections_for(concurrency)).await
    }

    /// Connects enough channels for `concurrency`, but no more than
    /// `max_connections` of them.
    pub async fn connect_under(
        endpoint: &str,
        concurrency: usize,
        max_connections: usize,
    ) -> Result<Self> {
        Self::connect_channels(endpoint, connections_under(concurrency, max_connections)).await
    }

    /// Connects exactly `channels` channels, each its own TCP connection.
    pub async fn connect_channels(endpoint: &str, channels: usize) -> Result<Self> {
        let endpoint = if endpoint.contains("://") {
            endpoint.to_string()
        } else {
            format!("http://{endpoint}")
        };

        let mut clients = Vec::with_capacity(channels.max(1));
        for _ in 0..channels.max(1) {
            // Connected one at a time: a server that refuses the first
            // connection will refuse the rest, and the error should name that
            // rather than arrive as a pile of identical failures.
            let client = GrpcInferenceServiceClient::connect(endpoint.clone())
                .await
                .map_err(|e| InferError::Internal(format!("cannot connect to {endpoint}: {e}")))?;
            clients.push(client);
        }

        Ok(Self {
            clients,
            next: AtomicUsize::new(0),
            token: None,
        })
    }

    /// Authenticates every call with a bearer token.
    ///
    /// gRPC metadata is HTTP/2 headers, so this is the same
    /// `authorization: Bearer <token>` the REST client sends, and the server
    /// checks it against the same authenticator.
    pub fn with_bearer_token(mut self, token: &str) -> Result<Self> {
        let value: MetadataValue<Ascii> = format!("Bearer {token}").parse().map_err(|_| {
            InferError::InvalidRequest(
                "the bearer token contains characters a header cannot carry".to_string(),
            )
        })?;
        self.token = Some(value);
        Ok(self)
    }

    /// How many connections this client holds.
    pub fn connections(&self) -> usize {
        self.clients.len()
    }

    /// Wraps a message in a request carrying this client's credential.
    ///
    /// Every call goes through here, so a token cannot be forgotten on one of
    /// them -- which is the failure that looks like a server bug.
    fn authorized<T>(&self, message: T) -> Request<T> {
        let mut request = Request::new(message);
        if let Some(token) = &self.token {
            request
                .metadata_mut()
                .insert("authorization", token.clone());
        }
        request
    }

    fn clone_client(&self) -> GrpcInferenceServiceClient<Channel> {
        if self.clients.len() == 1 {
            return self.clients[0].clone();
        }
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.clients.len();
        self.clients[index].clone()
    }
}

/// Connections to open for a given concurrency, under the configured
/// ceiling.
fn connections_for(concurrency: usize) -> usize {
    connections_under(concurrency, crate::limits::max_connections())
}

fn connections_under(concurrency: usize, max_connections: usize) -> usize {
    let wanted = concurrency.div_ceil(REQUESTS_PER_CONNECTION);
    // A concurrency of zero still needs something to talk on, and a ceiling
    // of zero is not a reason to have no channel at all.
    wanted.max(1).min(max_connections.max(1))
}

#[async_trait]
impl PredictV2Client for GrpcClient {
    async fn server_live(&self) -> Result<bool> {
        Ok(self
            .clone_client()
            .server_live(self.authorized(pb::ServerLiveRequest {}))
            .await
            .map_err(from_status)?
            .into_inner()
            .live)
    }

    async fn server_ready(&self) -> Result<bool> {
        Ok(self
            .clone_client()
            .server_ready(self.authorized(pb::ServerReadyRequest {}))
            .await
            .map_err(from_status)?
            .into_inner()
            .ready)
    }

    async fn model_ready(&self, model: &ModelRef) -> Result<bool> {
        Ok(self
            .clone_client()
            .model_ready(self.authorized(pb::ModelReadyRequest {
                name: model.name.clone(),
                version: model.version.clone().unwrap_or_default(),
            }))
            .await
            .map_err(from_status)?
            .into_inner()
            .ready)
    }

    async fn server_metadata(&self) -> Result<ServerMetadata> {
        let response = self
            .clone_client()
            .server_metadata(self.authorized(pb::ServerMetadataRequest {}))
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
            .model_metadata(self.authorized(pb::ModelMetadataRequest {
                name: model.name.clone(),
                version: model.version.clone().unwrap_or_default(),
            }))
            .await
            .map_err(from_status)?
            .into_inner();

        metadata_from_proto(response)
    }

    async fn infer(&self, request: InferRequest) -> Result<InferResponse> {
        let response = self
            .clone_client()
            .model_infer(self.authorized(request_to_proto(&request)))
            .await
            .map_err(from_status)?
            .into_inner();

        response_from_proto(response)
    }

    fn transport(&self) -> &'static str {
        "grpc"
    }

    /// However high the concurrency, a gRPC run holds the channels it
    /// connected and no more: the requests are streams on them.
    fn sockets_for(&self, _concurrency: usize) -> usize {
        self.connections()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One connection is right for a handful of requests and wrong for
    /// thousands, so the count follows the concurrency -- up to a ceiling,
    /// past which multiplexing is the point.
    #[test]
    fn connections_follow_the_concurrency_up_to_a_ceiling() {
        let ceiling = crate::limits::DEFAULT_MAX_CONNECTIONS;
        assert_eq!(connections_under(1, ceiling), 1);
        assert_eq!(connections_under(64, ceiling), 1);
        assert_eq!(connections_under(256, ceiling), 1);
        assert_eq!(connections_under(257, ceiling), 2);
        assert_eq!(connections_under(700, ceiling), 3);
        assert_eq!(connections_under(10_000, ceiling), 40);
        assert_eq!(connections_under(100_000, ceiling), 256);
        // A concurrency of zero still needs something to talk on, and so
        // does a ceiling someone set to zero.
        assert_eq!(connections_under(0, ceiling), 1);
        assert_eq!(connections_under(10_000, 0), 1);
        // A ceiling set low is honoured.
        assert_eq!(connections_under(10_000, 4), 4);
    }
}
