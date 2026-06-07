mod data_model;
mod healthcheck;
mod metadata_model;
mod model;
mod openai_models;
mod protocol;
mod server;
mod unified_endpoints;

use crate::healthcheck::new_health_check_router;
use crate::model::new_model_router;
use crate::server::new_server_router;
use crate::unified_endpoints::new_unified_router;
use anyhow::Result;
use async_trait::async_trait;
use axum::{serve, Router};
use foundation::{InferenceServerBuilder, InferenceServerConfig, ModelDiscoveryService};
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;

pub struct RestServerBuilder {
    addr: SocketAddr,
    app: Router,
}

#[async_trait]
impl InferenceServerBuilder for RestServerBuilder {
    fn configure(
        context: InferenceServerConfig,
        model_manager: Arc<ModelDiscoveryService>,
    ) -> Self {
        let addr = format!("{}:{}", context.rest_hostname, context.rest_port)
            .parse()
            .expect("Invalid Host/Port");
        let app = Router::new()
            .nest("/{version}", new_server_router())
            .nest("/{version}/health", new_health_check_router())
            .nest("/{version}/models", new_model_router(model_manager.clone()))
            .nest("/v1", new_unified_router(model_manager.clone()))
            .layer(TraceLayer::new_for_http());

        Self { addr, app }
    }

    async fn start(self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let listener = TcpListener::bind(self.addr)
            .await
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;

        let local_addr = listener.local_addr()?;
        println!("Rest Server listening on {}", local_addr);
        serve(listener, self.app)
            .await
            .map_err(|e| Box::<dyn Error + Send + Sync>::from(e.to_string()))?;

        Ok(())
    }
}
