use super::buffer_events::{BufferEvent, BufferEventEmitter, create_buffer_event_channel};
use super::inference_buffer::InferenceBuffer;
use crate::api::inference::{InferenceRequest, InferenceResponse};
use crate::api::inference_runtime::InferenceRuntime;
use anyhow::{Result, anyhow};
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::oneshot;
use tokio::task;

/// Represents a pending inference request with a response channel
pub struct PendingInferenceRequest {
    pub request: InferenceRequest,
    pub response_tx: oneshot::Sender<InferenceResponse>,
}

impl std::fmt::Debug for PendingInferenceRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingInferenceRequest")
            .field("request", &self.request)
            .field("response_tx", &"<oneshot::Sender>")
            .finish()
    }
}

/// Per-model context with buffer and runtime
pub struct ModelContext {
    buffer: InferenceBuffer,
    runtime: Arc<dyn InferenceRuntime>,
    pending_requests: Vec<PendingInferenceRequest>,
}

impl ModelContext {
    pub fn new(
        runtime: Arc<dyn InferenceRuntime>,
        buffer_capacity: usize,
        threshold_percentage: f32,
        event_emitter: BufferEventEmitter,
    ) -> Self {
        let model_id = runtime.model_id().to_string();
        let buffer = InferenceBuffer::new(
            buffer_capacity,
            model_id,
            threshold_percentage,
            Some(event_emitter),
        );

        Self {
            buffer,
            runtime,
            pending_requests: Vec::new(),
        }
    }

    pub fn add_request(&mut self, pending_request: PendingInferenceRequest) {
        // Add request to buffer for batching consideration
        self.buffer.push(pending_request.request.clone());

        // Store pending request for response handling
        self.pending_requests.push(pending_request);
    }

    pub fn get_buffer_info(&self) -> (usize, usize, f32) {
        (self.buffer.len(), self.buffer.capacity(), self.buffer.fill_percentage())
    }

    pub fn drain_buffer_contents(&mut self) -> Vec<InferenceRequest> {
        self.buffer.drain_contents()
    }

    pub fn take_pending_requests(&mut self) -> Vec<PendingInferenceRequest> {
        std::mem::take(&mut self.pending_requests)
    }
}

/// Event-driven Model Manager that responds to buffer events
pub struct EventDrivenModelManager {
    models: DashMap<String, ModelContext>,
    event_emitter: BufferEventEmitter,
    default_buffer_capacity: usize,
    default_threshold_percentage: f32,
}

impl EventDrivenModelManager {
    pub fn new() -> Self {
        let (event_emitter, mut event_receiver) = create_buffer_event_channel();

        // Spawn event handler task
        let models_ref = Arc::new(DashMap::new());
        let models_clone = models_ref.clone();

        task::spawn(async move {
            while let Some(event) = event_receiver.recv().await {
                Self::handle_buffer_event(event, &models_clone).await;
            }
        });

        Self {
            models: DashMap::new(),
            event_emitter,
            default_buffer_capacity: 100,
            default_threshold_percentage: 80.0,
        }
    }

    /// Handle buffer events (threshold reached, buffer full, etc.)
    async fn handle_buffer_event(
        event: BufferEvent,
        models: &DashMap<String, ModelContext>,
    ) {
        match event {
            BufferEvent::ThresholdReached {
                model_id,
                current_size,
                capacity,
                fill_percentage,
            } => {
                println!(
                    "🚨 Model '{}' buffer reached {}% threshold ({}/{} items)",
                    model_id, fill_percentage, current_size, capacity
                );

                // Trigger offloading for this model
                if let Some(mut model_entry) = models.get_mut(&model_id) {
                    Self::trigger_offloading(&model_id, &mut model_entry).await;
                }
            }

            BufferEvent::BufferFull {
                model_id,
                buffer_contents,
                buffer_capacity,
            } => {
                println!(
                    "💾 Model '{}' buffer is full ({} items), triggering immediate offloading",
                    model_id, buffer_capacity
                );

                // For buffer full, we immediately process the contents
                if let Some(mut model_entry) = models.get_mut(&model_id) {
                    Self::process_buffer_contents(
                        &model_id,
                        buffer_contents,
                        &model_entry.runtime,
                    ).await;
                }
            }

            BufferEvent::BufferStats {
                model_id,
                current_size,
                capacity,
                fill_percentage,
            } => {
                println!(
                    "📊 Model '{}' buffer stats: {}/{} items ({}%)",
                    model_id, current_size, capacity, fill_percentage
                );
            }
        }
    }

    /// Trigger offloading for a specific model
    async fn trigger_offloading(model_id: &str, model_context: &mut ModelContext) {
        let buffer_contents = model_context.drain_buffer_contents();
        let pending_requests = model_context.take_pending_requests();

        if !buffer_contents.is_empty() {
            println!(
                "🔄 Offloading {} requests for model '{}' to inference runtime",
                buffer_contents.len(),
                model_id
            );

            // Process batch with the runtime
            let runtime = model_context.runtime.clone();
            Self::process_batch_with_responses(buffer_contents, pending_requests, runtime).await;
        }
    }

    /// Process buffer contents with the inference runtime
    async fn process_buffer_contents(
        model_id: &str,
        buffer_contents: Vec<InferenceRequest>,
        runtime: &Arc<dyn InferenceRuntime>,
    ) {
        if !buffer_contents.is_empty() {
            println!(
                "⚡ Processing {} requests for model '{}' via inference runtime",
                buffer_contents.len(),
                model_id
            );

            let responses = runtime.process_batch(buffer_contents).await;
            println!(
                "✅ Completed batch processing for model '{}', got {} responses",
                model_id,
                responses.len()
            );
        }
    }

    /// Process batch and send responses back through channels
    async fn process_batch_with_responses(
        requests: Vec<InferenceRequest>,
        pending_requests: Vec<PendingInferenceRequest>,
        runtime: Arc<dyn InferenceRuntime>,
    ) {
        let responses = runtime.process_batch(requests).await;

        // Send responses back through the channels
        for (pending, response) in pending_requests.into_iter().zip(responses.into_iter()) {
            if let Err(_) = pending.response_tx.send(response) {
                eprintln!("Failed to send response back to caller");
            }
        }
    }

    pub fn register_model(&self, runtime: Arc<dyn InferenceRuntime>) -> Result<()> {
        let model_id = runtime.model_id().to_string();

        let model_context = ModelContext::new(
            runtime,
            self.default_buffer_capacity,
            self.default_threshold_percentage,
            self.event_emitter.clone(),
        );

        self.models.insert(model_id.clone(), model_context);
        println!("📝 Registered model '{}' with event-driven buffer", model_id);
        Ok(())
    }

    pub async fn process_inference(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let model_id = &request.model_name;

        // Check if model is registered
        if !self.models.contains_key(model_id) {
            return Err(anyhow!("Model '{}' not found", model_id));
        }

        // Create response channel
        let (response_tx, response_rx) = oneshot::channel();

        // Create pending request
        let pending_request = PendingInferenceRequest {
            request: request.clone(),
            response_tx,
        };

        // Add to model's buffer (this will trigger events automatically)
        {
            let mut model_entry = self.models.get_mut(model_id)
                .ok_or_else(|| anyhow!("Model '{}' not found", model_id))?;
            model_entry.add_request(pending_request);
        }

        // For immediate response, also process directly (non-batched)
        let model_entry = self.models.get(model_id)
            .ok_or_else(|| anyhow!("Model '{}' not found", model_id))?;

        let response = model_entry.runtime.process_single(request).await;
        Ok(response)
    }

    pub fn get_model_stats(&self) -> Vec<(String, usize, usize, f32)> {
        self.models
            .iter()
            .map(|entry| {
                let model_id = entry.key().clone();
                let (current_len, capacity, fill_percentage) = entry.value().get_buffer_info();
                (model_id, current_len, capacity, fill_percentage)
            })
            .collect()
    }

    pub fn set_buffer_config(&mut self, capacity: usize, threshold_percentage: f32) -> Result<()> {
        if threshold_percentage < 0.0 || threshold_percentage > 100.0 {
            return Err(anyhow!("Threshold percentage must be between 0 and 100"));
        }

        self.default_buffer_capacity = capacity;
        self.default_threshold_percentage = threshold_percentage;
        Ok(())
    }
}