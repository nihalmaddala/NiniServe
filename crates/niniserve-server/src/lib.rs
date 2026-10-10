#![forbid(unsafe_code)]

use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderValue, StatusCode, header},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use niniserve_engine::{EngineHandle, EngineSubmitError, GenerationStream};
use niniserve_protocol::{
    FinishReason, GenerationEvent, GenerationRequest, RequestId, RequestLimits,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

const MODEL_ID: &str = "local-gguf";

#[derive(Debug, Clone)]
struct AppState {
    engine: EngineHandle,
    limits: RequestLimits,
    next_request_id: Arc<AtomicU64>,
}

pub fn router(engine: EngineHandle, limits: RequestLimits) -> Router {
    let state = AppState {
        engine,
        limits,
        next_request_id: Arc::new(AtomicU64::new(1)),
    };
    Router::new()
        .route("/healthz", get(health))
        .route("/v1/completions", post(completions))
        .with_state(state)
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    model_loaded: bool,
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let ready = state.engine.is_ready();
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(HealthResponse {
            status: if ready { "ok" } else { "not_ready" },
            model_loaded: ready,
        }),
    )
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: String,
    prompt: String,
    max_tokens: u32,
    #[serde(default)]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    stream: bool,
}

const fn default_top_p() -> f32 {
    1.0
}

async fn completions(
    State(state): State<AppState>,
    Json(wire): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    if wire.model != MODEL_ID {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("unknown model {:?}", wire.model),
        ));
    }
    if !wire.stream {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "only stream=true is supported",
        ));
    }

    let id = RequestId(state.next_request_id.fetch_add(1, Ordering::Relaxed));
    let request = GenerationRequest {
        id,
        prompt: wire.prompt,
        max_new_tokens: wire.max_tokens,
        temperature: wire.temperature,
        top_p: wire.top_p,
        seed: None,
    };
    state
        .limits
        .validate(&request)
        .map_err(|error| ApiError::new(StatusCode::BAD_REQUEST, error.to_string()))?;
    let engine_events = state
        .engine
        .try_generate(request)
        .map_err(ApiError::from_engine)?;
    let (sse_events, receiver) = mpsc::channel(8);
    tokio::spawn(forward_sse(engine_events, sse_events, wire.model));

    let mut response = Sse::new(ReceiverStream::new(receiver)).into_response();
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    Ok(response)
}

async fn forward_sse(
    mut engine_events: GenerationStream,
    output: mpsc::Sender<Result<Event, Infallible>>,
    model: String,
) {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    while let Some(event) = engine_events.recv().await {
        let (sse, terminal, done) = match event {
            GenerationEvent::Started { .. } => continue,
            GenerationEvent::Token {
                request_id, text, ..
            } => {
                if text.is_empty() {
                    continue;
                }
                (
                    completion_event(request_id, &model, created, text, None),
                    false,
                    false,
                )
            }
            GenerationEvent::Completed {
                request_id,
                finish_reason,
            } => (
                completion_event(
                    request_id,
                    &model,
                    created,
                    String::new(),
                    Some(match finish_reason {
                        FinishReason::Stop => "stop",
                        FinishReason::Length => "length",
                    }),
                ),
                true,
                true,
            ),
            GenerationEvent::Cancelled { request_id } => (
                completion_event(
                    request_id,
                    &model,
                    created,
                    String::new(),
                    Some("cancelled"),
                ),
                true,
                true,
            ),
            GenerationEvent::TimedOut { request_id } => (
                completion_event(request_id, &model, created, String::new(), Some("timeout")),
                true,
                true,
            ),
            GenerationEvent::Error {
                request_id,
                message,
            } => (error_event(request_id, message), true, false),
        };
        if output.send(Ok(sse)).await.is_err() {
            return;
        }
        if terminal {
            if done {
                let _ = output.send(Ok(Event::default().data("[DONE]"))).await;
            }
            return;
        }
    }
}

fn completion_event(
    request_id: RequestId,
    model: &str,
    created: u64,
    text: String,
    finish_reason: Option<&'static str>,
) -> Event {
    Event::default()
        .json_data(CompletionChunk {
            id: format!("req_{}", request_id.0),
            object: "text_completion",
            created,
            model,
            choices: [CompletionChoice {
                index: 0,
                text,
                finish_reason,
            }],
        })
        .expect("completion SSE payload is serializable")
}

fn error_event(request_id: RequestId, message: String) -> Event {
    Event::default()
        .event("error")
        .json_data(StreamError {
            id: format!("req_{}", request_id.0),
            error: message,
        })
        .expect("error SSE payload is serializable")
}

#[derive(Debug, Serialize)]
struct CompletionChunk<'a> {
    id: String,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: [CompletionChoice; 1],
}

#[derive(Debug, Serialize)]
struct CompletionChoice {
    index: u32,
    text: String,
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct StreamError {
    id: String,
    error: String,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn from_engine(error: EngineSubmitError) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, error.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorResponse {
                error: self.message,
            }),
        )
            .into_response()
    }
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[cfg(test)]
mod tests {
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode, header},
    };
    use niniserve_backend::{
        BackendError, BackendLimits, BackendTokenEvent, ExecutionPlan, MockExecutor, ModelExecutor,
    };
    use niniserve_engine::{EngineConfig, EngineHandle, SchedulerConfig};
    use niniserve_protocol::{RequestLimits, SequenceId};
    use tower::ServiceExt;

    use super::router;

    fn app() -> axum::Router {
        let engine = EngineHandle::spawn(
            MockExecutor::new(BackendLimits {
                max_batch_tokens: 64,
                max_active_sequences: 1,
                max_sequence_tokens: 128,
            }),
            EngineConfig {
                command_capacity: 2,
                pending_capacity: 2,
                event_capacity: 8,
                request_timeout: std::time::Duration::from_secs(30),
                scheduler: SchedulerConfig::DecodePriority,
            },
        );
        router(
            engine,
            RequestLimits {
                max_prompt_bytes: 1_024,
                max_new_tokens: 16,
            },
        )
    }

    #[tokio::test]
    async fn health_and_streaming_completion_are_observable_through_http() {
        let health = app()
            .oneshot(Request::get("/healthz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);
        assert_eq!(
            to_bytes(health.into_body(), 1_024).await.unwrap(),
            r#"{"status":"ok","model_loaded":true}"#
        );

        let response = app()
            .oneshot(
                Request::post("/v1/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        r#"{"model":"local-gguf","prompt":"A","max_tokens":3,"temperature":0.0,"stream":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-cache");
        let body = to_bytes(response.into_body(), 8_192).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains(r#""text":"<69>""#));
        assert!(body.contains(r#""finish_reason":"length""#));
        assert!(body.ends_with("data: [DONE]\n\n"));
    }

    #[tokio::test]
    async fn non_streaming_requests_are_rejected_before_engine_submission() {
        let response = app()
            .oneshot(
                Request::post("/v1/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        r#"{"model":"local-gguf","prompt":"A","max_tokens":3,"temperature":0.0,"stream":false}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            to_bytes(response.into_body(), 1_024).await.unwrap(),
            r#"{"error":"only stream=true is supported"}"#
        );
    }

    #[tokio::test]
    async fn backend_errors_after_headers_are_explicit_sse_errors_without_done() {
        let engine = EngineHandle::spawn(
            FailingExecutor,
            EngineConfig {
                command_capacity: 1,
                pending_capacity: 1,
                event_capacity: 4,
                request_timeout: std::time::Duration::from_secs(30),
                scheduler: SchedulerConfig::DecodePriority,
            },
        );
        let response = router(
            engine,
            RequestLimits {
                max_prompt_bytes: 1_024,
                max_new_tokens: 16,
            },
        )
        .oneshot(
            Request::post("/v1/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"model":"local-gguf","prompt":"A","max_tokens":3,"temperature":0.0,"stream":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 8_192).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.starts_with("event: error\n"));
        assert!(body.contains("synthetic decode failure"));
        assert!(!body.contains("[DONE]"));
    }

    struct FailingExecutor;

    impl ModelExecutor for FailingExecutor {
        fn tokenize(&self, _prompt: &str) -> Result<Vec<u32>, BackendError> {
            Ok(vec![1])
        }

        fn execute(
            &mut self,
            _plan: &ExecutionPlan,
        ) -> Result<Vec<BackendTokenEvent>, BackendError> {
            Err(BackendError::Native("synthetic decode failure".to_owned()))
        }

        fn release_sequence(&mut self, _sequence_id: SequenceId) -> Result<(), BackendError> {
            Ok(())
        }

        fn limits(&self) -> BackendLimits {
            BackendLimits {
                max_batch_tokens: 1,
                max_active_sequences: 1,
                max_sequence_tokens: 8,
            }
        }
    }
}
