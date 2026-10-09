#![forbid(unsafe_code)]

use std::{
    fmt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use niniserve_backend::{
    BackendTokenEvent, BatchToken, ExecutionPlan, ModelExecutor, SamplingConfig,
};
use niniserve_protocol::{FinishReason, GenerationEvent, GenerationRequest, SequenceId};
use tokio::sync::mpsc;

const SINGLE_SEQUENCE: SequenceId = SequenceId(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineConfig {
    pub command_capacity: usize,
    pub event_capacity: usize,
}

#[derive(Debug, Clone)]
pub struct EngineHandle {
    inner: Arc<EngineInner>,
}

#[derive(Debug)]
struct EngineInner {
    commands: Mutex<Option<mpsc::Sender<EngineCommand>>>,
    ready: Arc<AtomicBool>,
    event_capacity: usize,
    worker: Mutex<Option<thread::JoinHandle<()>>>,
}

impl Drop for EngineInner {
    fn drop(&mut self) {
        self.commands
            .get_mut()
            .expect("engine command mutex poisoned")
            .take();
        if let Some(worker) = self
            .worker
            .get_mut()
            .expect("engine worker mutex poisoned")
            .take()
        {
            let _ = worker.join();
        }
    }
}

impl EngineHandle {
    #[must_use]
    pub fn spawn<E>(executor: E, config: EngineConfig) -> Self
    where
        E: ModelExecutor + 'static,
    {
        assert!(
            config.command_capacity > 0,
            "command capacity must be positive"
        );
        assert!(config.event_capacity > 0, "event capacity must be positive");

        let (commands, mut receiver) = mpsc::channel(config.command_capacity);
        let ready = Arc::new(AtomicBool::new(false));
        let worker_ready = Arc::clone(&ready);
        let (started, wait_for_start) = std::sync::mpsc::sync_channel(0);
        let worker = thread::Builder::new()
            .name("niniserve-engine".to_owned())
            .spawn(move || {
                let mut executor = executor;
                worker_ready.store(true, Ordering::Release);
                let _ = started.send(());
                while let Some(command) = receiver.blocking_recv() {
                    match command {
                        EngineCommand::Generate { request, events } => {
                            process_request(&mut executor, request, &events);
                        }
                    }
                }
                worker_ready.store(false, Ordering::Release);
            })
            .expect("failed to spawn the engine worker thread");
        wait_for_start
            .recv()
            .expect("engine worker stopped before becoming ready");

        Self {
            inner: Arc::new(EngineInner {
                commands: Mutex::new(Some(commands)),
                ready,
                event_capacity: config.event_capacity,
                worker: Mutex::new(Some(worker)),
            }),
        }
    }

    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.inner.ready.load(Ordering::Acquire)
    }

    pub fn try_generate(
        &self,
        request: GenerationRequest,
    ) -> Result<mpsc::Receiver<GenerationEvent>, EngineSubmitError> {
        if !self.is_ready() {
            return Err(EngineSubmitError::NotReady);
        }
        let (events, receiver) = mpsc::channel(self.inner.event_capacity);
        self.inner
            .commands
            .lock()
            .map_err(|_| EngineSubmitError::Stopped)?
            .as_ref()
            .ok_or(EngineSubmitError::Stopped)?
            .try_send(EngineCommand::Generate { request, events })
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => EngineSubmitError::QueueFull,
                mpsc::error::TrySendError::Closed(_) => EngineSubmitError::Stopped,
            })?;
        Ok(receiver)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineSubmitError {
    NotReady,
    QueueFull,
    Stopped,
}

impl fmt::Display for EngineSubmitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NotReady => "engine is not ready",
            Self::QueueFull => "engine command queue is full",
            Self::Stopped => "engine has stopped",
        })
    }
}

impl std::error::Error for EngineSubmitError {}

#[derive(Debug)]
enum EngineCommand {
    Generate {
        request: GenerationRequest,
        events: mpsc::Sender<GenerationEvent>,
    },
}

#[derive(Debug)]
struct RunError {
    message: String,
    notify_client: bool,
}

impl RunError {
    fn backend(operation: &str, error: impl fmt::Display) -> Self {
        Self {
            message: format!("{operation}: {error}"),
            notify_client: true,
        }
    }

    fn client_gone() -> Self {
        Self {
            message: "request output channel closed or filled".to_owned(),
            notify_client: false,
        }
    }
}

fn process_request(
    executor: &mut impl ModelExecutor,
    request: GenerationRequest,
    events: &mpsc::Sender<GenerationEvent>,
) {
    let request_id = request.id;
    if send_event(events, GenerationEvent::Started { request_id }).is_err() {
        return;
    }

    let mut submitted_sequence = false;
    let result = run_generation(executor, &request, events, &mut submitted_sequence);
    let cleanup = if submitted_sequence {
        executor.release_sequence(SINGLE_SEQUENCE)
    } else {
        Ok(())
    };

    match (result, cleanup) {
        (Ok(reason), Ok(())) => {
            let _ = send_event(
                events,
                GenerationEvent::Completed {
                    request_id,
                    finish_reason: reason,
                },
            );
        }
        (Err(error), cleanup_result) => {
            if error.notify_client {
                let message = match cleanup_result {
                    Ok(()) => error.message,
                    Err(cleanup_error) => {
                        format!("{}; cleanup failed: {cleanup_error}", error.message)
                    }
                };
                let _ = send_event(
                    events,
                    GenerationEvent::Error {
                        request_id,
                        message,
                    },
                );
            }
        }
        (Ok(_), Err(error)) => {
            let _ = send_event(
                events,
                GenerationEvent::Error {
                    request_id,
                    message: format!("release sequence: {error}"),
                },
            );
        }
    }
}

fn run_generation(
    executor: &mut impl ModelExecutor,
    request: &GenerationRequest,
    events: &mpsc::Sender<GenerationEvent>,
    submitted_sequence: &mut bool,
) -> Result<FinishReason, RunError> {
    if request.max_new_tokens == 0 {
        return Err(RunError::backend(
            "validate request",
            "max_new_tokens must be positive",
        ));
    }
    let prompt_tokens = executor
        .tokenize(&request.prompt)
        .map_err(|error| RunError::backend("tokenize prompt", error))?;
    if prompt_tokens.len() > executor.limits().max_batch_tokens {
        return Err(RunError::backend(
            "validate prompt",
            format!(
                "prompt has {} tokens; backend batch limit is {}",
                prompt_tokens.len(),
                executor.limits().max_batch_tokens
            ),
        ));
    }

    let last_prompt_index = prompt_tokens.len().saturating_sub(1);
    let tokens = prompt_tokens
        .iter()
        .copied()
        .enumerate()
        .map(|(index, token_id)| {
            let position = u32::try_from(index)
                .map_err(|error| RunError::backend("convert prompt position", error))?;
            Ok(BatchToken {
                sequence_id: SINGLE_SEQUENCE,
                token_id,
                position,
                request_logits: index == last_prompt_index,
            })
        })
        .collect::<Result<Vec<_>, RunError>>()?;

    let seed = request.seed.map_or(0, fold_seed);
    executor
        .start_sequence(
            SINGLE_SEQUENCE,
            SamplingConfig {
                temperature: request.temperature,
                top_p: request.top_p,
                seed,
            },
        )
        .map_err(|error| RunError::backend("start sequence", error))?;
    *submitted_sequence = true;
    let output = one_output(
        executor
            .execute(&ExecutionPlan { tokens })
            .map_err(|error| RunError::backend("prefill prompt", error))?,
    )?;
    let mut text = Utf8Accumulator::default();
    let mut generated = 1_u32;
    let mut pending = output.sampled_token_id;
    if output.is_eog {
        return Ok(FinishReason::Stop);
    }
    emit_token(events, request.id, output, &mut text)?;

    while generated < request.max_new_tokens {
        let prompt_len = u32::try_from(prompt_tokens.len())
            .map_err(|error| RunError::backend("convert prompt length", error))?;
        let position = prompt_len
            .checked_add(generated - 1)
            .ok_or_else(|| RunError::backend("advance position", "position overflow"))?;
        let output = one_output(
            executor
                .execute(&ExecutionPlan {
                    tokens: vec![BatchToken {
                        sequence_id: SINGLE_SEQUENCE,
                        token_id: pending,
                        position,
                        request_logits: true,
                    }],
                })
                .map_err(|error| RunError::backend("decode token", error))?,
        )?;
        generated += 1;
        pending = output.sampled_token_id;
        if output.is_eog {
            flush_text(events, request.id, pending, &mut text)?;
            return Ok(FinishReason::Stop);
        }
        emit_token(events, request.id, output, &mut text)?;
    }

    flush_text(events, request.id, pending, &mut text)?;
    Ok(FinishReason::Length)
}

fn fold_seed(seed: u64) -> u32 {
    let high = u32::try_from(seed >> 32).expect("shifted u64 fits in u32");
    let low = seed as u32;
    high ^ low
}

fn one_output(mut outputs: Vec<BackendTokenEvent>) -> Result<BackendTokenEvent, RunError> {
    if outputs.len() != 1 {
        return Err(RunError::backend(
            "map logits output",
            format!("expected one sampled token, received {}", outputs.len()),
        ));
    }
    Ok(outputs.remove(0))
}

fn emit_token(
    events: &mpsc::Sender<GenerationEvent>,
    request_id: niniserve_protocol::RequestId,
    output: BackendTokenEvent,
    text: &mut Utf8Accumulator,
) -> Result<(), RunError> {
    if let Some(fragment) = text.push(&output.text_bytes)? {
        send_event(
            events,
            GenerationEvent::Token {
                request_id,
                token_id: output.sampled_token_id,
                text: fragment,
            },
        )?;
    }
    Ok(())
}

fn flush_text(
    events: &mpsc::Sender<GenerationEvent>,
    request_id: niniserve_protocol::RequestId,
    token_id: u32,
    text: &mut Utf8Accumulator,
) -> Result<(), RunError> {
    if let Some(fragment) = text.finish() {
        send_event(
            events,
            GenerationEvent::Token {
                request_id,
                token_id,
                text: fragment,
            },
        )?;
    }
    Ok(())
}

fn send_event(
    events: &mpsc::Sender<GenerationEvent>,
    event: GenerationEvent,
) -> Result<(), RunError> {
    events.try_send(event).map_err(|_| RunError::client_gone())
}

#[derive(Debug, Default)]
struct Utf8Accumulator {
    pending: Vec<u8>,
}

impl Utf8Accumulator {
    fn push(&mut self, bytes: &[u8]) -> Result<Option<String>, RunError> {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(text) => {
                let text = text.to_owned();
                self.pending.clear();
                Ok((!text.is_empty()).then_some(text))
            }
            Err(error) if error.error_len().is_none() => {
                let valid = error.valid_up_to();
                if valid == 0 {
                    return Ok(None);
                }
                let remainder = self.pending.split_off(valid);
                let prefix = String::from_utf8(std::mem::replace(&mut self.pending, remainder))
                    .expect("validated UTF-8 prefix");
                Ok(Some(prefix))
            }
            Err(error) => Err(RunError::backend("decode token text", error)),
        }
    }

    fn finish(&mut self) -> Option<String> {
        (!self.pending.is_empty()).then(|| {
            let bytes = std::mem::take(&mut self.pending);
            String::from_utf8_lossy(&bytes).into_owned()
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    use niniserve_backend::{
        BackendError, BackendLimits, BackendTokenEvent, ExecutionPlan, MockExecutor, ModelExecutor,
        SamplingConfig,
    };
    use niniserve_protocol::{GenerationEvent, GenerationRequest, RequestId, SequenceId};

    use super::{EngineConfig, EngineHandle};

    fn request(id: u64) -> GenerationRequest {
        GenerationRequest {
            id: RequestId(id),
            prompt: "A".to_owned(),
            max_new_tokens: 3,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(42),
        }
    }

    async fn collect(engine: &EngineHandle, id: u64) -> Vec<GenerationEvent> {
        let mut events = engine
            .try_generate(request(id))
            .expect("request should enter the bounded command queue");
        let mut collected = Vec::new();
        while let Some(event) = events.recv().await {
            let terminal = event.is_terminal();
            collected.push(event);
            if terminal {
                break;
            }
        }
        collected
    }

    #[tokio::test]
    async fn two_sequential_requests_stream_and_release_the_single_backend_slot() {
        let executor = MockExecutor::new(BackendLimits {
            max_batch_tokens: 8,
            max_active_sequences: 1,
        });
        let engine = EngineHandle::spawn(
            executor,
            EngineConfig {
                command_capacity: 1,
                event_capacity: 8,
            },
        );

        let first = collect(&engine, 10).await;
        let second = collect(&engine, 11).await;

        assert_eq!(
            first,
            vec![
                GenerationEvent::Started {
                    request_id: RequestId(10),
                },
                GenerationEvent::Token {
                    request_id: RequestId(10),
                    token_id: 69,
                    text: "<69>".to_owned(),
                },
                GenerationEvent::Token {
                    request_id: RequestId(10),
                    token_id: 71,
                    text: "<71>".to_owned(),
                },
                GenerationEvent::Token {
                    request_id: RequestId(10),
                    token_id: 74,
                    text: "<74>".to_owned(),
                },
                GenerationEvent::Completed {
                    request_id: RequestId(10),
                    finish_reason: niniserve_protocol::FinishReason::Length,
                },
            ]
        );
        assert!(matches!(
            second.last(),
            Some(GenerationEvent::Completed {
                request_id: RequestId(11),
                finish_reason: niniserve_protocol::FinishReason::Length,
            })
        ));
    }

    #[test]
    fn dropping_the_last_handle_joins_the_worker_and_drops_the_executor() {
        let dropped = Arc::new(AtomicBool::new(false));
        let executor = DropTrackingExecutor {
            inner: MockExecutor::new(BackendLimits {
                max_batch_tokens: 8,
                max_active_sequences: 1,
            }),
            dropped: Arc::clone(&dropped),
        };

        let engine = EngineHandle::spawn(
            executor,
            EngineConfig {
                command_capacity: 1,
                event_capacity: 1,
            },
        );
        let clone = engine.clone();
        drop(engine);
        assert!(!dropped.load(Ordering::Acquire));
        drop(clone);
        assert!(dropped.load(Ordering::Acquire));
    }

    struct DropTrackingExecutor {
        inner: MockExecutor,
        dropped: Arc<AtomicBool>,
    }

    impl Drop for DropTrackingExecutor {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::Release);
        }
    }

    impl ModelExecutor for DropTrackingExecutor {
        fn tokenize(&self, prompt: &str) -> Result<Vec<u32>, BackendError> {
            self.inner.tokenize(prompt)
        }

        fn start_sequence(
            &mut self,
            sequence_id: SequenceId,
            sampling: SamplingConfig,
        ) -> Result<(), BackendError> {
            self.inner.start_sequence(sequence_id, sampling)
        }

        fn execute(
            &mut self,
            plan: &ExecutionPlan,
        ) -> Result<Vec<BackendTokenEvent>, BackendError> {
            self.inner.execute(plan)
        }

        fn release_sequence(&mut self, sequence_id: SequenceId) -> Result<(), BackendError> {
            self.inner.release_sequence(sequence_id)
        }

        fn limits(&self) -> BackendLimits {
            self.inner.limits()
        }
    }
}
