#![forbid(unsafe_code)]

use std::{
    collections::{BTreeMap, VecDeque},
    fmt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use niniserve_backend::{
    BackendTokenEvent, BatchToken, ExecutionPlan, ModelExecutor, SamplingConfig,
};
use niniserve_protocol::{FinishReason, GenerationEvent, GenerationRequest, RequestId, SequenceId};
use niniserve_scheduler::{
    BaselineScheduler, EngineView, Scheduler, SequencePhase, SequenceSnapshot, StepBudget, WorkKind,
};
use tokio::sync::mpsc;

pub use niniserve_scheduler::SchedulerConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineConfig {
    pub command_capacity: usize,
    pub pending_capacity: usize,
    pub event_capacity: usize,
    pub request_timeout: Duration,
    pub scheduler: SchedulerConfig,
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
    pub fn spawn<E: ModelExecutor + 'static>(executor: E, config: EngineConfig) -> Self {
        assert!(
            config.command_capacity > 0,
            "command capacity must be positive"
        );
        assert!(
            config.pending_capacity > 0,
            "pending capacity must be positive"
        );
        assert!(config.event_capacity > 0, "event capacity must be positive");
        let (commands, receiver) = mpsc::channel(config.command_capacity);
        let ready = Arc::new(AtomicBool::new(false));
        let worker_ready = Arc::clone(&ready);
        let (started, wait_for_start) = std::sync::mpsc::sync_channel(0);
        let worker = thread::Builder::new()
            .name("niniserve-engine".to_owned())
            .spawn(move || {
                let mut runtime = EngineRuntime::new(
                    executor,
                    receiver,
                    config.pending_capacity,
                    config.request_timeout,
                    config.scheduler,
                );
                worker_ready.store(true, Ordering::Release);
                let _ = started.send(());
                runtime.run();
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
    ) -> Result<GenerationStream, EngineSubmitError> {
        if !self.is_ready() {
            return Err(EngineSubmitError::NotReady);
        }
        let request_id = request.id;
        let (events, receiver) = mpsc::channel(self.inner.event_capacity);
        self.try_send(EngineCommand::Generate { request, events })?;
        Ok(GenerationStream {
            request_id,
            events: receiver,
            engine: Arc::clone(&self.inner),
            terminal: false,
        })
    }

    pub fn try_cancel(&self, request_id: RequestId) -> Result<(), EngineSubmitError> {
        self.try_send(EngineCommand::Cancel { request_id })
    }

    fn try_send(&self, command: EngineCommand) -> Result<(), EngineSubmitError> {
        self.inner
            .commands
            .lock()
            .map_err(|_| EngineSubmitError::Stopped)?
            .as_ref()
            .ok_or(EngineSubmitError::Stopped)?
            .try_send(command)
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => EngineSubmitError::QueueFull,
                mpsc::error::TrySendError::Closed(_) => EngineSubmitError::Stopped,
            })
    }
}

#[derive(Debug)]
pub struct GenerationStream {
    request_id: RequestId,
    events: mpsc::Receiver<GenerationEvent>,
    engine: Arc<EngineInner>,
    terminal: bool,
}

impl GenerationStream {
    pub async fn recv(&mut self) -> Option<GenerationEvent> {
        let event = self.events.recv().await;
        if event.as_ref().is_some_and(GenerationEvent::is_terminal) || event.is_none() {
            self.terminal = true;
        }
        event
    }
}

impl Drop for GenerationStream {
    fn drop(&mut self) {
        if self.terminal {
            return;
        }
        let Ok(commands) = self.engine.commands.lock() else {
            return;
        };
        if let Some(commands) = commands.as_ref() {
            let _ = commands.try_send(EngineCommand::Cancel {
                request_id: self.request_id,
            });
        }
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
    Cancel {
        request_id: RequestId,
    },
}

struct EngineRuntime<E> {
    executor: E,
    commands: mpsc::Receiver<EngineCommand>,
    pending: VecDeque<PendingRequest>,
    pending_capacity: usize,
    request_timeout: Duration,
    active: BTreeMap<SequenceId, ActiveRequest>,
    free_slots: Vec<SequenceId>,
    max_batch_tokens: usize,
    scheduler: BaselineScheduler,
    next_admission_order: u64,
    step: u64,
}

impl<E: ModelExecutor> EngineRuntime<E> {
    fn new(
        executor: E,
        commands: mpsc::Receiver<EngineCommand>,
        pending_capacity: usize,
        request_timeout: Duration,
        scheduler_config: SchedulerConfig,
    ) -> Self {
        let limits = executor.limits();
        let capacity = limits.max_active_sequences.min(limits.max_batch_tokens);
        assert!(capacity > 0, "backend active capacity must be positive");
        let free_slots = (0..capacity)
            .rev()
            .map(|slot| SequenceId(u32::try_from(slot).expect("sequence capacity exceeds u32")))
            .collect();
        Self {
            executor,
            commands,
            pending: VecDeque::new(),
            pending_capacity,
            request_timeout,
            active: BTreeMap::new(),
            free_slots,
            max_batch_tokens: limits.max_batch_tokens,
            scheduler: BaselineScheduler::new(scheduler_config),
            next_admission_order: 0,
            step: 0,
        }
    }

    fn run(&mut self) {
        loop {
            if self.active.is_empty() && self.pending.is_empty() {
                let Some(command) = self.commands.blocking_recv() else {
                    break;
                };
                self.handle_command(command);
            }
            let mut channel_closed = false;
            loop {
                match self.commands.try_recv() {
                    Ok(command) => self.handle_command(command),
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        channel_closed = true;
                        break;
                    }
                }
            }
            self.cancel_disconnected();
            self.expire_requests();
            if channel_closed {
                self.cancel_everything();
                break;
            }
            self.admit_pending();
            if !self.active.is_empty() {
                self.execute_step();
            }
        }
    }

    fn handle_command(&mut self, command: EngineCommand) {
        match command {
            EngineCommand::Generate { request, events } => {
                if self.pending.len() >= self.pending_capacity {
                    let _ = events.try_send(GenerationEvent::Error {
                        request_id: request.id,
                        message: "engine pending queue is full".to_owned(),
                    });
                } else {
                    let admission_order = self.next_admission_order;
                    self.next_admission_order = self.next_admission_order.saturating_add(1);
                    self.pending.push_back(PendingRequest {
                        request,
                        events,
                        enqueued_at: Instant::now(),
                        admission_order,
                    });
                }
            }
            EngineCommand::Cancel { request_id } => self.cancel_request(request_id, true),
        }
    }

    fn admit_pending(&mut self) {
        while let Some(sequence_id) = self.free_slots.pop() {
            let Some(pending) = self.pending.pop_front() else {
                self.free_slots.push(sequence_id);
                break;
            };
            if pending.events.is_closed() {
                self.free_slots.push(sequence_id);
                continue;
            }
            match ActiveRequest::prepare(&self.executor, pending) {
                Ok(mut request) => {
                    let sampling = SamplingConfig {
                        temperature: request.request.temperature,
                        top_p: request.request.top_p,
                        seed: request.request.seed.map_or(0, fold_seed),
                    };
                    if let Err(error) = self.executor.start_sequence(sequence_id, sampling) {
                        request.send_error(format!("start sequence: {error}"));
                        self.free_slots.push(sequence_id);
                        continue;
                    }
                    if request
                        .send(GenerationEvent::Started {
                            request_id: request.request.id,
                        })
                        .is_err()
                    {
                        let _ = self.executor.release_sequence(sequence_id);
                        self.free_slots.push(sequence_id);
                        continue;
                    }
                    self.active.insert(sequence_id, request);
                }
                Err((events, request_id, message)) => {
                    let _ = events.try_send(GenerationEvent::Error {
                        request_id,
                        message,
                    });
                    self.free_slots.push(sequence_id);
                }
            }
        }
    }

    fn execute_step(&mut self) {
        let scheduling_started = Instant::now();
        let snapshot = self
            .active
            .iter()
            .map(|(&sequence_id, request)| SequenceSnapshot {
                sequence_id,
                admission_order: request.admission_order,
                phase: request.scheduler_phase(),
                remaining_prefill_tokens: request.remaining_prefill_tokens(),
            })
            .collect::<Vec<_>>();
        let scheduled = self.scheduler.plan(
            EngineView {
                sequences: &snapshot,
            },
            StepBudget {
                max_batch_tokens: self.max_batch_tokens,
            },
        );
        let planned = scheduled
            .work
            .into_iter()
            .filter_map(|work| {
                self.active
                    .get(&work.sequence_id)
                    .map(|request| PlannedWork {
                        sequence_id: work.sequence_id,
                        tokens: request.batch_tokens(work.sequence_id, work.token_count),
                        kind: work.kind,
                    })
            })
            .collect::<Vec<_>>();
        let plan = ExecutionPlan {
            tokens: planned
                .iter()
                .flat_map(|work| work.tokens.iter().copied())
                .collect(),
        };
        if plan.tokens.is_empty() {
            self.fail_all("scheduler returned an empty plan for active requests".to_owned());
            return;
        }
        let scheduling_us = scheduling_started.elapsed().as_micros();
        let backend_started = Instant::now();
        let result = self.executor.execute(&plan);
        let backend_us = backend_started.elapsed().as_micros();
        eprintln!(
            "scheduler_step step={} policy={} prefill_tokens={} decode_tokens={} members=[{}] scheduling_us={} backend_us={}",
            self.step,
            self.scheduler.name(),
            planned
                .iter()
                .filter(|work| work.kind == WorkKind::Prefill)
                .map(|work| work.tokens.len())
                .sum::<usize>(),
            planned
                .iter()
                .filter(|work| work.kind == WorkKind::Decode)
                .map(|work| work.tokens.len())
                .sum::<usize>(),
            planned
                .iter()
                .flat_map(|work| work.tokens.iter())
                .map(|token| format!(
                    "seq:{} pos:{} phase:{} logits:{}",
                    token.sequence_id.0,
                    token.position,
                    if token.request_logits {
                        "decode-or-prefill-end"
                    } else {
                        "prefill"
                    },
                    token.request_logits
                ))
                .collect::<Vec<_>>()
                .join(", "),
            scheduling_us,
            backend_us
        );
        self.step = self.step.saturating_add(1);
        let outputs = match result {
            Ok(outputs) => outputs,
            Err(error) => {
                self.fail_all(format!("execute shared batch: {error}"));
                return;
            }
        };
        let mut outputs = outputs
            .into_iter()
            .map(|output| (output.sequence_id, output))
            .collect::<BTreeMap<_, _>>();
        let mut finished = Vec::new();
        for work in planned {
            let output = outputs.remove(&work.sequence_id);
            let Some(request) = self.active.get_mut(&work.sequence_id) else {
                continue;
            };
            match request.advance(&work.tokens, output) {
                Ok(Some(reason)) => finished.push((work.sequence_id, EndState::Completed(reason))),
                Ok(None) => {}
                Err(error) => finished.push((work.sequence_id, error)),
            }
        }
        if !outputs.is_empty() {
            self.fail_all("backend returned output for an unrequested sequence".to_owned());
            return;
        }
        for (sequence_id, state) in finished {
            self.finish_sequence(sequence_id, state);
        }
    }

    fn cancel_disconnected(&mut self) {
        let disconnected = self
            .active
            .iter()
            .filter_map(|(&id, request)| request.events.is_closed().then_some(id))
            .collect::<Vec<_>>();
        for id in disconnected {
            self.finish_sequence(id, EndState::Abandoned);
        }
        self.pending.retain(|pending| !pending.events.is_closed());
    }

    fn expire_requests(&mut self) {
        let mut retained = VecDeque::with_capacity(self.pending.len());
        while let Some(pending) = self.pending.pop_front() {
            if pending.enqueued_at.elapsed() >= self.request_timeout {
                let _ = pending.events.try_send(GenerationEvent::TimedOut {
                    request_id: pending.request.id,
                });
            } else {
                retained.push_back(pending);
            }
        }
        self.pending = retained;
        let expired = self
            .active
            .iter()
            .filter_map(|(&sequence_id, request)| {
                (request.enqueued_at.elapsed() >= self.request_timeout).then_some(sequence_id)
            })
            .collect::<Vec<_>>();
        for sequence_id in expired {
            self.finish_sequence(sequence_id, EndState::TimedOut);
        }
    }

    fn cancel_request(&mut self, request_id: RequestId, notify: bool) {
        if let Some(index) = self
            .pending
            .iter()
            .position(|pending| pending.request.id == request_id)
        {
            if let Some(pending) = self.pending.remove(index)
                && notify
            {
                let _ = pending
                    .events
                    .try_send(GenerationEvent::Cancelled { request_id });
            }
            return;
        }
        if let Some(sequence_id) = self
            .active
            .iter()
            .find_map(|(&id, request)| (request.request.id == request_id).then_some(id))
        {
            self.finish_sequence(
                sequence_id,
                if notify {
                    EndState::Cancelled
                } else {
                    EndState::Abandoned
                },
            );
        }
    }

    fn finish_sequence(&mut self, sequence_id: SequenceId, state: EndState) {
        let Some(mut request) = self.active.remove(&sequence_id) else {
            return;
        };
        let cleanup = self.executor.release_sequence(sequence_id);
        if cleanup.is_ok() {
            self.free_slots.push(sequence_id);
            self.free_slots
                .sort_unstable_by(|left, right| right.cmp(left));
        }
        match (state, cleanup) {
            (EndState::Abandoned, _) => {}
            (_, Err(error)) => request.send_error(format!("release sequence: {error}")),
            (EndState::Completed(reason), Ok(())) => {
                let _ = request.send(GenerationEvent::Completed {
                    request_id: request.request.id,
                    finish_reason: reason,
                });
            }
            (EndState::Cancelled, Ok(())) => {
                let _ = request.send(GenerationEvent::Cancelled {
                    request_id: request.request.id,
                });
            }
            (EndState::TimedOut, Ok(())) => {
                let _ = request.send(GenerationEvent::TimedOut {
                    request_id: request.request.id,
                });
            }
            (EndState::Error(message), Ok(())) => request.send_error(message),
        }
    }

    fn fail_all(&mut self, message: String) {
        let sequences = self.active.keys().copied().collect::<Vec<_>>();
        for sequence_id in sequences {
            self.finish_sequence(sequence_id, EndState::Error(message.clone()));
        }
    }

    fn cancel_everything(&mut self) {
        self.pending.clear();
        let sequences = self.active.keys().copied().collect::<Vec<_>>();
        for sequence_id in sequences {
            self.finish_sequence(sequence_id, EndState::Abandoned);
        }
    }
}

struct PendingRequest {
    request: GenerationRequest,
    events: mpsc::Sender<GenerationEvent>,
    enqueued_at: Instant,
    admission_order: u64,
}

struct ActiveRequest {
    request: GenerationRequest,
    events: mpsc::Sender<GenerationEvent>,
    phase: RequestPhase,
    text: Utf8Accumulator,
    enqueued_at: Instant,
    admission_order: u64,
}

impl ActiveRequest {
    fn prepare<E: ModelExecutor>(
        executor: &E,
        pending: PendingRequest,
    ) -> Result<Self, (mpsc::Sender<GenerationEvent>, RequestId, String)> {
        let request_id = pending.request.id;
        if pending.request.max_new_tokens == 0 {
            return Err((
                pending.events,
                request_id,
                "validate request: max_new_tokens must be positive".to_owned(),
            ));
        }
        let prompt = match executor.tokenize(&pending.request.prompt) {
            Ok(prompt) => prompt,
            Err(error) => {
                return Err((
                    pending.events,
                    request_id,
                    format!("tokenize prompt: {error}"),
                ));
            }
        };
        if prompt.is_empty() {
            return Err((
                pending.events,
                request_id,
                "tokenize prompt: backend returned no prompt tokens".to_owned(),
            ));
        }
        let output_tokens = usize::try_from(pending.request.max_new_tokens).map_err(|error| {
            (
                pending.events.clone(),
                request_id,
                format!("validate request token limit: {error}"),
            )
        })?;
        let required_tokens = prompt
            .len()
            .checked_add(output_tokens.saturating_sub(1))
            .ok_or_else(|| {
                (
                    pending.events.clone(),
                    request_id,
                    "validate request: sequence length overflow".to_owned(),
                )
            })?;
        let maximum = executor.limits().max_sequence_tokens;
        if required_tokens > maximum {
            return Err((
                pending.events,
                request_id,
                format!(
                    "validate request: prompt and output require {required_tokens} sequence tokens; backend limit is {maximum}"
                ),
            ));
        }
        Ok(Self {
            request: pending.request,
            events: pending.events,
            phase: RequestPhase::Prefill { prompt, next: 0 },
            text: Utf8Accumulator::default(),
            enqueued_at: pending.enqueued_at,
            admission_order: pending.admission_order,
        })
    }

    fn batch_tokens(&self, sequence_id: SequenceId, token_count: usize) -> Vec<BatchToken> {
        match &self.phase {
            RequestPhase::Prefill { prompt, next } => {
                let end = next.saturating_add(token_count).min(prompt.len());
                (*next..end)
                    .map(|position| BatchToken {
                        sequence_id,
                        token_id: prompt[position],
                        position: u32::try_from(position).expect("prompt position exceeds u32"),
                        request_logits: position + 1 == prompt.len(),
                    })
                    .collect()
            }
            RequestPhase::Decode {
                pending_token,
                next_position,
                ..
            } => vec![BatchToken {
                sequence_id,
                token_id: *pending_token,
                position: *next_position,
                request_logits: true,
            }],
        }
    }

    fn scheduler_phase(&self) -> SequencePhase {
        match self.phase {
            RequestPhase::Prefill { .. } => SequencePhase::Prefill,
            RequestPhase::Decode { .. } => SequencePhase::Decode,
        }
    }

    fn remaining_prefill_tokens(&self) -> usize {
        match &self.phase {
            RequestPhase::Prefill { prompt, next } => prompt.len() - next,
            RequestPhase::Decode { .. } => 0,
        }
    }

    fn advance(
        &mut self,
        submitted: &[BatchToken],
        output: Option<BackendTokenEvent>,
    ) -> Result<Option<FinishReason>, EndState> {
        let final_token = submitted
            .last()
            .copied()
            .ok_or_else(|| EndState::Error("scheduler submitted empty sequence work".to_owned()))?;
        match &mut self.phase {
            RequestPhase::Prefill { prompt, next } => {
                *next += submitted.len();
                if *next < prompt.len() {
                    if output.is_some() {
                        return Err(EndState::Error(
                            "backend returned logits before final prompt token".to_owned(),
                        ));
                    }
                    return Ok(None);
                }
                let output = validate_output(final_token, output)?;
                self.accept_sample(output, 1, final_token.position.saturating_add(1))
            }
            RequestPhase::Decode {
                generated,
                next_position,
                ..
            } => {
                let output = validate_output(final_token, output)?;
                let generated = generated.saturating_add(1);
                let next_position = next_position.saturating_add(1);
                self.accept_sample(output, generated, next_position)
            }
        }
    }

    fn accept_sample(
        &mut self,
        output: BackendTokenEvent,
        generated: u32,
        next_position: u32,
    ) -> Result<Option<FinishReason>, EndState> {
        if output.is_eog {
            self.flush(output.sampled_token_id)?;
            return Ok(Some(FinishReason::Stop));
        }
        self.emit(&output)?;
        if generated >= self.request.max_new_tokens {
            self.flush(output.sampled_token_id)?;
            return Ok(Some(FinishReason::Length));
        }
        self.phase = RequestPhase::Decode {
            pending_token: output.sampled_token_id,
            next_position,
            generated,
        };
        Ok(None)
    }

    fn emit(&mut self, output: &BackendTokenEvent) -> Result<(), EndState> {
        if let Some(fragment) = self.text.push(&output.text_bytes)? {
            self.send(GenerationEvent::Token {
                request_id: self.request.id,
                token_id: output.sampled_token_id,
                text: fragment,
            })?;
        }
        Ok(())
    }

    fn flush(&mut self, token_id: u32) -> Result<(), EndState> {
        if let Some(fragment) = self.text.finish() {
            self.send(GenerationEvent::Token {
                request_id: self.request.id,
                token_id,
                text: fragment,
            })?;
        }
        Ok(())
    }

    fn send(&self, event: GenerationEvent) -> Result<(), EndState> {
        self.events.try_send(event).map_err(|_| EndState::Abandoned)
    }
    fn send_error(&mut self, message: String) {
        let _ = self.events.try_send(GenerationEvent::Error {
            request_id: self.request.id,
            message,
        });
    }
}

enum RequestPhase {
    Prefill {
        prompt: Vec<u32>,
        next: usize,
    },
    Decode {
        pending_token: u32,
        next_position: u32,
        generated: u32,
    },
}
struct PlannedWork {
    sequence_id: SequenceId,
    tokens: Vec<BatchToken>,
    kind: WorkKind,
}
enum EndState {
    Completed(FinishReason),
    Cancelled,
    TimedOut,
    Abandoned,
    Error(String),
}

fn validate_output(
    submitted: BatchToken,
    output: Option<BackendTokenEvent>,
) -> Result<BackendTokenEvent, EndState> {
    let output = output.ok_or_else(|| {
        EndState::Error(format!(
            "backend returned no sampled token for sequence {}",
            submitted.sequence_id.0
        ))
    })?;
    if output.evaluated_position != submitted.position {
        return Err(EndState::Error(format!(
            "backend output position {} did not match submitted position {}",
            output.evaluated_position, submitted.position
        )));
    }
    Ok(output)
}

fn fold_seed(seed: u64) -> u32 {
    u32::try_from(seed >> 32).expect("shifted u64 fits in u32") ^ seed as u32
}

#[derive(Debug, Default)]
struct Utf8Accumulator {
    pending: Vec<u8>,
}
impl Utf8Accumulator {
    fn push(&mut self, bytes: &[u8]) -> Result<Option<String>, EndState> {
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
            Err(error) => Err(EndState::Error(format!("decode token text: {error}"))),
        }
    }
    fn finish(&mut self) -> Option<String> {
        (!self.pending.is_empty())
            .then(|| String::from_utf8_lossy(&std::mem::take(&mut self.pending)).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::{EngineConfig, EngineHandle, SchedulerConfig};
    use niniserve_backend::{
        BackendError, BackendLimits, BackendTokenEvent, BatchToken, ExecutionPlan, MockExecutor,
        ModelExecutor, SamplingConfig,
    };
    use niniserve_protocol::{GenerationEvent, GenerationRequest, RequestId, SequenceId};
    use std::sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    };
    use std::time::Duration;

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
    fn config() -> EngineConfig {
        EngineConfig {
            command_capacity: 4,
            pending_capacity: 4,
            event_capacity: 128,
            request_timeout: Duration::from_secs(30),
            scheduler: SchedulerConfig::DecodePriority,
        }
    }
    async fn collect(engine: &EngineHandle, id: u64) -> Vec<GenerationEvent> {
        let events = engine
            .try_generate(request(id))
            .expect("request should enter queue");
        collect_stream(events).await
    }

    async fn collect_stream(mut events: super::GenerationStream) -> Vec<GenerationEvent> {
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
    async fn concurrent_requests_share_batches_without_mixing_outputs() {
        let plans = Arc::new(Mutex::new(Vec::<Vec<BatchToken>>::new()));
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let executor = RecordingGateExecutor {
            inner: MockExecutor::new(BackendLimits {
                max_batch_tokens: 8,
                max_active_sequences: 2,
                max_sequence_tokens: 128,
            }),
            plans: Arc::clone(&plans),
            gate: Arc::clone(&gate),
            first_tokenize: AtomicBool::new(true),
            tokenize_entered: Arc::new(AtomicBool::new(false)),
            execute_gate: None,
            first_execute: AtomicBool::new(true),
        };
        let engine = EngineHandle::spawn(executor, config());

        let first = engine.try_generate(request(30)).expect("first is queued");
        let second = engine.try_generate(request(31)).expect("second is queued");
        let (open, wake) = &*gate;
        *open.lock().unwrap() = true;
        wake.notify_one();

        let (first_events, second_events) =
            tokio::join!(collect_stream(first), collect_stream(second));

        assert!(
            first_events
                .iter()
                .all(|event| event.request_id() == RequestId(30))
        );
        assert!(
            second_events
                .iter()
                .all(|event| event.request_id() == RequestId(31))
        );
        assert!(plans.lock().unwrap().iter().any(|batch| {
            batch.iter().any(|token| token.sequence_id == SequenceId(0))
                && batch.iter().any(|token| token.sequence_id == SequenceId(1))
        }));
    }

    #[tokio::test]
    async fn cancelling_one_active_request_preserves_its_peer_and_reuses_the_slot() {
        let tokenize_gate = Arc::new((Mutex::new(false), Condvar::new()));
        let execute_gate = Arc::new((Mutex::new(false), Condvar::new()));
        let executor = RecordingGateExecutor {
            inner: MockExecutor::new(BackendLimits {
                max_batch_tokens: 8,
                max_active_sequences: 2,
                max_sequence_tokens: 128,
            }),
            plans: Arc::new(Mutex::new(Vec::new())),
            gate: Arc::clone(&tokenize_gate),
            first_tokenize: AtomicBool::new(true),
            tokenize_entered: Arc::new(AtomicBool::new(false)),
            execute_gate: Some(Arc::clone(&execute_gate)),
            first_execute: AtomicBool::new(true),
        };
        let engine = EngineHandle::spawn(executor, config());
        let mut long = request(40);
        long.max_new_tokens = 100;
        let mut cancelled = engine.try_generate(long).expect("first is queued");
        let mut peer = engine.try_generate(request(41)).expect("peer is queued");
        let (open, wake) = &*tokenize_gate;
        *open.lock().unwrap() = true;
        wake.notify_one();

        assert!(matches!(
            cancelled.recv().await,
            Some(GenerationEvent::Started { .. })
        ));
        assert!(matches!(
            peer.recv().await,
            Some(GenerationEvent::Started { .. })
        ));
        engine.try_cancel(RequestId(40)).expect("cancel is queued");
        let (open, wake) = &*execute_gate;
        *open.lock().unwrap() = true;
        wake.notify_one();

        let (cancelled_events, peer_events) =
            tokio::join!(collect_stream(cancelled), collect_stream(peer));
        assert!(matches!(
            cancelled_events.last(),
            Some(GenerationEvent::Cancelled {
                request_id: RequestId(40)
            })
        ));
        assert!(matches!(
            peer_events.last(),
            Some(GenerationEvent::Completed {
                request_id: RequestId(41),
                ..
            })
        ));
        assert!(matches!(
            collect(&engine, 42).await.last(),
            Some(GenerationEvent::Completed {
                request_id: RequestId(42),
                ..
            })
        ));
    }

    #[tokio::test]
    async fn bounded_command_queue_rejects_overload() {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let entered = Arc::new(AtomicBool::new(false));
        let executor = RecordingGateExecutor {
            inner: MockExecutor::new(BackendLimits {
                max_batch_tokens: 8,
                max_active_sequences: 1,
                max_sequence_tokens: 128,
            }),
            plans: Arc::new(Mutex::new(Vec::new())),
            gate: Arc::clone(&gate),
            first_tokenize: AtomicBool::new(true),
            tokenize_entered: Arc::clone(&entered),
            execute_gate: None,
            first_execute: AtomicBool::new(true),
        };
        let engine = EngineHandle::spawn(
            executor,
            EngineConfig {
                command_capacity: 1,
                pending_capacity: 1,
                event_capacity: 128,
                request_timeout: Duration::from_secs(30),
                scheduler: SchedulerConfig::DecodePriority,
            },
        );
        let first = engine.try_generate(request(50)).expect("first is accepted");
        while !entered.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        let second = engine
            .try_generate(request(51))
            .expect("queue has one slot");
        assert!(matches!(
            engine.try_generate(request(52)),
            Err(super::EngineSubmitError::QueueFull)
        ));
        drop(first);
        drop(second);
        let (open, wake) = &*gate;
        *open.lock().unwrap() = true;
        wake.notify_one();
    }

    #[tokio::test]
    async fn request_exceeding_sequence_context_is_rejected_before_backend_start() {
        let engine = EngineHandle::spawn(
            MockExecutor::new(BackendLimits {
                max_batch_tokens: 8,
                max_active_sequences: 1,
                max_sequence_tokens: 2,
            }),
            config(),
        );
        let events = collect(&engine, 60).await;
        assert!(matches!(
            events.as_slice(),
            [GenerationEvent::Error {
                request_id: RequestId(60),
                message
            }] if message.contains("backend limit is 2")
        ));
    }

    #[tokio::test]
    async fn expired_request_reports_timeout_without_entering_the_backend() {
        let engine = EngineHandle::spawn(
            MockExecutor::new(BackendLimits {
                max_batch_tokens: 8,
                max_active_sequences: 1,
                max_sequence_tokens: 128,
            }),
            EngineConfig {
                request_timeout: Duration::ZERO,
                ..config()
            },
        );
        assert_eq!(
            collect(&engine, 61).await,
            vec![GenerationEvent::TimedOut {
                request_id: RequestId(61)
            }]
        );
    }

    #[tokio::test]
    async fn two_sequential_requests_stream_and_release_the_single_backend_slot() {
        let engine = EngineHandle::spawn(
            MockExecutor::new(BackendLimits {
                max_batch_tokens: 8,
                max_active_sequences: 1,
                max_sequence_tokens: 128,
            }),
            config(),
        );
        let first = collect(&engine, 10).await;
        let second = collect(&engine, 11).await;
        assert_eq!(
            first,
            vec![
                GenerationEvent::Started {
                    request_id: RequestId(10)
                },
                GenerationEvent::Token {
                    request_id: RequestId(10),
                    token_id: 69,
                    text: "<69>".to_owned()
                },
                GenerationEvent::Token {
                    request_id: RequestId(10),
                    token_id: 71,
                    text: "<71>".to_owned()
                },
                GenerationEvent::Token {
                    request_id: RequestId(10),
                    token_id: 74,
                    text: "<74>".to_owned()
                },
                GenerationEvent::Completed {
                    request_id: RequestId(10),
                    finish_reason: niniserve_protocol::FinishReason::Length
                },
            ]
        );
        assert!(matches!(
            second.last(),
            Some(GenerationEvent::Completed {
                request_id: RequestId(11),
                ..
            })
        ));
    }

    #[tokio::test]
    async fn an_active_request_can_be_cancelled_explicitly() {
        let engine = EngineHandle::spawn(
            MockExecutor::new(BackendLimits {
                max_batch_tokens: 8,
                max_active_sequences: 1,
                max_sequence_tokens: 128,
            }),
            config(),
        );
        let mut input = request(20);
        input.max_new_tokens = 100;
        let mut events = engine.try_generate(input).expect("request is admitted");
        assert_eq!(
            events.recv().await,
            Some(GenerationEvent::Started {
                request_id: RequestId(20)
            })
        );
        engine
            .try_cancel(RequestId(20))
            .expect("cancellation enters queue");
        let mut terminal = None;
        while let Some(event) = events.recv().await {
            if event.is_terminal() {
                terminal = Some(event);
                break;
            }
        }
        assert_eq!(
            terminal,
            Some(GenerationEvent::Cancelled {
                request_id: RequestId(20)
            })
        );
    }

    #[test]
    fn dropping_the_last_handle_joins_the_worker_and_drops_the_executor() {
        let dropped = Arc::new(AtomicBool::new(false));
        let executor = DropTrackingExecutor {
            inner: MockExecutor::new(BackendLimits {
                max_batch_tokens: 8,
                max_active_sequences: 1,
                max_sequence_tokens: 128,
            }),
            dropped: Arc::clone(&dropped),
        };
        let engine = EngineHandle::spawn(executor, config());
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
            id: SequenceId,
            sampling: SamplingConfig,
        ) -> Result<(), BackendError> {
            self.inner.start_sequence(id, sampling)
        }
        fn execute(
            &mut self,
            plan: &ExecutionPlan,
        ) -> Result<Vec<BackendTokenEvent>, BackendError> {
            self.inner.execute(plan)
        }
        fn release_sequence(&mut self, id: SequenceId) -> Result<(), BackendError> {
            self.inner.release_sequence(id)
        }
        fn limits(&self) -> BackendLimits {
            self.inner.limits()
        }
    }

    struct RecordingGateExecutor {
        inner: MockExecutor,
        plans: Arc<Mutex<Vec<Vec<BatchToken>>>>,
        gate: Arc<(Mutex<bool>, Condvar)>,
        first_tokenize: AtomicBool,
        tokenize_entered: Arc<AtomicBool>,
        execute_gate: Option<Arc<(Mutex<bool>, Condvar)>>,
        first_execute: AtomicBool,
    }

    impl ModelExecutor for RecordingGateExecutor {
        fn tokenize(&self, prompt: &str) -> Result<Vec<u32>, BackendError> {
            if self.first_tokenize.swap(false, Ordering::AcqRel) {
                self.tokenize_entered.store(true, Ordering::Release);
                let (open, wake) = &*self.gate;
                let mut open = open.lock().unwrap();
                while !*open {
                    open = wake.wait(open).unwrap();
                }
            }
            self.inner.tokenize(prompt)
        }

        fn start_sequence(
            &mut self,
            id: SequenceId,
            sampling: SamplingConfig,
        ) -> Result<(), BackendError> {
            self.inner.start_sequence(id, sampling)
        }

        fn execute(
            &mut self,
            plan: &ExecutionPlan,
        ) -> Result<Vec<BackendTokenEvent>, BackendError> {
            if self.first_execute.swap(false, Ordering::AcqRel)
                && let Some(gate) = &self.execute_gate
            {
                let (open, wake) = &**gate;
                let mut open = open.lock().unwrap();
                while !*open {
                    open = wake.wait(open).unwrap();
                }
            }
            self.plans.lock().unwrap().push(plan.tokens.clone());
            self.inner.execute(plan)
        }

        fn release_sequence(&mut self, id: SequenceId) -> Result<(), BackendError> {
            self.inner.release_sequence(id)
        }

        fn limits(&self) -> BackendLimits {
            self.inner.limits()
        }
    }
}
