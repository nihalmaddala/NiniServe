//! Pinned llama.cpp feasibility probe.

use std::{
    collections::HashMap,
    fmt,
    num::NonZeroU32,
    path::{Path, PathBuf},
};

use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{LlamaModel, params::LlamaModelParams},
    sampling::LlamaSampler,
    token::LlamaToken,
};
use niniserve_protocol::SequenceId;
use self_cell::self_cell;

use crate::{
    BackendError, BackendLimits, BackendTokenEvent, ExecutionPlan, ModelExecutor, SamplingConfig,
};

type OwnedContext<'model> = llama_cpp_2::context::LlamaContext<'model>;

self_cell!(
    struct ModelContextCell {
        owner: LlamaModel,

        #[covariant]
        dependent: OwnedContext,
    }

    impl {Debug}
);

#[derive(Debug, Clone)]
pub struct LlamaCppConfig {
    pub model_path: PathBuf,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub gpu_layers: u32,
}

impl LlamaCppConfig {
    #[must_use]
    pub fn single_request(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            n_ctx: 2_048,
            n_batch: 512,
            n_ubatch: 128,
            n_seq_max: 1,
            gpu_layers: u32::MAX,
        }
    }
}

#[derive(Debug)]
pub struct LlamaCppExecutor {
    // Drop the self-referential model/context before shutting down the backend.
    runtime: ModelContextCell,
    _backend: LlamaBackend,
    samplers: HashMap<SequenceId, LlamaSampler>,
    limits: BackendLimits,
    gpu_offload_supported: bool,
}

impl LlamaCppExecutor {
    pub fn load(config: &LlamaCppConfig) -> Result<Self, BackendError> {
        if config.n_ubatch > config.n_batch {
            return Err(native_error("n_ubatch must not exceed n_batch"));
        }
        if config.n_seq_max == 0 {
            return Err(native_error("n_seq_max must be positive"));
        }
        let backend = LlamaBackend::init()
            .map_err(|error| native_operation("initialize llama backend", error))?;
        let gpu_offload_supported = backend.supports_gpu_offload();
        let model_params = LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers);
        let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)
            .map_err(|error| native_operation("load GGUF model", error))?;
        let context_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(config.n_ctx))
            .with_n_batch(config.n_batch)
            .with_n_ubatch(config.n_ubatch)
            .with_n_seq_max(config.n_seq_max);
        let runtime =
            ModelContextCell::try_new(model, |model| model.new_context(&backend, context_params))
                .map_err(|error| native_operation("create llama context", error))?;
        let limits = BackendLimits {
            max_batch_tokens: usize::try_from(runtime.borrow_dependent().n_batch())
                .map_err(|error| native_operation("convert n_batch", error))?,
            max_active_sequences: usize::try_from(config.n_seq_max)
                .map_err(|error| native_operation("convert n_seq_max", error))?,
        };
        Ok(Self {
            runtime,
            _backend: backend,
            samplers: HashMap::new(),
            limits,
            gpu_offload_supported,
        })
    }

    #[must_use]
    pub fn gpu_offload_supported(&self) -> bool {
        self.gpu_offload_supported
    }

    #[must_use]
    pub fn context_limits(&self) -> (u32, u32, u32) {
        let context = self.runtime.borrow_dependent();
        (context.n_ctx(), context.n_batch(), context.n_ubatch())
    }
}

impl ModelExecutor for LlamaCppExecutor {
    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>, BackendError> {
        if prompt.is_empty() {
            return Err(BackendError::EmptyPrompt);
        }
        self.runtime
            .borrow_owner()
            .vocab()
            .tokenize(prompt.as_bytes(), true, true)
            .into_iter()
            .map(|token| {
                u32::try_from(token.0).map_err(|error| native_operation("convert token ID", error))
            })
            .collect()
    }

    fn start_sequence(
        &mut self,
        sequence_id: SequenceId,
        sampling: SamplingConfig,
    ) -> Result<(), BackendError> {
        validate_sequence_slot(sequence_id, self.limits.max_active_sequences)?;
        if self.samplers.contains_key(&sequence_id) {
            return Err(native_error(format!(
                "sequence {} already has sampler state",
                sequence_id.0
            )));
        }
        let sampler = if sampling.temperature == 0.0 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::top_p(sampling.top_p, 1),
                LlamaSampler::temp(sampling.temperature),
                LlamaSampler::dist(sampling.seed),
            ])
        };
        self.samplers.insert(sequence_id, sampler);
        Ok(())
    }

    fn execute(&mut self, plan: &ExecutionPlan) -> Result<Vec<BackendTokenEvent>, BackendError> {
        if plan.tokens.is_empty() {
            return Err(BackendError::EmptyPlan);
        }
        if plan.tokens.len() > self.limits.max_batch_tokens {
            return Err(BackendError::BatchTooLarge {
                actual: plan.tokens.len(),
                maximum: self.limits.max_batch_tokens,
            });
        }

        let mut batch = LlamaBatch::new(plan.tokens.len(), 1);
        let mut requested = Vec::new();
        for token in &plan.tokens {
            validate_sequence_slot(token.sequence_id, self.limits.max_active_sequences)?;
            if !self.samplers.contains_key(&token.sequence_id) {
                return Err(BackendError::UnknownSequence(token.sequence_id));
            }
            let token_id = i32::try_from(token.token_id)
                .map_err(|error| native_operation("convert token ID", error))?;
            let position = i32::try_from(token.position)
                .map_err(|error| native_operation("convert token position", error))?;
            let sequence_id = i32::try_from(token.sequence_id.0)
                .map_err(|error| native_operation("convert sequence ID", error))?;
            let batch_index = batch.n_tokens();
            batch
                .add(
                    LlamaToken(token_id),
                    position,
                    &[sequence_id],
                    token.request_logits,
                )
                .map_err(|error| native_operation("add token to batch", error))?;
            if token.request_logits {
                requested.push((batch_index, *token));
            }
        }

        let samplers = &mut self.samplers;
        self.runtime.with_dependent_mut(|model, context| {
            context
                .decode(&mut batch)
                .map_err(|error| native_operation("decode batch", error))?;
            let vocab = model.vocab();
            requested
                .into_iter()
                .map(|(batch_index, input)| {
                    let sampler = samplers
                        .get_mut(&input.sequence_id)
                        .ok_or(BackendError::UnknownSequence(input.sequence_id))?;
                    let sampled = sampler.sample(context, batch_index);
                    let sampled_token_id = u32::try_from(sampled.0)
                        .map_err(|error| native_operation("convert sampled token", error))?;
                    Ok(BackendTokenEvent {
                        sequence_id: input.sequence_id,
                        evaluated_position: input.position,
                        sampled_token_id,
                        text_bytes: vocab.token_to_piece(sampled, true, None),
                        is_eog: vocab.is_eog(sampled),
                    })
                })
                .collect()
        })
    }

    fn release_sequence(&mut self, sequence_id: SequenceId) -> Result<(), BackendError> {
        let sequence_slot = validate_sequence_slot(sequence_id, self.limits.max_active_sequences)?;
        if !self.samplers.contains_key(&sequence_id) {
            return Err(BackendError::UnknownSequence(sequence_id));
        }
        self.runtime.with_dependent_mut(|_, context| {
            context
                .kv_cache_seq_rm(sequence_slot, None, None)
                .map_err(|error| native_operation("remove sequence memory", error))?;
            if context.kv_cache_seq_pos_max(sequence_slot) != -1 {
                return Err(native_error(format!(
                    "sequence {} memory remained after removal",
                    sequence_id.0
                )));
            }
            Ok(())
        })?;
        self.samplers.remove(&sequence_id);
        Ok(())
    }

    fn limits(&self) -> BackendLimits {
        self.limits
    }
}

fn validate_sequence_slot(sequence_id: SequenceId, maximum: usize) -> Result<i32, BackendError> {
    let slot = usize::try_from(sequence_id.0)
        .map_err(|error| native_operation("convert sequence slot", error))?;
    if slot >= maximum {
        return Err(BackendError::TooManySequences { maximum });
    }
    i32::try_from(sequence_id.0).map_err(|error| native_operation("convert sequence slot", error))
}

fn native_operation(operation: &str, error: impl fmt::Display) -> BackendError {
    native_error(format!("{operation}: {error}"))
}

fn native_error(message: impl Into<String>) -> BackendError {
    BackendError::Native(message.into())
}

// With non-unified KV streams, llama.cpp requires sequence IDs to be dense in
// `0..n_seq_max`; these are backend slots, not user-facing request IDs.
const SEQUENCE_A: i32 = 0;
const SEQUENCE_B: i32 = 1;
const SAMPLER_SEED_A: u32 = 0x00C0_FFEE;
const SAMPLER_SEED_B: u32 = 0x0BAD_5EED;

/// Configuration for the real two-sequence feasibility probe.
#[derive(Debug, Clone, Copy)]
pub struct ProbeConfig {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub generated_tokens: usize,
    pub gpu_layers: u32,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            n_ctx: 1_024,
            n_batch: 512,
            n_ubatch: 128,
            generated_tokens: 12,
            gpu_layers: u32::MAX,
        }
    }
}

/// One token actually submitted to llama.cpp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceEntry {
    pub batch_id: usize,
    pub phase: &'static str,
    pub sequence_id: i32,
    pub position: i32,
    pub token_id: i32,
    pub requested_logits: bool,
}

/// Measured result of one real-model probe run.
#[derive(Debug)]
pub struct ProbeReport {
    pub gpu_offload_supported: bool,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub sequence_a_tokens: Vec<i32>,
    pub sequence_b_tokens: Vec<i32>,
    pub sequence_a_text: String,
    pub sequence_b_text: String,
    pub trace: Vec<TraceEntry>,
    pub sequence_a_cleaned: bool,
    pub sequence_b_cleaned: bool,
}

#[derive(Debug)]
pub struct ProbeError(String);

impl ProbeError {
    fn message(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    fn operation(operation: &str, error: impl fmt::Display) -> Self {
        Self(format!("{operation}: {error}"))
    }
}

impl fmt::Display for ProbeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ProbeError {}

#[derive(Debug)]
struct SequenceState {
    id: i32,
    prompt_len: i32,
    pending: Option<LlamaToken>,
    generated: Vec<LlamaToken>,
}

impl SequenceState {
    fn new(id: i32, prompt_len: usize) -> Result<Self, ProbeError> {
        let prompt_len = i32::try_from(prompt_len)
            .map_err(|error| ProbeError::operation("prompt length exceeds i32", error))?;
        Ok(Self {
            id,
            prompt_len,
            pending: None,
            generated: Vec::new(),
        })
    }

    fn next_position(&self) -> Result<i32, ProbeError> {
        let generated = i32::try_from(self.generated.len())
            .map_err(|error| ProbeError::operation("generated length exceeds i32", error))?;
        self.prompt_len
            .checked_add(generated - 1)
            .ok_or_else(|| ProbeError::message("token position overflowed"))
    }
}

/// Runs the Phase 0 proof in one shared context and always attempts sequence cleanup.
pub fn run_two_sequence_probe(
    model_path: &Path,
    config: ProbeConfig,
) -> Result<ProbeReport, ProbeError> {
    if !(8..=16).contains(&config.generated_tokens) {
        return Err(ProbeError::message(
            "generated_tokens must be in the Phase 0 range 8..=16",
        ));
    }
    if config.n_ubatch > config.n_batch {
        return Err(ProbeError::message("n_ubatch must not exceed n_batch"));
    }

    let backend = LlamaBackend::init()
        .map_err(|error| ProbeError::operation("initialize llama backend", error))?;
    let gpu_offload_supported = backend.supports_gpu_offload();
    let model_params = LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers);
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .map_err(|error| ProbeError::operation("load GGUF model", error))?;
    let context_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(config.n_ctx))
        .with_n_batch(config.n_batch)
        .with_n_ubatch(config.n_ubatch)
        .with_n_seq_max(2);
    let mut context = model
        .new_context(&backend, context_params)
        .map_err(|error| ProbeError::operation("create llama context", error))?;

    let actual_n_ctx = context.n_ctx();
    let actual_n_batch = context.n_batch();
    let actual_n_ubatch = context.n_ubatch();
    let vocab = model.vocab();
    let prompt_a =
        b"<|im_start|>user\nName one benefit of Rust.\n<|im_end|>\n<|im_start|>assistant\n";
    let prompt_b = b"<|im_start|>user\nName one benefit of local inference.\n<|im_end|>\n<|im_start|>assistant\n";
    let tokens_a = vocab.tokenize(prompt_a, true, true);
    let tokens_b = vocab.tokenize(prompt_b, true, true);

    let total_prompt_tokens = tokens_a
        .len()
        .checked_add(tokens_b.len())
        .ok_or_else(|| ProbeError::message("combined prompt length overflowed"))?;
    if total_prompt_tokens > usize::try_from(actual_n_batch).unwrap_or(usize::MAX) {
        return Err(ProbeError::message(format!(
            "combined prompt has {total_prompt_tokens} tokens but n_batch is {actual_n_batch}"
        )));
    }

    let mut sequence_a = SequenceState::new(SEQUENCE_A, tokens_a.len())?;
    let mut sequence_b = SequenceState::new(SEQUENCE_B, tokens_b.len())?;
    let mut trace = Vec::new();
    let mut sampler_a = independent_sampler(SAMPLER_SEED_A);
    let mut sampler_b = independent_sampler(SAMPLER_SEED_B);

    let run_result = (|| -> Result<(), ProbeError> {
        let mut batch = LlamaBatch::new(total_prompt_tokens.max(2), 1);
        add_prompt(&mut batch, &tokens_a, SEQUENCE_A, 0, &mut trace)?;
        let logits_index_a = batch.n_tokens() - 1;
        add_prompt(&mut batch, &tokens_b, SEQUENCE_B, 0, &mut trace)?;
        let logits_index_b = batch.n_tokens() - 1;
        context
            .decode(&mut batch)
            .map_err(|error| ProbeError::operation("decode prefill batch", error))?;

        sample_next(
            &context,
            &vocab,
            &mut sampler_a,
            logits_index_a,
            &mut sequence_a,
        );
        sample_next(
            &context,
            &vocab,
            &mut sampler_b,
            logits_index_b,
            &mut sequence_b,
        );

        let mut batch_id = 1;
        while sequence_a.generated.len() < config.generated_tokens
            || sequence_b.generated.len() < config.generated_tokens
        {
            batch.clear();
            let index_a = add_pending(
                &mut batch,
                &sequence_a,
                config.generated_tokens,
                batch_id,
                &mut trace,
            )?;
            let index_b = add_pending(
                &mut batch,
                &sequence_b,
                config.generated_tokens,
                batch_id,
                &mut trace,
            )?;
            if batch.n_tokens() == 0 {
                break;
            }
            context
                .decode(&mut batch)
                .map_err(|error| ProbeError::operation("decode interleaved batch", error))?;

            if let Some(index) = index_a {
                sample_next(&context, &vocab, &mut sampler_a, index, &mut sequence_a);
            }
            if let Some(index) = index_b {
                sample_next(&context, &vocab, &mut sampler_b, index, &mut sequence_b);
            }
            batch_id += 1;
        }

        if sequence_a.generated.len() < 8 || sequence_b.generated.len() < 8 {
            return Err(ProbeError::message(format!(
                "EOG arrived before the required eight tokens: seq {SEQUENCE_A}={}, seq {SEQUENCE_B}={}",
                sequence_a.generated.len(),
                sequence_b.generated.len()
            )));
        }
        Ok(())
    })();

    let cleanup_a = context.kv_cache_seq_rm(SEQUENCE_A, None, None);
    let cleanup_b = context.kv_cache_seq_rm(SEQUENCE_B, None, None);
    let sequence_a_cleaned = cleanup_a.is_ok() && context.kv_cache_seq_pos_max(SEQUENCE_A) == -1;
    let sequence_b_cleaned = cleanup_b.is_ok() && context.kv_cache_seq_pos_max(SEQUENCE_B) == -1;
    run_result?;
    cleanup_a.map_err(|error| ProbeError::operation("clean sequence 0", error))?;
    cleanup_b.map_err(|error| ProbeError::operation("clean sequence 1", error))?;
    if !sequence_a_cleaned || !sequence_b_cleaned {
        return Err(ProbeError::message(
            "sequence memory remained after cleanup",
        ));
    }

    let generated_a = sequence_a.generated;
    let generated_b = sequence_b.generated;
    let sequence_a_text =
        String::from_utf8_lossy(&vocab.detokenize(&generated_a, true, true)).into_owned();
    let sequence_b_text =
        String::from_utf8_lossy(&vocab.detokenize(&generated_b, true, true)).into_owned();

    Ok(ProbeReport {
        gpu_offload_supported,
        n_ctx: actual_n_ctx,
        n_batch: actual_n_batch,
        n_ubatch: actual_n_ubatch,
        sequence_a_tokens: generated_a.into_iter().map(|token| token.0).collect(),
        sequence_b_tokens: generated_b.into_iter().map(|token| token.0).collect(),
        sequence_a_text,
        sequence_b_text,
        trace,
        sequence_a_cleaned,
        sequence_b_cleaned,
    })
}

fn independent_sampler(seed: u32) -> LlamaSampler {
    LlamaSampler::chain_simple([
        LlamaSampler::top_k(40),
        LlamaSampler::temp(0.7),
        LlamaSampler::dist(seed),
    ])
}

fn add_prompt(
    batch: &mut LlamaBatch<'_>,
    tokens: &[LlamaToken],
    sequence_id: i32,
    batch_id: usize,
    trace: &mut Vec<TraceEntry>,
) -> Result<(), ProbeError> {
    let last = tokens.len().saturating_sub(1);
    for (offset, token) in tokens.iter().copied().enumerate() {
        let position = i32::try_from(offset)
            .map_err(|error| ProbeError::operation("prompt position exceeds i32", error))?;
        let requested_logits = offset == last;
        batch
            .add(token, position, &[sequence_id], requested_logits)
            .map_err(|error| ProbeError::operation("add prompt token to batch", error))?;
        trace.push(TraceEntry {
            batch_id,
            phase: "prefill",
            sequence_id,
            position,
            token_id: token.0,
            requested_logits,
        });
    }
    Ok(())
}

fn add_pending(
    batch: &mut LlamaBatch<'_>,
    sequence: &SequenceState,
    generated_tokens: usize,
    batch_id: usize,
    trace: &mut Vec<TraceEntry>,
) -> Result<Option<i32>, ProbeError> {
    if sequence.generated.len() >= generated_tokens {
        return Ok(None);
    }
    let Some(token) = sequence.pending else {
        return Ok(None);
    };
    let batch_index = batch.n_tokens();
    let position = sequence.next_position()?;
    batch
        .add(token, position, &[sequence.id], true)
        .map_err(|error| ProbeError::operation("add decode token to batch", error))?;
    trace.push(TraceEntry {
        batch_id,
        phase: "decode",
        sequence_id: sequence.id,
        position,
        token_id: token.0,
        requested_logits: true,
    });
    Ok(Some(batch_index))
}

fn sample_next(
    context: &llama_cpp_2::context::LlamaContext<'_>,
    vocab: &llama_cpp_2::vocab::LlamaVocab<'_>,
    sampler: &mut LlamaSampler,
    logits_index: i32,
    sequence: &mut SequenceState,
) {
    let token = sampler.sample(context, logits_index);
    sequence.generated.push(token);
    sequence.pending = (!vocab.is_eog(token)).then_some(token);
}
