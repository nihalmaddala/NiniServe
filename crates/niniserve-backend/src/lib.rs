#![forbid(unsafe_code)]

use std::{error::Error, fmt};

use niniserve_protocol::SequenceId;

mod mock;

pub use mock::MockExecutor;

#[cfg(feature = "llamacpp")]
pub mod llamacpp;

/// Conservative limits reported by a model executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendLimits {
    pub max_batch_tokens: usize,
    pub max_active_sequences: usize,
    pub max_sequence_tokens: usize,
}

/// One token submitted at an explicit model position for an explicit sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchToken {
    pub sequence_id: SequenceId,
    pub token_id: u32,
    pub position: u32,
    pub request_logits: bool,
}

/// Immutable token work selected for one backend call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionPlan {
    pub tokens: Vec<BatchToken>,
}

/// Owned sampled output; no backend logit pointer escapes the executor call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendTokenEvent {
    pub sequence_id: SequenceId,
    pub evaluated_position: u32,
    pub sampled_token_id: u32,
    /// Owned token bytes; callers must aggregate incomplete UTF-8 safely.
    pub text_bytes: Vec<u8>,
    pub is_eog: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u32,
}

/// Synchronous model execution owned by the future dedicated engine worker.
pub trait ModelExecutor: Send {
    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>, BackendError>;

    fn start_sequence(
        &mut self,
        _sequence_id: SequenceId,
        _sampling: SamplingConfig,
    ) -> Result<(), BackendError> {
        Ok(())
    }

    fn execute(&mut self, plan: &ExecutionPlan) -> Result<Vec<BackendTokenEvent>, BackendError>;

    fn release_sequence(&mut self, sequence_id: SequenceId) -> Result<(), BackendError>;

    fn limits(&self) -> BackendLimits;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    EmptyPrompt,
    EmptyPlan,
    BatchTooLarge {
        actual: usize,
        maximum: usize,
    },
    TooManySequences {
        maximum: usize,
    },
    PositionMismatch {
        sequence_id: SequenceId,
        expected: u32,
        actual: u32,
    },
    PositionOverflow {
        sequence_id: SequenceId,
    },
    UnknownSequence(SequenceId),
    Native(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPrompt => formatter.write_str("cannot tokenize an empty prompt"),
            Self::EmptyPlan => formatter.write_str("execution plan contains no tokens"),
            Self::BatchTooLarge { actual, maximum } => {
                write!(formatter, "batch has {actual} tokens; maximum is {maximum}")
            }
            Self::TooManySequences { maximum } => {
                write!(formatter, "active sequence limit of {maximum} was reached")
            }
            Self::PositionMismatch {
                sequence_id,
                expected,
                actual,
            } => write!(
                formatter,
                "sequence {} expected position {expected}, got {actual}",
                sequence_id.0
            ),
            Self::PositionOverflow { sequence_id } => {
                write!(formatter, "sequence {} position overflowed", sequence_id.0)
            }
            Self::UnknownSequence(sequence_id) => {
                write!(formatter, "sequence {} is not active", sequence_id.0)
            }
            Self::Native(message) => formatter.write_str(message),
        }
    }
}

impl Error for BackendError {}
