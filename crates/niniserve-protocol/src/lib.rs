#![forbid(unsafe_code)]

use std::{error::Error, fmt};

/// Stable identifier for one client generation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestId(pub u64);

/// Identifier for model-side state owned by one active request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SequenceId(pub u32);

/// Model-independent generation parameters after wire-format parsing.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationRequest {
    pub id: RequestId,
    pub prompt: String,
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: Option<u64>,
}

/// Why generation ended normally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

/// Model-independent events delivered from the engine to one request stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationEvent {
    Started {
        request_id: RequestId,
    },
    Token {
        request_id: RequestId,
        token_id: u32,
        text: String,
    },
    Completed {
        request_id: RequestId,
        finish_reason: FinishReason,
    },
    Error {
        request_id: RequestId,
        message: String,
    },
}

impl GenerationEvent {
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed { .. } | Self::Error { .. })
    }
}

/// Server-enforced request bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestLimits {
    pub max_prompt_bytes: usize,
    pub max_new_tokens: u32,
}

impl RequestLimits {
    /// Validates values that do not require model tokenization.
    pub fn validate(self, request: &GenerationRequest) -> Result<(), ValidationError> {
        if request.prompt.trim().is_empty() {
            return Err(ValidationError::EmptyPrompt);
        }

        let prompt_bytes = request.prompt.len();
        if prompt_bytes > self.max_prompt_bytes {
            return Err(ValidationError::PromptTooLarge {
                actual: prompt_bytes,
                maximum: self.max_prompt_bytes,
            });
        }

        if request.max_new_tokens == 0 || request.max_new_tokens > self.max_new_tokens {
            return Err(ValidationError::MaxNewTokensOutOfRange {
                requested: request.max_new_tokens,
                maximum: self.max_new_tokens,
            });
        }

        if !request.temperature.is_finite() || request.temperature < 0.0 {
            return Err(ValidationError::InvalidTemperature(request.temperature));
        }

        if !request.top_p.is_finite() || !(0.0 < request.top_p && request.top_p <= 1.0) {
            return Err(ValidationError::InvalidTopP(request.top_p));
        }

        Ok(())
    }
}

/// Request validation failure suitable for mapping to an API error later.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    EmptyPrompt,
    PromptTooLarge { actual: usize, maximum: usize },
    MaxNewTokensOutOfRange { requested: u32, maximum: u32 },
    InvalidTemperature(f32),
    InvalidTopP(f32),
}

impl fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPrompt => formatter.write_str("prompt must not be empty"),
            Self::PromptTooLarge { actual, maximum } => {
                write!(formatter, "prompt has {actual} bytes; maximum is {maximum}")
            }
            Self::MaxNewTokensOutOfRange { requested, maximum } => write!(
                formatter,
                "max_new_tokens is {requested}; expected a value from 1 through {maximum}"
            ),
            Self::InvalidTemperature(value) => {
                write!(
                    formatter,
                    "temperature must be finite and nonnegative; got {value}"
                )
            }
            Self::InvalidTopP(value) => {
                write!(
                    formatter,
                    "top_p must be finite, greater than 0, and at most 1; got {value}"
                )
            }
        }
    }
}

impl Error for ValidationError {}

/// Model-independent lifecycle state for one accepted request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    Queued,
    Prefilling,
    Decoding,
    Completed,
    Cancelled,
    TimedOut,
    Failed,
}

impl RequestState {
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Cancelled | Self::TimedOut | Self::Failed
        )
    }

    /// Applies one legal lifecycle transition.
    pub fn transition(self, next: Self) -> Result<Self, LifecycleError> {
        let allowed = matches!(
            (self, next),
            (Self::Queued, Self::Prefilling)
                | (Self::Prefilling, Self::Decoding)
                | (Self::Decoding, Self::Completed)
                | (
                    Self::Queued | Self::Prefilling | Self::Decoding,
                    Self::Cancelled
                )
                | (
                    Self::Queued | Self::Prefilling | Self::Decoding,
                    Self::TimedOut
                )
                | (
                    Self::Queued | Self::Prefilling | Self::Decoding,
                    Self::Failed
                )
        );

        allowed.then_some(next).ok_or(LifecycleError {
            current: self,
            requested: next,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LifecycleError {
    pub current: RequestState,
    pub requested: RequestState,
}

impl fmt::Display for LifecycleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "cannot transition request from {:?} to {:?}",
            self.current, self.requested
        )
    }
}

impl Error for LifecycleError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_request() -> GenerationRequest {
        GenerationRequest {
            id: RequestId(7),
            prompt: "Explain batching.".to_owned(),
            max_new_tokens: 32,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(42),
        }
    }

    fn limits() -> RequestLimits {
        RequestLimits {
            max_prompt_bytes: 1_024,
            max_new_tokens: 256,
        }
    }

    #[test]
    fn accepts_a_request_within_bounds() {
        assert_eq!(limits().validate(&valid_request()), Ok(()));
    }

    #[test]
    fn rejects_empty_and_non_finite_inputs() {
        let mut request = valid_request();
        request.prompt = "   ".to_owned();
        assert_eq!(
            limits().validate(&request),
            Err(ValidationError::EmptyPrompt)
        );

        request.prompt = "valid".to_owned();
        request.temperature = f32::NAN;
        assert!(matches!(
            limits().validate(&request),
            Err(ValidationError::InvalidTemperature(value)) if value.is_nan()
        ));

        request.temperature = 0.0;
        request.top_p = 1.1;
        assert_eq!(
            limits().validate(&request),
            Err(ValidationError::InvalidTopP(1.1))
        );

        request.top_p = 0.0;
        assert_eq!(
            limits().validate(&request),
            Err(ValidationError::InvalidTopP(0.0))
        );
    }

    #[test]
    fn rejects_prompt_and_output_limits() {
        let mut request = valid_request();
        request.prompt = "x".repeat(1_025);
        assert_eq!(
            limits().validate(&request),
            Err(ValidationError::PromptTooLarge {
                actual: 1_025,
                maximum: 1_024,
            })
        );

        request.prompt = "valid".to_owned();
        request.max_new_tokens = 257;
        assert_eq!(
            limits().validate(&request),
            Err(ValidationError::MaxNewTokensOutOfRange {
                requested: 257,
                maximum: 256,
            })
        );
    }

    #[test]
    fn lifecycle_allows_progress_and_rejects_reentry() {
        let state = RequestState::Queued
            .transition(RequestState::Prefilling)
            .expect("queued request should enter prefill")
            .transition(RequestState::Decoding)
            .expect("prefilled request should enter decode")
            .transition(RequestState::Completed)
            .expect("decoded request should complete");

        assert!(state.is_terminal());
        assert_eq!(
            state.transition(RequestState::Decoding),
            Err(LifecycleError {
                current: RequestState::Completed,
                requested: RequestState::Decoding,
            })
        );
    }
}
