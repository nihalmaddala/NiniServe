use std::collections::HashMap;

use niniserve_protocol::SequenceId;

use crate::{
    BackendError, BackendLimits, BackendTokenEvent, BatchToken, ExecutionPlan, ModelExecutor,
};

const MOCK_VOCAB_SIZE: u32 = 32_000;

/// Deterministic executor for testing positions, routing, and cleanup.
#[derive(Debug)]
pub struct MockExecutor {
    limits: BackendLimits,
    next_positions: HashMap<SequenceId, u32>,
    trace: Vec<BatchToken>,
}

impl MockExecutor {
    #[must_use]
    pub fn new(limits: BackendLimits) -> Self {
        Self {
            limits,
            next_positions: HashMap::new(),
            trace: Vec::new(),
        }
    }

    #[must_use]
    pub fn active_sequence_count(&self) -> usize {
        self.next_positions.len()
    }

    #[must_use]
    pub fn trace(&self) -> &[BatchToken] {
        &self.trace
    }

    fn sampled_token(token: BatchToken) -> u32 {
        token
            .token_id
            .wrapping_add(token.sequence_id.0)
            .wrapping_add(token.position)
            .wrapping_add(1)
            % MOCK_VOCAB_SIZE
    }
}

impl ModelExecutor for MockExecutor {
    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>, BackendError> {
        if prompt.is_empty() {
            return Err(BackendError::EmptyPrompt);
        }

        Ok(prompt.bytes().map(|byte| u32::from(byte) + 3).collect())
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

        // Validate against a copy so a failed backend call cannot partially
        // advance the mock's model-side sequence state.
        let mut candidate_positions = self.next_positions.clone();
        let mut events = Vec::new();

        for token in &plan.tokens {
            if !candidate_positions.contains_key(&token.sequence_id)
                && candidate_positions.len() >= self.limits.max_active_sequences
            {
                return Err(BackendError::TooManySequences {
                    maximum: self.limits.max_active_sequences,
                });
            }

            let expected = candidate_positions
                .get(&token.sequence_id)
                .copied()
                .unwrap_or(0);
            if token.position != expected {
                return Err(BackendError::PositionMismatch {
                    sequence_id: token.sequence_id,
                    expected,
                    actual: token.position,
                });
            }

            let next_position =
                token
                    .position
                    .checked_add(1)
                    .ok_or(BackendError::PositionOverflow {
                        sequence_id: token.sequence_id,
                    })?;
            candidate_positions.insert(token.sequence_id, next_position);

            if token.request_logits {
                let sampled_token_id = Self::sampled_token(*token);
                events.push(BackendTokenEvent {
                    sequence_id: token.sequence_id,
                    evaluated_position: token.position,
                    sampled_token_id,
                    text_bytes: format!("<{sampled_token_id}>").into_bytes(),
                    is_eog: false,
                });
            }
        }

        self.next_positions = candidate_positions;
        self.trace.extend_from_slice(&plan.tokens);
        Ok(events)
    }

    fn release_sequence(&mut self, sequence_id: SequenceId) -> Result<(), BackendError> {
        self.next_positions
            .remove(&sequence_id)
            .map(|_| ())
            .ok_or(BackendError::UnknownSequence(sequence_id))
    }

    fn limits(&self) -> BackendLimits {
        self.limits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn executor() -> MockExecutor {
        MockExecutor::new(BackendLimits {
            max_batch_tokens: 4,
            max_active_sequences: 2,
            max_sequence_tokens: 32,
        })
    }

    fn token(sequence: u32, token_id: u32, position: u32, request_logits: bool) -> BatchToken {
        BatchToken {
            sequence_id: SequenceId(sequence),
            token_id,
            position,
            request_logits,
        }
    }

    #[test]
    fn tokenization_is_nonempty_and_deterministic() {
        let executor = executor();
        let first = executor
            .tokenize("hello")
            .expect("tokenization should work");
        let second = executor
            .tokenize("hello")
            .expect("tokenization should work");

        assert_eq!(first, second);
        assert_eq!(first.len(), 5);
        assert_eq!(executor.tokenize(""), Err(BackendError::EmptyPrompt));
    }

    #[test]
    fn routes_logits_to_independent_sequences() {
        let mut executor = executor();
        let plan = ExecutionPlan {
            tokens: vec![
                token(10, 100, 0, false),
                token(20, 200, 0, false),
                token(10, 101, 1, true),
                token(20, 201, 1, true),
            ],
        };

        let events = executor.execute(&plan).expect("valid plan should execute");

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].sequence_id, SequenceId(10));
        assert_eq!(events[0].evaluated_position, 1);
        assert_eq!(events[1].sequence_id, SequenceId(20));
        assert_eq!(events[1].evaluated_position, 1);
        assert_ne!(events[0].sampled_token_id, events[1].sampled_token_id);
        assert_eq!(events[0].text_bytes, b"<113>");
        assert!(!events[0].is_eog);
        assert_eq!(executor.active_sequence_count(), 2);
        assert_eq!(executor.trace(), plan.tokens.as_slice());
    }

    #[test]
    fn failed_plan_does_not_partially_advance_sequence_state() {
        let mut executor = executor();
        let invalid = ExecutionPlan {
            tokens: vec![token(1, 10, 0, false), token(1, 11, 3, true)],
        };

        assert_eq!(
            executor.execute(&invalid),
            Err(BackendError::PositionMismatch {
                sequence_id: SequenceId(1),
                expected: 1,
                actual: 3,
            })
        );
        assert_eq!(executor.active_sequence_count(), 0);
        assert!(executor.trace().is_empty());

        let valid = ExecutionPlan {
            tokens: vec![token(1, 10, 0, true)],
        };
        assert!(executor.execute(&valid).is_ok());
    }

    #[test]
    fn rejects_oversized_batch_without_creating_sequences() {
        let mut executor = executor();
        let oversized = ExecutionPlan {
            tokens: (0..5)
                .map(|position| token(1, 10 + position, position, position == 4))
                .collect(),
        };

        assert_eq!(
            executor.execute(&oversized),
            Err(BackendError::BatchTooLarge {
                actual: 5,
                maximum: 4,
            })
        );
        assert_eq!(executor.active_sequence_count(), 0);
        assert!(executor.trace().is_empty());
    }

    #[test]
    fn release_frees_sequence_capacity_exactly_once() {
        let mut executor = executor();
        executor
            .execute(&ExecutionPlan {
                tokens: vec![token(1, 10, 0, true), token(2, 20, 0, true)],
            })
            .expect("two sequences fit");

        let third = ExecutionPlan {
            tokens: vec![token(3, 30, 0, true)],
        };
        assert_eq!(
            executor.execute(&third),
            Err(BackendError::TooManySequences { maximum: 2 })
        );

        assert_eq!(executor.release_sequence(SequenceId(1)), Ok(()));
        assert!(executor.execute(&third).is_ok());
        assert_eq!(
            executor.release_sequence(SequenceId(1)),
            Err(BackendError::UnknownSequence(SequenceId(1)))
        );
    }
}
