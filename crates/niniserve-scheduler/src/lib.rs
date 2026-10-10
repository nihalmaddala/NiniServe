#![forbid(unsafe_code)]

use std::{error::Error, fmt, str::FromStr};

use niniserve_protocol::SequenceId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequencePhase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceSnapshot {
    pub sequence_id: SequenceId,
    pub admission_order: u64,
    pub phase: SequencePhase,
    pub remaining_prefill_tokens: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct EngineView<'a> {
    pub sequences: &'a [SequenceSnapshot],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepBudget {
    pub max_batch_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkKind {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledWork {
    pub sequence_id: SequenceId,
    pub token_count: usize,
    pub kind: WorkKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulePlan {
    pub work: Vec<ScheduledWork>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerConfig {
    Fcfs,
    DecodePriority,
    FixedChunk { prefill_chunk_tokens: usize },
}

pub const DEFAULT_PREFILL_CHUNK_TOKENS: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseSchedulerError {
    value: String,
}

impl fmt::Display for ParseSchedulerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "unknown scheduler {:?}; expected fcfs, decode-priority, or fixed-chunk",
            self.value
        )
    }
}

impl Error for ParseSchedulerError {}

impl FromStr for SchedulerConfig {
    type Err = ParseSchedulerError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "fcfs" => Ok(Self::Fcfs),
            "decode-priority" => Ok(Self::DecodePriority),
            "fixed-chunk" => Ok(Self::FixedChunk {
                prefill_chunk_tokens: DEFAULT_PREFILL_CHUNK_TOKENS,
            }),
            _ => Err(ParseSchedulerError {
                value: value.to_owned(),
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepObservation {
    pub scheduled_tokens: usize,
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
    pub scheduling_micros: u128,
    pub backend_micros: u128,
}

pub trait Scheduler {
    fn name(&self) -> &'static str;
    fn plan(&self, view: EngineView<'_>, budget: StepBudget) -> SchedulePlan;
    fn observe(&mut self, _observation: StepObservation) {}
}

#[derive(Debug, Clone, Copy)]
pub struct BaselineScheduler {
    config: SchedulerConfig,
}

impl BaselineScheduler {
    #[must_use]
    pub fn new(config: SchedulerConfig) -> Self {
        if let SchedulerConfig::FixedChunk {
            prefill_chunk_tokens,
        } = config
        {
            assert!(
                prefill_chunk_tokens > 0,
                "prefill chunk size must be positive"
            );
        }
        Self { config }
    }

    #[must_use]
    pub const fn config(&self) -> SchedulerConfig {
        self.config
    }
}

impl Scheduler for BaselineScheduler {
    fn name(&self) -> &'static str {
        match self.config {
            SchedulerConfig::Fcfs => "fcfs",
            SchedulerConfig::DecodePriority => "decode-priority",
            SchedulerConfig::FixedChunk { .. } => "fixed-chunk",
        }
    }

    fn plan(&self, view: EngineView<'_>, budget: StepBudget) -> SchedulePlan {
        if budget.max_batch_tokens == 0 {
            return SchedulePlan { work: Vec::new() };
        }
        let mut sequences = view.sequences.to_vec();
        sequences.sort_unstable_by_key(|sequence| (sequence.admission_order, sequence.sequence_id));
        let work = match self.config {
            SchedulerConfig::Fcfs => plan_fcfs(&sequences, budget.max_batch_tokens),
            SchedulerConfig::DecodePriority => {
                plan_decode_priority(&sequences, budget.max_batch_tokens, None)
            }
            SchedulerConfig::FixedChunk {
                prefill_chunk_tokens,
            } => plan_decode_priority(
                &sequences,
                budget.max_batch_tokens,
                Some(prefill_chunk_tokens),
            ),
        };
        SchedulePlan { work }
    }
}

fn plan_fcfs(sequences: &[SequenceSnapshot], budget: usize) -> Vec<ScheduledWork> {
    sequences
        .first()
        .map(|sequence| ScheduledWork {
            sequence_id: sequence.sequence_id,
            token_count: match sequence.phase {
                SequencePhase::Prefill => sequence.remaining_prefill_tokens.min(budget),
                SequencePhase::Decode => 1,
            },
            kind: match sequence.phase {
                SequencePhase::Prefill => WorkKind::Prefill,
                SequencePhase::Decode => WorkKind::Decode,
            },
        })
        .filter(|work| work.token_count > 0)
        .into_iter()
        .collect()
}

fn plan_decode_priority(
    sequences: &[SequenceSnapshot],
    budget: usize,
    prefill_chunk: Option<usize>,
) -> Vec<ScheduledWork> {
    let mut remaining = budget;
    let mut work = Vec::new();
    for sequence in sequences
        .iter()
        .filter(|sequence| sequence.phase == SequencePhase::Decode)
    {
        if remaining == 0 {
            return work;
        }
        work.push(ScheduledWork {
            sequence_id: sequence.sequence_id,
            token_count: 1,
            kind: WorkKind::Decode,
        });
        remaining -= 1;
    }
    for sequence in sequences
        .iter()
        .filter(|sequence| sequence.phase == SequencePhase::Prefill)
    {
        if remaining == 0 {
            break;
        }
        let token_count = sequence
            .remaining_prefill_tokens
            .min(prefill_chunk.unwrap_or(remaining))
            .min(remaining);
        if token_count > 0 {
            work.push(ScheduledWork {
                sequence_id: sequence.sequence_id,
                token_count,
                kind: WorkKind::Prefill,
            });
            remaining -= token_count;
        }
    }
    work
}

#[cfg(test)]
mod tests {
    use super::{
        BaselineScheduler, EngineView, SchedulePlan, ScheduledWork, Scheduler, SchedulerConfig,
        SequencePhase, SequenceSnapshot, StepBudget, WorkKind,
    };
    use niniserve_protocol::SequenceId;

    fn sequence(id: u32, order: u64, phase: SequencePhase, remaining: usize) -> SequenceSnapshot {
        SequenceSnapshot {
            sequence_id: SequenceId(id),
            admission_order: order,
            phase,
            remaining_prefill_tokens: remaining,
        }
    }

    #[test]
    fn fcfs_only_schedules_the_oldest_request() {
        let sequences = [
            sequence(0, 20, SequencePhase::Decode, 0),
            sequence(1, 10, SequencePhase::Prefill, 7),
        ];
        let plan = BaselineScheduler::new(SchedulerConfig::Fcfs).plan(
            EngineView {
                sequences: &sequences,
            },
            StepBudget {
                max_batch_tokens: 4,
            },
        );

        assert_eq!(
            plan,
            SchedulePlan {
                work: vec![ScheduledWork {
                    sequence_id: SequenceId(1),
                    token_count: 4,
                    kind: WorkKind::Prefill,
                }]
            }
        );
    }

    #[test]
    fn decode_priority_schedules_decodes_before_prefill() {
        let sequences = [
            sequence(0, 1, SequencePhase::Prefill, 5),
            sequence(1, 2, SequencePhase::Decode, 0),
            sequence(2, 3, SequencePhase::Decode, 0),
        ];
        let plan = BaselineScheduler::new(SchedulerConfig::DecodePriority).plan(
            EngineView {
                sequences: &sequences,
            },
            StepBudget {
                max_batch_tokens: 4,
            },
        );

        assert_eq!(
            plan.work,
            vec![
                ScheduledWork {
                    sequence_id: SequenceId(1),
                    token_count: 1,
                    kind: WorkKind::Decode,
                },
                ScheduledWork {
                    sequence_id: SequenceId(2),
                    token_count: 1,
                    kind: WorkKind::Decode,
                },
                ScheduledWork {
                    sequence_id: SequenceId(0),
                    token_count: 2,
                    kind: WorkKind::Prefill,
                },
            ]
        );
    }

    #[test]
    fn fixed_chunk_caps_each_prefill_allocation() {
        let sequences = [
            sequence(0, 1, SequencePhase::Prefill, 8),
            sequence(1, 2, SequencePhase::Prefill, 8),
        ];
        let plan = BaselineScheduler::new(SchedulerConfig::FixedChunk {
            prefill_chunk_tokens: 3,
        })
        .plan(
            EngineView {
                sequences: &sequences,
            },
            StepBudget {
                max_batch_tokens: 5,
            },
        );

        assert_eq!(
            plan.work,
            vec![
                ScheduledWork {
                    sequence_id: SequenceId(0),
                    token_count: 3,
                    kind: WorkKind::Prefill,
                },
                ScheduledWork {
                    sequence_id: SequenceId(1),
                    token_count: 2,
                    kind: WorkKind::Prefill,
                },
            ]
        );
    }

    #[test]
    fn policy_names_parse_to_their_documented_configs() {
        assert_eq!("fcfs".parse(), Ok(SchedulerConfig::Fcfs));
        assert_eq!(
            "decode-priority".parse(),
            Ok(SchedulerConfig::DecodePriority)
        );
        assert!("adaptive".parse::<SchedulerConfig>().is_err());
    }

    #[test]
    fn admission_order_is_the_deterministic_tie_breaker() {
        let sequences = [
            sequence(7, 2, SequencePhase::Decode, 0),
            sequence(9, 1, SequencePhase::Decode, 0),
            sequence(3, 1, SequencePhase::Decode, 0),
        ];

        let plan = BaselineScheduler::new(SchedulerConfig::DecodePriority).plan(
            EngineView {
                sequences: &sequences,
            },
            StepBudget {
                max_batch_tokens: 3,
            },
        );

        assert_eq!(
            plan.work
                .iter()
                .map(|work| work.sequence_id)
                .collect::<Vec<_>>(),
            vec![SequenceId(3), SequenceId(9), SequenceId(7)]
        );
    }

    #[test]
    fn decode_priority_documents_prefill_starvation_at_a_saturated_budget() {
        let sequences = [
            sequence(0, 1, SequencePhase::Prefill, 20),
            sequence(1, 2, SequencePhase::Decode, 0),
            sequence(2, 3, SequencePhase::Decode, 0),
        ];

        let plan = BaselineScheduler::new(SchedulerConfig::DecodePriority).plan(
            EngineView {
                sequences: &sequences,
            },
            StepBudget {
                max_batch_tokens: 2,
            },
        );

        assert!(plan.work.iter().all(|work| work.kind == WorkKind::Decode));
    }
}
