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
    pub prefill_wait_micros: u128,
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SchedulerConfig {
    Fcfs,
    DecodePriority,
    FixedChunk { prefill_chunk_tokens: usize },
    Adaptive(AdaptiveConfig),
}

pub const DEFAULT_PREFILL_CHUNK_TOKENS: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveConfig {
    pub target_decode_gap_micros: u128,
    pub min_chunk_tokens: usize,
    pub max_chunk_tokens: usize,
    pub initial_chunk_tokens: usize,
    pub increase_step_tokens: usize,
    pub decrease_factor: f64,
    pub ewma_alpha: f64,
    pub low_threshold_factor: f64,
    pub high_threshold_factor: f64,
    pub max_prefill_wait_micros: u128,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            target_decode_gap_micros: 75_000,
            min_chunk_tokens: 32,
            max_chunk_tokens: 512,
            initial_chunk_tokens: 128,
            increase_step_tokens: 32,
            decrease_factor: 0.5,
            ewma_alpha: 0.2,
            low_threshold_factor: 0.85,
            high_threshold_factor: 1.10,
            max_prefill_wait_micros: 500_000,
        }
    }
}

impl AdaptiveConfig {
    pub fn validate(self) -> Result<Self, AdaptiveConfigError> {
        let valid = self.target_decode_gap_micros > 0
            && self.min_chunk_tokens > 0
            && self.min_chunk_tokens <= self.initial_chunk_tokens
            && self.initial_chunk_tokens <= self.max_chunk_tokens
            && self.increase_step_tokens > 0
            && self.decrease_factor.is_finite()
            && (0.0..1.0).contains(&self.decrease_factor)
            && self.ewma_alpha.is_finite()
            && (0.0..=1.0).contains(&self.ewma_alpha)
            && self.low_threshold_factor.is_finite()
            && self.high_threshold_factor.is_finite()
            && self.low_threshold_factor > 0.0
            && self.low_threshold_factor < 1.0
            && self.high_threshold_factor > 1.0
            && self.max_prefill_wait_micros > 0;
        if valid {
            Ok(self)
        } else {
            Err(AdaptiveConfigError)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveConfigError;

impl fmt::Display for AdaptiveConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("invalid adaptive scheduler configuration")
    }
}

impl Error for AdaptiveConfigError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseSchedulerError {
    value: String,
}

impl fmt::Display for ParseSchedulerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "unknown scheduler {:?}; expected fcfs, decode-priority, fixed-chunk, or adaptive",
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
            "adaptive" => Ok(Self::Adaptive(AdaptiveConfig::default())),
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
    pub decode_gap_micros: Option<u128>,
}

pub trait Scheduler {
    fn name(&self) -> &'static str;
    fn plan(&mut self, view: EngineView<'_>, budget: StepBudget) -> SchedulePlan;
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
            SchedulerConfig::Adaptive(_) => {
                unreachable!("adaptive configuration requires AdaptiveScheduler")
            }
        }
    }

    fn plan(&mut self, view: EngineView<'_>, budget: StepBudget) -> SchedulePlan {
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
            SchedulerConfig::Adaptive(_) => {
                unreachable!("adaptive configuration requires AdaptiveScheduler")
            }
        };
        SchedulePlan { work }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveDecision {
    Startup,
    NoSignal,
    Hold,
    Increase,
    Decrease,
    NoDecoder,
    MaxWait,
    FeedbackBudget,
}

impl AdaptiveDecision {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Startup => "startup",
            Self::NoSignal => "no-signal",
            Self::Hold => "hold",
            Self::Increase => "increase",
            Self::Decrease => "decrease",
            Self::NoDecoder => "no-decoder",
            Self::MaxWait => "max-wait",
            Self::FeedbackBudget => "feedback-budget",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveStatus {
    pub current_chunk_tokens: usize,
    pub decode_gap_ewma_micros: Option<f64>,
    pub decision: AdaptiveDecision,
    pub plan_reason: AdaptiveDecision,
}

#[derive(Debug, Clone, Copy)]
pub struct AdaptiveScheduler {
    config: AdaptiveConfig,
    status: AdaptiveStatus,
}

impl AdaptiveScheduler {
    pub fn new(config: AdaptiveConfig) -> Result<Self, AdaptiveConfigError> {
        let config = config.validate()?;
        Ok(Self {
            status: AdaptiveStatus {
                current_chunk_tokens: config.initial_chunk_tokens,
                decode_gap_ewma_micros: None,
                decision: AdaptiveDecision::Startup,
                plan_reason: AdaptiveDecision::Startup,
            },
            config,
        })
    }

    #[must_use]
    pub const fn config(&self) -> AdaptiveConfig {
        self.config
    }

    #[must_use]
    pub const fn status(&self) -> AdaptiveStatus {
        self.status
    }
}

impl Scheduler for AdaptiveScheduler {
    fn name(&self) -> &'static str {
        "adaptive"
    }

    fn plan(&mut self, view: EngineView<'_>, budget: StepBudget) -> SchedulePlan {
        if budget.max_batch_tokens == 0 {
            return SchedulePlan { work: Vec::new() };
        }
        let mut sequences = view.sequences.to_vec();
        sequences.sort_unstable_by_key(|sequence| (sequence.admission_order, sequence.sequence_id));
        let has_decoder = sequences
            .iter()
            .any(|sequence| sequence.phase == SequencePhase::Decode);
        if !has_decoder {
            self.status.plan_reason = AdaptiveDecision::NoDecoder;
            return SchedulePlan {
                work: plan_decode_priority(&sequences, budget.max_batch_tokens, None),
            };
        }
        let max_wait_exceeded = sequences.iter().any(|sequence| {
            sequence.phase == SequencePhase::Prefill
                && sequence.prefill_wait_micros >= self.config.max_prefill_wait_micros
        });
        if max_wait_exceeded {
            self.status.plan_reason = AdaptiveDecision::MaxWait;
            return SchedulePlan {
                work: plan_with_prefill_reservation(
                    &sequences,
                    budget.max_batch_tokens,
                    self.config.min_chunk_tokens,
                ),
            };
        }
        self.status.plan_reason = AdaptiveDecision::FeedbackBudget;
        SchedulePlan {
            work: plan_decode_priority(
                &sequences,
                budget.max_batch_tokens,
                Some(self.status.current_chunk_tokens),
            ),
        }
    }

    fn observe(&mut self, observation: StepObservation) {
        let Some(sample) = observation.decode_gap_micros else {
            self.status.decision = AdaptiveDecision::NoSignal;
            return;
        };
        let sample = sample as f64;
        let ewma = self
            .status
            .decode_gap_ewma_micros
            .map_or(sample, |previous| {
                self.config
                    .ewma_alpha
                    .mul_add(sample, (1.0 - self.config.ewma_alpha) * previous)
            });
        self.status.decode_gap_ewma_micros = Some(ewma);
        let target = self.config.target_decode_gap_micros as f64;
        if ewma > target * self.config.high_threshold_factor {
            let decreased = (self.status.current_chunk_tokens as f64 * self.config.decrease_factor)
                .floor() as usize;
            self.status.current_chunk_tokens = decreased.max(self.config.min_chunk_tokens);
            self.status.decision = AdaptiveDecision::Decrease;
        } else if ewma < target * self.config.low_threshold_factor {
            self.status.current_chunk_tokens = self
                .status
                .current_chunk_tokens
                .saturating_add(self.config.increase_step_tokens)
                .min(self.config.max_chunk_tokens);
            self.status.decision = AdaptiveDecision::Increase;
        } else {
            self.status.decision = AdaptiveDecision::Hold;
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ConfiguredScheduler {
    Baseline(BaselineScheduler),
    Adaptive(AdaptiveScheduler),
}

impl ConfiguredScheduler {
    #[must_use]
    pub fn new(config: SchedulerConfig) -> Self {
        match config {
            SchedulerConfig::Adaptive(config) => Self::Adaptive(
                AdaptiveScheduler::new(config).expect("adaptive scheduler config must be valid"),
            ),
            baseline => Self::Baseline(BaselineScheduler::new(baseline)),
        }
    }

    #[must_use]
    pub const fn adaptive_status(&self) -> Option<AdaptiveStatus> {
        match self {
            Self::Baseline(_) => None,
            Self::Adaptive(scheduler) => Some(scheduler.status()),
        }
    }

    #[must_use]
    pub const fn config(&self) -> SchedulerConfig {
        match self {
            Self::Baseline(scheduler) => scheduler.config(),
            Self::Adaptive(scheduler) => SchedulerConfig::Adaptive(scheduler.config()),
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.config());
    }
}

impl Scheduler for ConfiguredScheduler {
    fn name(&self) -> &'static str {
        match self {
            Self::Baseline(scheduler) => scheduler.name(),
            Self::Adaptive(scheduler) => scheduler.name(),
        }
    }

    fn plan(&mut self, view: EngineView<'_>, budget: StepBudget) -> SchedulePlan {
        match self {
            Self::Baseline(scheduler) => scheduler.plan(view, budget),
            Self::Adaptive(scheduler) => scheduler.plan(view, budget),
        }
    }

    fn observe(&mut self, observation: StepObservation) {
        match self {
            Self::Baseline(scheduler) => scheduler.observe(observation),
            Self::Adaptive(scheduler) => scheduler.observe(observation),
        }
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
    let mut remaining_prefill = prefill_chunk.unwrap_or(remaining).min(remaining);
    for sequence in sequences
        .iter()
        .filter(|sequence| sequence.phase == SequencePhase::Prefill)
    {
        if remaining == 0 || remaining_prefill == 0 {
            break;
        }
        let token_count = sequence
            .remaining_prefill_tokens
            .min(remaining_prefill)
            .min(remaining);
        if token_count > 0 {
            work.push(ScheduledWork {
                sequence_id: sequence.sequence_id,
                token_count,
                kind: WorkKind::Prefill,
            });
            remaining -= token_count;
            remaining_prefill -= token_count;
        }
    }
    work
}

fn plan_with_prefill_reservation(
    sequences: &[SequenceSnapshot],
    budget: usize,
    reservation: usize,
) -> Vec<ScheduledWork> {
    let reserved = reservation.min(budget);
    let mut work = Vec::new();
    if let Some(sequence) = sequences
        .iter()
        .filter(|sequence| sequence.phase == SequencePhase::Prefill)
        .max_by_key(|sequence| {
            (
                sequence.prefill_wait_micros,
                std::cmp::Reverse(sequence.admission_order),
            )
        })
    {
        let token_count = sequence.remaining_prefill_tokens.min(reserved);
        if token_count > 0 {
            work.push(ScheduledWork {
                sequence_id: sequence.sequence_id,
                token_count,
                kind: WorkKind::Prefill,
            });
        }
    }
    let remaining = budget.saturating_sub(work.iter().map(|item| item.token_count).sum::<usize>());
    work.extend(
        sequences
            .iter()
            .filter(|sequence| sequence.phase == SequencePhase::Decode)
            .take(remaining)
            .map(|sequence| ScheduledWork {
                sequence_id: sequence.sequence_id,
                token_count: 1,
                kind: WorkKind::Decode,
            }),
    );
    work
}

#[cfg(test)]
mod tests {
    use super::{
        AdaptiveConfig, AdaptiveDecision, AdaptiveScheduler, BaselineScheduler,
        ConfiguredScheduler, EngineView, SchedulePlan, ScheduledWork, Scheduler, SchedulerConfig,
        SequencePhase, SequenceSnapshot, StepBudget, StepObservation, WorkKind,
    };
    use niniserve_protocol::SequenceId;

    fn sequence(id: u32, order: u64, phase: SequencePhase, remaining: usize) -> SequenceSnapshot {
        SequenceSnapshot {
            sequence_id: SequenceId(id),
            admission_order: order,
            phase,
            remaining_prefill_tokens: remaining,
            prefill_wait_micros: 0,
        }
    }

    fn observation(decode_gap_micros: Option<u128>) -> StepObservation {
        StepObservation {
            scheduled_tokens: 2,
            prefill_tokens: 1,
            decode_tokens: 1,
            scheduling_micros: 10,
            backend_micros: 20,
            decode_gap_micros,
        }
    }

    #[test]
    fn adaptive_starts_at_the_configured_chunk_and_holds_without_a_signal() {
        let mut scheduler = AdaptiveScheduler::new(AdaptiveConfig::default()).unwrap();

        scheduler.observe(observation(None));

        assert_eq!(scheduler.status().current_chunk_tokens, 128);
        assert_eq!(scheduler.status().decision, AdaptiveDecision::NoSignal);
        assert_eq!(scheduler.status().decode_gap_ewma_micros, None);
    }

    #[test]
    fn adaptive_decreases_above_high_threshold_and_recovers_below_low_threshold() {
        let mut scheduler = AdaptiveScheduler::new(AdaptiveConfig::default()).unwrap();

        scheduler.observe(observation(Some(100_000)));
        assert_eq!(scheduler.status().current_chunk_tokens, 64);
        assert_eq!(scheduler.status().decision, AdaptiveDecision::Decrease);

        for _ in 0..20 {
            scheduler.observe(observation(Some(10_000)));
        }
        assert!(scheduler.status().current_chunk_tokens > 64);
        assert_eq!(scheduler.status().decision, AdaptiveDecision::Increase);
    }

    #[test]
    fn adaptive_hysteresis_holds_and_repeated_overload_never_grows() {
        let mut scheduler = AdaptiveScheduler::new(AdaptiveConfig::default()).unwrap();

        scheduler.observe(observation(Some(70_000)));
        assert_eq!(scheduler.status().current_chunk_tokens, 128);
        assert_eq!(scheduler.status().decision, AdaptiveDecision::Hold);

        let mut previous = scheduler.status().current_chunk_tokens;
        for _ in 0..8 {
            scheduler.observe(observation(Some(200_000)));
            let current = scheduler.status().current_chunk_tokens;
            assert!(current <= previous);
            previous = current;
        }
        assert_eq!(previous, 32);
    }

    #[test]
    fn configured_scheduler_reset_restores_adaptive_startup_state() {
        let mut scheduler =
            ConfiguredScheduler::new(SchedulerConfig::Adaptive(AdaptiveConfig::default()));
        scheduler.observe(observation(Some(200_000)));
        assert_eq!(
            scheduler.adaptive_status().unwrap().current_chunk_tokens,
            64
        );

        scheduler.reset();

        let status = scheduler.adaptive_status().unwrap();
        assert_eq!(status.current_chunk_tokens, 128);
        assert_eq!(status.decision, AdaptiveDecision::Startup);
    }

    #[test]
    fn adaptive_uses_the_full_safe_budget_when_no_decoder_is_active() {
        let sequences = [sequence(0, 1, SequencePhase::Prefill, 1_000)];
        let mut scheduler = AdaptiveScheduler::new(AdaptiveConfig::default()).unwrap();

        let plan = scheduler.plan(
            EngineView {
                sequences: &sequences,
            },
            StepBudget {
                max_batch_tokens: 512,
            },
        );

        assert_eq!(plan.work[0].token_count, 512);
    }

    #[test]
    fn adaptive_max_wait_reserves_prefill_progress_under_saturated_decode_load() {
        let mut prefill = sequence(0, 1, SequencePhase::Prefill, 1_000);
        prefill.prefill_wait_micros = 500_000;
        let sequences = [
            prefill,
            sequence(1, 2, SequencePhase::Decode, 0),
            sequence(2, 3, SequencePhase::Decode, 0),
        ];
        let mut scheduler = AdaptiveScheduler::new(AdaptiveConfig::default()).unwrap();

        let plan = scheduler.plan(
            EngineView {
                sequences: &sequences,
            },
            StepBudget {
                max_batch_tokens: 2,
            },
        );

        assert!(plan.work.iter().any(|work| work.kind == WorkKind::Prefill));
        assert_eq!(scheduler.status().plan_reason, AdaptiveDecision::MaxWait);
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
    fn fixed_chunk_caps_total_prefill_work_per_step() {
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
            vec![ScheduledWork {
                sequence_id: SequenceId(0),
                token_count: 3,
                kind: WorkKind::Prefill,
            }]
        );
    }

    #[test]
    fn policy_names_parse_to_their_documented_configs() {
        assert_eq!("fcfs".parse(), Ok(SchedulerConfig::Fcfs));
        assert_eq!(
            "decode-priority".parse(),
            Ok(SchedulerConfig::DecodePriority)
        );
        assert_eq!(
            "adaptive".parse(),
            Ok(SchedulerConfig::Adaptive(AdaptiveConfig::default()))
        );
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
