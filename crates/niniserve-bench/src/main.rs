#![forbid(unsafe_code)]

use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use niniserve_backend::{
    ModelExecutor,
    llamacpp::{LlamaCppConfig, LlamaCppExecutor},
};
use niniserve_engine::{
    AdaptiveConfig, EngineConfig, EngineHandle, EngineStepObservation, SchedulerConfig,
};
use niniserve_protocol::{FinishReason, GenerationEvent, GenerationRequest, RequestId};
use serde::Serialize;

const BACKEND_VERSION: &str = "llama-cpp-2 0.1.159";

#[derive(Debug)]
enum Cli {
    Run(RunConfig),
    Summary(PathBuf),
    Plot(PathBuf),
}

#[derive(Debug)]
struct RunConfig {
    model: PathBuf,
    scheduler: SchedulerConfig,
    workload: Workload,
    output: Option<PathBuf>,
    repeats: usize,
}

#[derive(Debug, Clone, Copy)]
enum Workload {
    W0,
    W1,
    W2,
    W3,
    W5,
}

impl Workload {
    fn parse(value: &str) -> Result<Self, String> {
        match value.to_ascii_lowercase().as_str() {
            "w0" => Ok(Self::W0),
            "w1" => Ok(Self::W1),
            "w2" => Ok(Self::W2),
            "w3" => Ok(Self::W3),
            "w5" => Ok(Self::W5),
            _ => Err(format!(
                "unknown workload {value:?}; expected W0, W1, W2, W3, or W5"
            )),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::W0 => "W0_single_request",
            Self::W1 => "W1_concurrent_short",
            Self::W2 => "W2_long_prompt_interference",
            Self::W3 => "W3_deterministic_mixed",
            Self::W5 => "W5_load_transition",
        }
    }
}

#[derive(Debug, Clone)]
struct RequestSpec {
    id: RequestId,
    prompt: String,
    prompt_tokens: usize,
    max_new_tokens: u32,
    arrival_delay_ms: u64,
}

#[derive(Debug, Serialize)]
struct Metadata<'a> {
    run_id: &'a str,
    git_commit: String,
    model_name: String,
    backend_version: &'static str,
    os: String,
    hardware: String,
    scheduler: &'static str,
    scheduler_config: SchedulerMetadata,
    workload: &'static str,
    warmup_requests: usize,
    measured_requests: usize,
    repeats: usize,
    model_load_ms: u128,
    started_at_unix_seconds: u64,
}

#[derive(Debug, Serialize)]
struct SchedulerMetadata {
    prefill_chunk_tokens: Option<usize>,
    target_decode_gap_micros: Option<u128>,
    min_chunk_tokens: Option<usize>,
    max_chunk_tokens: Option<usize>,
    initial_chunk_tokens: Option<usize>,
    increase_step_tokens: Option<usize>,
    decrease_factor: Option<f64>,
    ewma_alpha: Option<f64>,
    low_threshold_factor: Option<f64>,
    high_threshold_factor: Option<f64>,
    max_prefill_wait_micros: Option<u128>,
}

#[derive(Debug)]
struct RequestResult {
    request_id: u64,
    policy: &'static str,
    prompt_tokens: usize,
    output_tokens: usize,
    arrival_ms: u128,
    queue_wait_ms: Option<u128>,
    ttft_ms: Option<u128>,
    e2e_ms: u128,
    finished: bool,
    finish_reason: String,
    token_gaps: Vec<TokenGap>,
}

#[derive(Debug)]
struct TokenGap {
    request_id: u64,
    token_index: usize,
    ready_at_ms: u128,
    itl_ms: Option<u128>,
}

#[derive(Debug, Serialize)]
struct Summary {
    request_count: usize,
    finished_count: usize,
    failed_count: usize,
    total_output_tokens: usize,
    elapsed_window_ms: u128,
    queue_wait_p95_ms: Option<u128>,
    queue_wait_max_ms: Option<u128>,
    ttft_p50_ms: Option<u128>,
    ttft_p95_ms: Option<u128>,
    itl_sample_count: usize,
    itl_p50_ms: Option<u128>,
    itl_p95_ms: Option<u128>,
    e2e_p50_ms: Option<u128>,
    e2e_p95_ms: Option<u128>,
    output_tokens_per_second: Option<f64>,
    requests_per_second: Option<f64>,
}

#[derive(Debug)]
struct SummarySamples {
    queue_wait: Vec<u128>,
    ttft: Vec<u128>,
    itl: Vec<u128>,
    e2e: Vec<u128>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    match parse_cli(env::args().skip(1))? {
        Cli::Run(config) => run(config).await,
        Cli::Summary(input) => summarize(&input),
        Cli::Plot(input) => plot(&input),
    }
}

fn parse_cli(mut arguments: impl Iterator<Item = String>) -> Result<Cli, String> {
    match arguments.next().as_deref() {
        Some("run") => {
            let mut model = None;
            let mut scheduler = SchedulerConfig::DecodePriority;
            let mut workload = None;
            let mut output = None;
            let mut chunk = None;
            let mut repeats = 1;
            let mut adaptive = AdaptiveConfig::default();
            let mut adaptive_changed = false;
            while let Some(argument) = arguments.next() {
                match argument.as_str() {
                    "--model" => model = Some(PathBuf::from(next(&mut arguments, "--model")?)),
                    "--scheduler" => {
                        scheduler = next(&mut arguments, "--scheduler")?
                            .parse()
                            .map_err(|error| format!("invalid --scheduler: {error}"))?;
                    }
                    "--prefill-chunk-tokens" => {
                        chunk = Some(parse_positive(
                            &next(&mut arguments, "--prefill-chunk-tokens")?,
                            "--prefill-chunk-tokens",
                        )?);
                    }
                    "--repeats" => {
                        repeats = parse_positive(
                            &next(&mut arguments, "--repeats")?,
                            "--repeats",
                        )?;
                    }
                    "--target-itl-ms" => {
                        adaptive.target_decode_gap_micros = parse_positive_u128(
                            &next(&mut arguments, "--target-itl-ms")?,
                            "--target-itl-ms",
                        )?
                        .checked_mul(1_000)
                        .ok_or("--target-itl-ms is too large")?;
                        adaptive_changed = true;
                    }
                    "--adaptive-min-chunk-tokens" => {
                        adaptive.min_chunk_tokens = parse_positive(
                            &next(&mut arguments, "--adaptive-min-chunk-tokens")?,
                            "--adaptive-min-chunk-tokens",
                        )?;
                        adaptive_changed = true;
                    }
                    "--adaptive-max-chunk-tokens" => {
                        adaptive.max_chunk_tokens = parse_positive(
                            &next(&mut arguments, "--adaptive-max-chunk-tokens")?,
                            "--adaptive-max-chunk-tokens",
                        )?;
                        adaptive_changed = true;
                    }
                    "--adaptive-initial-chunk-tokens" => {
                        adaptive.initial_chunk_tokens = parse_positive(
                            &next(&mut arguments, "--adaptive-initial-chunk-tokens")?,
                            "--adaptive-initial-chunk-tokens",
                        )?;
                        adaptive_changed = true;
                    }
                    "--adaptive-increase-step-tokens" => {
                        adaptive.increase_step_tokens = parse_positive(
                            &next(&mut arguments, "--adaptive-increase-step-tokens")?,
                            "--adaptive-increase-step-tokens",
                        )?;
                        adaptive_changed = true;
                    }
                    "--adaptive-decrease-factor" => {
                        adaptive.decrease_factor = parse_number(
                            &next(&mut arguments, "--adaptive-decrease-factor")?,
                            "--adaptive-decrease-factor",
                        )?;
                        adaptive_changed = true;
                    }
                    "--adaptive-ewma-alpha" => {
                        adaptive.ewma_alpha = parse_number(
                            &next(&mut arguments, "--adaptive-ewma-alpha")?,
                            "--adaptive-ewma-alpha",
                        )?;
                        adaptive_changed = true;
                    }
                    "--adaptive-low-threshold-factor" => {
                        adaptive.low_threshold_factor = parse_number(
                            &next(&mut arguments, "--adaptive-low-threshold-factor")?,
                            "--adaptive-low-threshold-factor",
                        )?;
                        adaptive_changed = true;
                    }
                    "--adaptive-high-threshold-factor" => {
                        adaptive.high_threshold_factor = parse_number(
                            &next(&mut arguments, "--adaptive-high-threshold-factor")?,
                            "--adaptive-high-threshold-factor",
                        )?;
                        adaptive_changed = true;
                    }
                    "--adaptive-max-prefill-wait-ms" => {
                        adaptive.max_prefill_wait_micros = parse_positive_u128(
                            &next(&mut arguments, "--adaptive-max-prefill-wait-ms")?,
                            "--adaptive-max-prefill-wait-ms",
                        )?
                        .checked_mul(1_000)
                        .ok_or("--adaptive-max-prefill-wait-ms is too large")?;
                        adaptive_changed = true;
                    }
                    "--workload" => {
                        workload = Some(Workload::parse(&next(&mut arguments, "--workload")?)?);
                    }
                    "--output" => output = Some(PathBuf::from(next(&mut arguments, "--output")?)),
                    _ => return Err(format!("unknown argument {argument:?}")),
                }
            }
            scheduler = match (scheduler, chunk, adaptive_changed) {
                (SchedulerConfig::FixedChunk { prefill_chunk_tokens }, None, false) => {
                    SchedulerConfig::FixedChunk { prefill_chunk_tokens }
                }
                (SchedulerConfig::FixedChunk { .. }, Some(prefill_chunk_tokens), false) => {
                    SchedulerConfig::FixedChunk { prefill_chunk_tokens }
                }
                (SchedulerConfig::Adaptive(_), None, _) => SchedulerConfig::Adaptive(
                    adaptive.validate().map_err(|error| error.to_string())?,
                ),
                (scheduler, None, false) => scheduler,
                (_, Some(_), _) => {
                    return Err(
                        "--prefill-chunk-tokens requires --scheduler fixed-chunk".to_owned(),
                    );
                }
                (_, None, true) => {
                    return Err("adaptive tuning flags require --scheduler adaptive".to_owned());
                }
            };
            Ok(Cli::Run(RunConfig {
                model: model.ok_or("run requires --model MODEL.gguf")?,
                scheduler,
                workload: workload.ok_or("run requires --workload W0|W1|W2|W3|W5")?,
                output,
                repeats,
            }))
        }
        Some("summary") => Ok(Cli::Summary(PathBuf::from(next(
            &mut arguments,
            "summary",
        )?))),
        Some("plot") => Ok(Cli::Plot(PathBuf::from(next(&mut arguments, "plot")?))),
        _ => Err("usage: niniserve-bench run --model MODEL.gguf --scheduler POLICY --workload W0|W1|W2|W3|W5 [--repeats N] [--output DIR] | summary RUN_DIR | plot RUN_DIR".to_owned()),
    }
}

fn next(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_positive(value: &str, flag: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {flag}: {error}"))?;
    if parsed == 0 {
        return Err(format!("{flag} must be positive"));
    }
    Ok(parsed)
}

fn parse_positive_u128(value: &str, flag: &str) -> Result<u128, String> {
    let parsed = value
        .parse::<u128>()
        .map_err(|error| format!("invalid {flag}: {error}"))?;
    if parsed == 0 {
        return Err(format!("{flag} must be positive"));
    }
    Ok(parsed)
}

fn parse_number(value: &str, flag: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|error| format!("invalid {flag}: {error}"))
}

async fn run(config: RunConfig) -> Result<(), Box<dyn Error>> {
    let started_at_unix_seconds = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let run_id = format!(
        "{}-{}-{}",
        started_at_unix_seconds,
        scheduler_name(config.scheduler),
        config
            .workload
            .name()
            .split('_')
            .next()
            .unwrap_or("workload")
    );
    let output = config
        .output
        .unwrap_or_else(|| PathBuf::from("results").join(&run_id));
    fs::create_dir_all(&output)?;

    let backend_config = LlamaCppConfig {
        model_path: config.model.clone(),
        n_ctx: 8_192,
        n_batch: 512,
        n_ubatch: 128,
        n_seq_max: 4,
        gpu_layers: u32::MAX,
    };
    let load_started = Instant::now();
    let executor = LlamaCppExecutor::load(&backend_config)?;
    let model_load_ms = load_started.elapsed().as_millis();
    let requests = workload_requests(config.workload, &executor)?;
    let measured_requests = requests.len() * config.repeats;
    let engine = EngineHandle::spawn(
        executor,
        EngineConfig {
            command_capacity: 32,
            pending_capacity: 32,
            event_capacity: 256,
            request_timeout: Duration::from_secs(180),
            scheduler: config.scheduler,
        },
    );
    let warmup = requests
        .iter()
        .cloned()
        .map(|mut request| {
            request.id = RequestId(request.id.0 + 10_000);
            request
        })
        .collect();
    let serial = matches!(config.workload, Workload::W0);
    execute_requests(&engine, warmup, config.scheduler, serial).await?;
    engine.reset_scheduler()?;
    let _ = engine.take_step_observations();
    let measured_started = Instant::now();
    let mut results = Vec::with_capacity(measured_requests);
    for repeat in 0..config.repeats {
        if repeat > 0 {
            engine.reset_scheduler()?;
        }
        let offset_ms = measured_started.elapsed().as_millis();
        let repeated = requests
            .iter()
            .cloned()
            .map(|mut request| {
                request.id = RequestId(request.id.0 + repeat as u64 * 1_000);
                request
            })
            .collect();
        let mut repetition = execute_requests(&engine, repeated, config.scheduler, serial).await?;
        for result in &mut repetition {
            result.arrival_ms += offset_ms;
            for gap in &mut result.token_gaps {
                gap.ready_at_ms += offset_ms;
            }
        }
        results.extend(repetition);
    }
    let steps = engine.take_step_observations();
    let metadata = Metadata {
        run_id: &run_id,
        git_commit: command_output("git", &["rev-parse", "HEAD"]),
        model_name: config.model.file_name().map_or_else(
            || config.model.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        ),
        backend_version: BACKEND_VERSION,
        os: format!("{}-{}", env::consts::OS, env::consts::ARCH),
        hardware: command_output("sysctl", &["-n", "machdep.cpu.brand_string"]),
        scheduler: scheduler_name(config.scheduler),
        scheduler_config: scheduler_metadata(config.scheduler),
        workload: config.workload.name(),
        warmup_requests: requests.len(),
        measured_requests,
        repeats: config.repeats,
        model_load_ms,
        started_at_unix_seconds,
    };
    write_json(&output.join("metadata.json"), &metadata)?;
    write_requests(&output.join("requests.csv"), &results)?;
    write_token_gaps(&output.join("token_gaps.csv"), &results)?;
    write_steps(&output.join("scheduler_steps.csv"), &steps)?;
    let summary = summary_from_results(&results);
    write_json(&output.join("summary.json"), &summary)?;
    println!("results written to {}", output.display());
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

async fn execute_requests(
    engine: &EngineHandle,
    requests: Vec<RequestSpec>,
    scheduler: SchedulerConfig,
    serial: bool,
) -> Result<Vec<RequestResult>, Box<dyn Error>> {
    let run_started = Instant::now();
    if serial {
        let mut results = Vec::with_capacity(requests.len());
        for request in requests {
            results.push(
                measure_request(engine.clone(), request, scheduler, run_started)
                    .await
                    .map_err(std::io::Error::other)?,
            );
        }
        return Ok(results);
    }
    let mut tasks = Vec::with_capacity(requests.len());
    for request in requests {
        tasks.push(tokio::spawn(measure_request(
            engine.clone(),
            request,
            scheduler,
            run_started,
        )));
    }
    let mut results = Vec::with_capacity(tasks.len());
    for task in tasks {
        results.push(task.await??);
    }
    results.sort_unstable_by_key(|result| result.request_id);
    Ok(results)
}

async fn measure_request(
    engine: EngineHandle,
    spec: RequestSpec,
    scheduler: SchedulerConfig,
    run_started: Instant,
) -> Result<RequestResult, String> {
    tokio::time::sleep_until(tokio::time::Instant::from_std(
        run_started + Duration::from_millis(spec.arrival_delay_ms),
    ))
    .await;
    let arrival = Instant::now();
    let arrival_ms = run_started.elapsed().as_millis();
    let mut stream = engine
        .try_generate(GenerationRequest {
            id: spec.id,
            prompt: spec.prompt,
            max_new_tokens: spec.max_new_tokens,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(42),
        })
        .map_err(|error| error.to_string())?;
    let mut queue_wait_ms = None;
    let mut ttft_ms = None;
    let mut previous_token = None;
    let mut token_gaps = Vec::new();
    let mut finish_reason = "stream_closed".to_owned();
    let mut finished = false;
    while let Some(event) = stream.recv().await {
        let now = Instant::now();
        match event {
            GenerationEvent::Started { .. } => {
                queue_wait_ms = Some(now.duration_since(arrival).as_millis());
            }
            GenerationEvent::Token { .. } => {
                let ready_at_ms = now.duration_since(run_started).as_millis();
                ttft_ms.get_or_insert(now.duration_since(arrival).as_millis());
                token_gaps.push(TokenGap {
                    request_id: spec.id.0,
                    token_index: token_gaps.len(),
                    ready_at_ms,
                    itl_ms: previous_token.map(|previous| now.duration_since(previous).as_millis()),
                });
                previous_token = Some(now);
            }
            GenerationEvent::Completed {
                finish_reason: reason,
                ..
            } => {
                finished = true;
                finish_reason = match reason {
                    FinishReason::Stop => "stop",
                    FinishReason::Length => "length",
                }
                .to_owned();
                break;
            }
            GenerationEvent::Cancelled { .. } => {
                finish_reason = "cancelled".to_owned();
                break;
            }
            GenerationEvent::TimedOut { .. } => {
                finish_reason = "timeout".to_owned();
                break;
            }
            GenerationEvent::Error { message, .. } => {
                finish_reason = format!("error:{message}");
                break;
            }
        }
    }
    Ok(RequestResult {
        request_id: spec.id.0,
        policy: scheduler_name(scheduler),
        prompt_tokens: spec.prompt_tokens,
        output_tokens: token_gaps.len(),
        arrival_ms,
        queue_wait_ms,
        ttft_ms,
        e2e_ms: arrival.elapsed().as_millis(),
        finished,
        finish_reason,
        token_gaps,
    })
}

fn workload_requests(
    workload: Workload,
    executor: &LlamaCppExecutor,
) -> Result<Vec<RequestSpec>, Box<dyn Error>> {
    let shapes: Vec<(usize, u32, u64)> = match workload {
        Workload::W0 => vec![(56, 64, 0), (504, 64, 0), (1_900, 64, 0)],
        Workload::W1 => vec![(32, 64, 0); 4],
        Workload::W2 => vec![(32, 96, 0), (32, 96, 0), (850, 64, 75)],
        Workload::W3 => (0..10)
            .map(|index| {
                let repetitions = if index == 9 {
                    850
                } else if index >= 6 {
                    200
                } else {
                    32
                };
                (
                    repetitions,
                    if index < 6 { 48 } else { 64 },
                    index as u64 * 20,
                )
            })
            .collect(),
        Workload::W5 => vec![
            (32, 160, 0),
            (32, 160, 0),
            (850, 64, 400),
            (850, 64, 700),
            (850, 64, 1_000),
            (32, 64, 4_000),
        ],
    };
    shapes
        .into_iter()
        .enumerate()
        .map(|(index, (repetitions, max_new_tokens, arrival_delay_ms))| {
            let prompt = format!(
                "Explain efficient local language model serving.{}",
                " scheduling".repeat(repetitions)
            );
            let prompt_tokens = executor.tokenize(&prompt)?.len();
            if prompt_tokens + max_new_tokens as usize > 2_048 {
                return Err(format!(
                    "workload prompt has {prompt_tokens} tokens plus {max_new_tokens} output tokens; per-sequence limit is 2048"
                )
                .into());
            }
            Ok(RequestSpec {
                id: RequestId(index as u64 + 1),
                prompt,
                prompt_tokens,
                max_new_tokens,
                arrival_delay_ms,
            })
        })
        .collect()
}

const fn scheduler_name(config: SchedulerConfig) -> &'static str {
    match config {
        SchedulerConfig::Fcfs => "fcfs",
        SchedulerConfig::DecodePriority => "decode-priority",
        SchedulerConfig::FixedChunk { .. } => "fixed-chunk",
        SchedulerConfig::Adaptive(_) => "adaptive",
    }
}

const fn chunk_budget(config: SchedulerConfig) -> Option<usize> {
    match config {
        SchedulerConfig::FixedChunk {
            prefill_chunk_tokens,
        } => Some(prefill_chunk_tokens),
        SchedulerConfig::Adaptive(config) => Some(config.initial_chunk_tokens),
        SchedulerConfig::Fcfs | SchedulerConfig::DecodePriority => None,
    }
}

const fn scheduler_metadata(config: SchedulerConfig) -> SchedulerMetadata {
    match config {
        SchedulerConfig::Adaptive(adaptive) => SchedulerMetadata {
            prefill_chunk_tokens: None,
            target_decode_gap_micros: Some(adaptive.target_decode_gap_micros),
            min_chunk_tokens: Some(adaptive.min_chunk_tokens),
            max_chunk_tokens: Some(adaptive.max_chunk_tokens),
            initial_chunk_tokens: Some(adaptive.initial_chunk_tokens),
            increase_step_tokens: Some(adaptive.increase_step_tokens),
            decrease_factor: Some(adaptive.decrease_factor),
            ewma_alpha: Some(adaptive.ewma_alpha),
            low_threshold_factor: Some(adaptive.low_threshold_factor),
            high_threshold_factor: Some(adaptive.high_threshold_factor),
            max_prefill_wait_micros: Some(adaptive.max_prefill_wait_micros),
        },
        baseline => SchedulerMetadata {
            prefill_chunk_tokens: chunk_budget(baseline),
            target_decode_gap_micros: None,
            min_chunk_tokens: None,
            max_chunk_tokens: None,
            initial_chunk_tokens: None,
            increase_step_tokens: None,
            decrease_factor: None,
            ewma_alpha: None,
            low_threshold_factor: None,
            high_threshold_factor: None,
            max_prefill_wait_micros: None,
        },
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    writeln!(file)?;
    Ok(())
}

fn write_requests(path: &Path, results: &[RequestResult]) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "request_id,policy,prompt_tokens,output_tokens,arrival_ms,queue_wait_ms,ttft_ms,e2e_ms,finished,finish_reason"
    )?;
    for result in results {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{}",
            result.request_id,
            result.policy,
            result.prompt_tokens,
            result.output_tokens,
            result.arrival_ms,
            optional(result.queue_wait_ms),
            optional(result.ttft_ms),
            result.e2e_ms,
            result.finished,
            result.finish_reason.replace(',', ";")
        )?;
    }
    Ok(())
}

fn write_token_gaps(path: &Path, results: &[RequestResult]) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    writeln!(file, "request_id,token_index,ready_at_ms,itl_ms")?;
    for gap in results.iter().flat_map(|result| &result.token_gaps) {
        writeln!(
            file,
            "{},{},{},{}",
            gap.request_id,
            gap.token_index,
            gap.ready_at_ms,
            optional(gap.itl_ms)
        )?;
    }
    Ok(())
}

fn write_steps(path: &Path, steps: &[EngineStepObservation]) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "engine_step,timestamp_ms,policy,active_sequences,queued_requests,oldest_prefill_wait_us,prefill_tokens,decode_tokens,chunk_budget,next_chunk_budget,observed_decode_gap_us,decode_gap_ewma_us,adaptive_plan_reason,adaptive_decision,backend_step_ms,scheduler_step_us"
    )?;
    for step in steps {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.3},{}",
            step.engine_step,
            step.timestamp_ms,
            step.policy,
            step.active_sequences,
            step.queued_requests,
            optional(step.oldest_prefill_wait_micros),
            step.prefill_tokens,
            step.decode_tokens,
            optional(step.chunk_budget.map(|value| value as u128)),
            optional(step.next_chunk_budget.map(|value| value as u128)),
            optional(step.observed_decode_gap_micros),
            optional(step.decode_gap_ewma_micros),
            step.adaptive_plan_reason.unwrap_or(""),
            step.adaptive_decision.unwrap_or(""),
            step.backend_step_micros as f64 / 1_000.0,
            step.scheduler_step_micros
        )?;
    }
    Ok(())
}

fn optional(value: Option<u128>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn summary_from_results(results: &[RequestResult]) -> Summary {
    let ttft = results
        .iter()
        .filter_map(|result| result.ttft_ms)
        .collect::<Vec<_>>();
    let e2e = results
        .iter()
        .map(|result| result.e2e_ms)
        .collect::<Vec<_>>();
    let queue_wait = results
        .iter()
        .filter_map(|result| result.queue_wait_ms)
        .collect::<Vec<_>>();
    let itl = results
        .iter()
        .flat_map(|result| &result.token_gaps)
        .filter_map(|gap| gap.itl_ms)
        .collect::<Vec<_>>();
    build_summary(
        results.len(),
        results.iter().filter(|result| result.finished).count(),
        results.iter().map(|result| result.output_tokens).sum(),
        results
            .iter()
            .map(|result| result.arrival_ms + result.e2e_ms)
            .max()
            .unwrap_or(0),
        SummarySamples {
            queue_wait,
            ttft,
            itl,
            e2e,
        },
    )
}

fn percentile(mut values: Vec<u128>, percentile: usize) -> Option<u128> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let index = (values.len() * percentile).div_ceil(100).saturating_sub(1);
    values.get(index).copied()
}

fn build_summary(
    request_count: usize,
    finished_count: usize,
    total_output_tokens: usize,
    elapsed_window_ms: u128,
    samples: SummarySamples,
) -> Summary {
    let elapsed_seconds =
        Duration::from_millis(u64::try_from(elapsed_window_ms).unwrap_or(u64::MAX)).as_secs_f64();
    Summary {
        request_count,
        finished_count,
        failed_count: request_count.saturating_sub(finished_count),
        total_output_tokens,
        elapsed_window_ms,
        queue_wait_p95_ms: percentile(samples.queue_wait.clone(), 95),
        queue_wait_max_ms: samples.queue_wait.into_iter().max(),
        ttft_p50_ms: percentile(samples.ttft.clone(), 50),
        ttft_p95_ms: percentile(samples.ttft, 95),
        itl_sample_count: samples.itl.len(),
        itl_p50_ms: percentile(samples.itl.clone(), 50),
        itl_p95_ms: percentile(samples.itl, 95),
        e2e_p50_ms: percentile(samples.e2e.clone(), 50),
        e2e_p95_ms: percentile(samples.e2e, 95),
        output_tokens_per_second: (elapsed_seconds > 0.0)
            .then_some(total_output_tokens as f64 / elapsed_seconds),
        requests_per_second: (elapsed_seconds > 0.0)
            .then_some(finished_count as f64 / elapsed_seconds),
    }
}

fn summarize(input: &Path) -> Result<(), Box<dyn Error>> {
    let file = File::open(input.join("requests.csv"))?;
    let mut ttft = Vec::new();
    let mut queue_wait = Vec::new();
    let mut e2e = Vec::new();
    let mut total_output_tokens = 0;
    let mut elapsed_window_ms = 0;
    let mut request_count = 0;
    let mut finished_count = 0;
    for line in BufReader::new(file).lines().skip(1) {
        let line = line?;
        let columns = line.split(',').collect::<Vec<_>>();
        if columns.len() != 10 {
            return Err(format!("invalid requests.csv row: {line}").into());
        }
        request_count += 1;
        if columns[8] == "true" {
            finished_count += 1;
        }
        total_output_tokens += columns[3].parse::<usize>()?;
        if let Ok(value) = columns[6].parse() {
            ttft.push(value);
        }
        if let Ok(value) = columns[5].parse() {
            queue_wait.push(value);
        }
        let arrival = columns[4].parse::<u128>()?;
        let request_e2e = columns[7].parse::<u128>()?;
        elapsed_window_ms = elapsed_window_ms.max(arrival + request_e2e);
        e2e.push(request_e2e);
    }
    let gaps = File::open(input.join("token_gaps.csv"))?;
    let mut itl = Vec::new();
    for line in BufReader::new(gaps).lines().skip(1) {
        let line = line?;
        let columns = line.split(',').collect::<Vec<_>>();
        if columns.len() != 4 {
            return Err(format!("invalid token_gaps.csv row: {line}").into());
        }
        if let Ok(value) = columns[3].parse() {
            itl.push(value);
        }
    }
    let summary = build_summary(
        request_count,
        finished_count,
        total_output_tokens,
        elapsed_window_ms,
        SummarySamples {
            queue_wait,
            ttft,
            itl,
            e2e,
        },
    );
    write_json(&input.join("summary.json"), &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn plot(input: &Path) -> Result<(), Box<dyn Error>> {
    let file = File::open(input.join("scheduler_steps.csv"))?;
    let mut samples = Vec::new();
    for line in BufReader::new(file).lines().skip(1) {
        let line = line?;
        let columns = line.split(',').collect::<Vec<_>>();
        if columns.len() != 16 {
            return Err(format!("invalid scheduler_steps.csv row: {line}").into());
        }
        samples.push((
            columns[1].parse::<u128>()?,
            columns[8].parse::<usize>().ok(),
            columns[10].parse::<u128>().ok(),
            columns[11].parse::<u128>().ok(),
            columns[5].parse::<u128>().ok(),
        ));
    }
    if samples.is_empty() {
        return Err("scheduler_steps.csv contains no samples".into());
    }

    let left = 70.0;
    let right = 930.0;
    let top = 55.0;
    let bottom = 450.0;
    let max_time = samples
        .iter()
        .map(|sample| sample.0)
        .max()
        .unwrap_or(1)
        .max(1) as f64;
    let max_chunk = samples
        .iter()
        .filter_map(|sample| sample.1)
        .max()
        .unwrap_or(1)
        .max(1) as f64;
    let max_gap = samples
        .iter()
        .flat_map(|sample| [sample.2, sample.3, sample.4])
        .flatten()
        .max()
        .unwrap_or(1)
        .max(1) as f64;
    let project = |time: u128, value: f64, maximum: f64| {
        let x = left + time as f64 / max_time * (right - left);
        let y = bottom - value / maximum * (bottom - top);
        format!("{x:.1},{y:.1}")
    };
    let chunk_points = samples
        .iter()
        .filter_map(|sample| {
            sample
                .1
                .map(|value| project(sample.0, value as f64, max_chunk))
        })
        .collect::<Vec<_>>()
        .join(" ");
    let gap_points = samples
        .iter()
        .filter_map(|sample| {
            sample
                .2
                .map(|value| project(sample.0, value as f64, max_gap))
        })
        .collect::<Vec<_>>()
        .join(" ");
    let ewma_points = samples
        .iter()
        .filter_map(|sample| {
            sample
                .3
                .map(|value| project(sample.0, value as f64, max_gap))
        })
        .collect::<Vec<_>>()
        .join(" ");
    let wait_points = samples
        .iter()
        .filter_map(|sample| {
            sample
                .4
                .map(|value| project(sample.0, value as f64, max_gap))
        })
        .collect::<Vec<_>>()
        .join(" ");

    let mut output = File::create(input.join("adaptive_trajectory.svg"))?;
    writeln!(
        output,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="520" viewBox="0 0 1000 520">
<rect width="1000" height="520" fill="#fbfaf7"/>
<text x="70" y="30" font-family="system-ui" font-size="20" fill="#1d2433">NiniServe scheduler trajectory</text>
<line x1="70" y1="450" x2="930" y2="450" stroke="#687080"/>
<line x1="70" y1="55" x2="70" y2="450" stroke="#687080"/>
<line x1="930" y1="55" x2="930" y2="450" stroke="#687080"/>
<text x="430" y="495" font-family="system-ui" font-size="14">elapsed time (0–{max_time:.0} ms)</text>
<text x="10" y="250" font-family="system-ui" font-size="14" transform="rotate(-90 10 250)">chunk tokens (0–{max_chunk:.0})</text>
<text x="985" y="300" font-family="system-ui" font-size="14" transform="rotate(90 985 300)">gap / wait (0–{max_gap:.0} µs)</text>
<polyline points="{chunk_points}" fill="none" stroke="#2463eb" stroke-width="2"/>
<polyline points="{gap_points}" fill="none" stroke="#e05a33" stroke-width="1.5" opacity="0.55"/>
<polyline points="{ewma_points}" fill="none" stroke="#bf2f64" stroke-width="2.5"/>
<polyline points="{wait_points}" fill="none" stroke="#15866b" stroke-width="1.5" stroke-dasharray="5 4"/>
<line x1="90" y1="470" x2="120" y2="470" stroke="#2463eb" stroke-width="2"/><text x="126" y="475" font-family="system-ui" font-size="13">selected chunk</text>
<line x1="260" y1="470" x2="290" y2="470" stroke="#e05a33" stroke-width="2"/><text x="296" y="475" font-family="system-ui" font-size="13">observed gap</text>
<line x1="430" y1="470" x2="460" y2="470" stroke="#bf2f64" stroke-width="2.5"/><text x="466" y="475" font-family="system-ui" font-size="13">EWMA gap</text>
<line x1="590" y1="470" x2="620" y2="470" stroke="#15866b" stroke-width="2" stroke-dasharray="5 4"/><text x="626" y="475" font-family="system-ui" font-size="13">oldest prefill wait</text>
</svg>"##
    )?;
    println!(
        "plot written to {}",
        input.join("adaptive_trajectory.svg").display()
    );
    Ok(())
}

fn command_output(program: &str, arguments: &[&str]) -> String {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map_or_else(
            || "unavailable".to_owned(),
            |output| String::from_utf8_lossy(&output.stdout).trim().to_owned(),
        )
}

#[cfg(test)]
mod tests {
    use super::{Cli, Workload, parse_cli, percentile};
    use niniserve_engine::{AdaptiveConfig, SchedulerConfig};

    #[test]
    fn parses_fixed_chunk_run() {
        let arguments = [
            "run",
            "--model",
            "model.gguf",
            "--scheduler",
            "fixed-chunk",
            "--prefill-chunk-tokens",
            "64",
            "--workload",
            "W2",
        ];
        let parsed = parse_cli(arguments.into_iter().map(ToOwned::to_owned)).expect("valid CLI");
        let Cli::Run(config) = parsed else {
            panic!("expected run")
        };
        assert_eq!(
            config.scheduler,
            SchedulerConfig::FixedChunk {
                prefill_chunk_tokens: 64
            }
        );
    }

    #[test]
    fn percentiles_are_measured_samples() {
        assert_eq!(percentile(vec![40, 10, 30, 20], 50), Some(20));
        assert_eq!(percentile(vec![40, 10, 30, 20], 95), Some(40));
        assert_eq!(percentile(Vec::new(), 95), None);
    }

    #[test]
    fn parses_repeated_adaptive_w5_run() {
        let arguments = [
            "run",
            "--model",
            "model.gguf",
            "--scheduler",
            "adaptive",
            "--target-itl-ms",
            "80",
            "--workload",
            "W5",
            "--repeats",
            "3",
        ];

        let parsed = parse_cli(arguments.into_iter().map(ToOwned::to_owned)).expect("valid CLI");
        let Cli::Run(config) = parsed else {
            panic!("expected run")
        };
        let expected = AdaptiveConfig {
            target_decode_gap_micros: 80_000,
            ..AdaptiveConfig::default()
        };
        assert_eq!(config.scheduler, SchedulerConfig::Adaptive(expected));
        assert!(matches!(config.workload, Workload::W5));
        assert_eq!(config.repeats, 3);
    }
}
