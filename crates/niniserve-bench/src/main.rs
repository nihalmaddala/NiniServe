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
use niniserve_engine::{EngineConfig, EngineHandle, EngineStepObservation, SchedulerConfig};
use niniserve_protocol::{FinishReason, GenerationEvent, GenerationRequest, RequestId};
use serde::Serialize;

const BACKEND_VERSION: &str = "llama-cpp-2 0.1.159";

#[derive(Debug)]
enum Cli {
    Run(RunConfig),
    Summary(PathBuf),
}

#[derive(Debug)]
struct RunConfig {
    model: PathBuf,
    scheduler: SchedulerConfig,
    workload: Workload,
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy)]
enum Workload {
    W0,
    W1,
    W2,
    W3,
}

impl Workload {
    fn parse(value: &str) -> Result<Self, String> {
        match value.to_ascii_lowercase().as_str() {
            "w0" => Ok(Self::W0),
            "w1" => Ok(Self::W1),
            "w2" => Ok(Self::W2),
            "w3" => Ok(Self::W3),
            _ => Err(format!(
                "unknown workload {value:?}; expected W0, W1, W2, or W3"
            )),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::W0 => "W0_single_request",
            Self::W1 => "W1_concurrent_short",
            Self::W2 => "W2_long_prompt_interference",
            Self::W3 => "W3_deterministic_mixed",
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
    model_load_ms: u128,
    started_at_unix_seconds: u64,
}

#[derive(Debug, Serialize)]
struct SchedulerMetadata {
    prefill_chunk_tokens: Option<usize>,
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
    ttft_p50_ms: Option<u128>,
    ttft_p95_ms: Option<u128>,
    e2e_p50_ms: Option<u128>,
    e2e_p95_ms: Option<u128>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    match parse_cli(env::args().skip(1))? {
        Cli::Run(config) => run(config).await,
        Cli::Summary(input) => summarize(&input),
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
                    "--workload" => {
                        workload = Some(Workload::parse(&next(&mut arguments, "--workload")?)?);
                    }
                    "--output" => output = Some(PathBuf::from(next(&mut arguments, "--output")?)),
                    _ => return Err(format!("unknown argument {argument:?}")),
                }
            }
            scheduler = match (scheduler, chunk) {
                (SchedulerConfig::FixedChunk { prefill_chunk_tokens }, None) => {
                    SchedulerConfig::FixedChunk { prefill_chunk_tokens }
                }
                (SchedulerConfig::FixedChunk { .. }, Some(prefill_chunk_tokens)) => {
                    SchedulerConfig::FixedChunk { prefill_chunk_tokens }
                }
                (scheduler, None) => scheduler,
                (_, Some(_)) => {
                    return Err(
                        "--prefill-chunk-tokens requires --scheduler fixed-chunk".to_owned(),
                    );
                }
            };
            Ok(Cli::Run(RunConfig {
                model: model.ok_or("run requires --model MODEL.gguf")?,
                scheduler,
                workload: workload.ok_or("run requires --workload W0|W1|W2|W3")?,
                output,
            }))
        }
        Some("summary") => Ok(Cli::Summary(PathBuf::from(next(
            &mut arguments,
            "summary",
        )?))),
        _ => Err("usage: niniserve-bench run --model MODEL.gguf --scheduler POLICY --workload W0|W1|W2|W3 [--output DIR] | summary RUN_DIR".to_owned()),
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
    let measured_requests = requests.len();
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
    execute_requests(&engine, warmup, config.scheduler).await?;
    let _ = engine.take_step_observations();
    let results = execute_requests(&engine, requests, config.scheduler).await?;
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
        scheduler_config: SchedulerMetadata {
            prefill_chunk_tokens: chunk_budget(config.scheduler),
        },
        workload: config.workload.name(),
        warmup_requests: measured_requests,
        measured_requests,
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
) -> Result<Vec<RequestResult>, Box<dyn Error>> {
    let run_started = Instant::now();
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
        Workload::W0 => vec![(256, 64, 0)],
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
    }
}

const fn chunk_budget(config: SchedulerConfig) -> Option<usize> {
    match config {
        SchedulerConfig::FixedChunk {
            prefill_chunk_tokens,
        } => Some(prefill_chunk_tokens),
        SchedulerConfig::Fcfs | SchedulerConfig::DecodePriority => None,
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
        "engine_step,timestamp_ms,policy,active_sequences,queued_requests,prefill_tokens,decode_tokens,chunk_budget,backend_step_ms,scheduler_step_us"
    )?;
    for step in steps {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{:.3},{}",
            step.engine_step,
            step.timestamp_ms,
            step.policy,
            step.active_sequences,
            step.queued_requests,
            step.prefill_tokens,
            step.decode_tokens,
            optional(step.chunk_budget.map(|value| value as u128)),
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
    Summary {
        request_count: results.len(),
        finished_count: results.iter().filter(|result| result.finished).count(),
        ttft_p50_ms: percentile(ttft.clone(), 50),
        ttft_p95_ms: percentile(ttft, 95),
        e2e_p50_ms: percentile(e2e.clone(), 50),
        e2e_p95_ms: percentile(e2e, 95),
    }
}

fn percentile(mut values: Vec<u128>, percentile: usize) -> Option<u128> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let index = (values.len() - 1) * percentile / 100;
    values.get(index).copied()
}

fn summarize(input: &Path) -> Result<(), Box<dyn Error>> {
    let file = File::open(input.join("requests.csv"))?;
    let mut ttft = Vec::new();
    let mut e2e = Vec::new();
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
        if let Ok(value) = columns[6].parse() {
            ttft.push(value);
        }
        e2e.push(columns[7].parse()?);
    }
    let summary = Summary {
        request_count,
        finished_count,
        ttft_p50_ms: percentile(ttft.clone(), 50),
        ttft_p95_ms: percentile(ttft, 95),
        e2e_p50_ms: percentile(e2e.clone(), 50),
        e2e_p95_ms: percentile(e2e, 95),
    };
    write_json(&input.join("summary.json"), &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
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
    use super::{Cli, parse_cli, percentile};
    use niniserve_engine::SchedulerConfig;

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
        assert_eq!(percentile(vec![40, 10, 30, 20], 95), Some(30));
        assert_eq!(percentile(Vec::new(), 95), None);
    }
}
