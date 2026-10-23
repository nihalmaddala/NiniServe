#![forbid(unsafe_code)]

use std::{
    env,
    error::Error,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::PathBuf,
};

use niniserve_backend::llamacpp::{LlamaCppConfig, LlamaCppExecutor};
use niniserve_engine::{EngineConfig, EngineHandle, SchedulerConfig};
use niniserve_protocol::RequestLimits;
use niniserve_scheduler::AdaptiveConfig;
use niniserve_server::router;

#[derive(Debug)]
struct ServerConfig {
    model_path: PathBuf,
    port: u16,
    scheduler: SchedulerConfig,
}

impl ServerConfig {
    fn parse() -> Result<Self, String> {
        Self::parse_arguments(env::args().skip(1))
    }

    fn parse_arguments(mut arguments: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut model_path = None;
        let mut port = 8_080;
        let mut scheduler = SchedulerConfig::DecodePriority;
        let mut prefill_chunk_tokens = None;
        let mut adaptive = AdaptiveConfig::default();
        let mut adaptive_changed = false;
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--model" => {
                    model_path = Some(PathBuf::from(
                        arguments.next().ok_or("--model requires a path")?,
                    ));
                }
                "--port" => {
                    port = arguments
                        .next()
                        .ok_or("--port requires a number")?
                        .parse()
                        .map_err(|error| format!("invalid --port: {error}"))?;
                }
                "--scheduler" => {
                    scheduler = arguments
                        .next()
                        .ok_or("--scheduler requires a policy")?
                        .parse()
                        .map_err(|error| format!("invalid --scheduler: {error}"))?;
                }
                "--prefill-chunk-tokens" => {
                    let value = arguments
                        .next()
                        .ok_or("--prefill-chunk-tokens requires a number")?
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --prefill-chunk-tokens: {error}"))?;
                    if value == 0 {
                        return Err("--prefill-chunk-tokens must be positive".to_owned());
                    }
                    prefill_chunk_tokens = Some(value);
                }
                "--target-itl-ms" => {
                    adaptive.target_decode_gap_micros = parse_positive_u128(
                        arguments
                            .next()
                            .ok_or("--target-itl-ms requires a number")?,
                        "--target-itl-ms",
                    )?
                    .checked_mul(1_000)
                    .ok_or("--target-itl-ms is too large")?;
                    adaptive_changed = true;
                }
                "--adaptive-min-chunk-tokens" => {
                    adaptive.min_chunk_tokens = parse_positive_usize(
                        arguments
                            .next()
                            .ok_or("--adaptive-min-chunk-tokens requires a number")?,
                        "--adaptive-min-chunk-tokens",
                    )?;
                    adaptive_changed = true;
                }
                "--adaptive-max-chunk-tokens" => {
                    adaptive.max_chunk_tokens = parse_positive_usize(
                        arguments
                            .next()
                            .ok_or("--adaptive-max-chunk-tokens requires a number")?,
                        "--adaptive-max-chunk-tokens",
                    )?;
                    adaptive_changed = true;
                }
                "--adaptive-initial-chunk-tokens" => {
                    adaptive.initial_chunk_tokens = parse_positive_usize(
                        arguments
                            .next()
                            .ok_or("--adaptive-initial-chunk-tokens requires a number")?,
                        "--adaptive-initial-chunk-tokens",
                    )?;
                    adaptive_changed = true;
                }
                "--adaptive-increase-step-tokens" => {
                    adaptive.increase_step_tokens = parse_positive_usize(
                        arguments
                            .next()
                            .ok_or("--adaptive-increase-step-tokens requires a number")?,
                        "--adaptive-increase-step-tokens",
                    )?;
                    adaptive_changed = true;
                }
                "--adaptive-decrease-factor" => {
                    adaptive.decrease_factor = parse_f64(
                        arguments
                            .next()
                            .ok_or("--adaptive-decrease-factor requires a number")?,
                        "--adaptive-decrease-factor",
                    )?;
                    adaptive_changed = true;
                }
                "--adaptive-ewma-alpha" => {
                    adaptive.ewma_alpha = parse_f64(
                        arguments
                            .next()
                            .ok_or("--adaptive-ewma-alpha requires a number")?,
                        "--adaptive-ewma-alpha",
                    )?;
                    adaptive_changed = true;
                }
                "--adaptive-low-threshold-factor" => {
                    adaptive.low_threshold_factor = parse_f64(
                        arguments
                            .next()
                            .ok_or("--adaptive-low-threshold-factor requires a number")?,
                        "--adaptive-low-threshold-factor",
                    )?;
                    adaptive_changed = true;
                }
                "--adaptive-high-threshold-factor" => {
                    adaptive.high_threshold_factor = parse_f64(
                        arguments
                            .next()
                            .ok_or("--adaptive-high-threshold-factor requires a number")?,
                        "--adaptive-high-threshold-factor",
                    )?;
                    adaptive_changed = true;
                }
                "--adaptive-max-prefill-wait-ms" => {
                    adaptive.max_prefill_wait_micros = parse_positive_u128(
                        arguments
                            .next()
                            .ok_or("--adaptive-max-prefill-wait-ms requires a number")?,
                        "--adaptive-max-prefill-wait-ms",
                    )?
                    .checked_mul(1_000)
                    .ok_or("--adaptive-max-prefill-wait-ms is too large")?;
                    adaptive_changed = true;
                }
                _ => return Err(format!("unknown argument {argument:?}")),
            }
        }
        scheduler = match (scheduler, prefill_chunk_tokens, adaptive_changed) {
            (SchedulerConfig::FixedChunk { .. }, chunk, false) => SchedulerConfig::FixedChunk {
                prefill_chunk_tokens: chunk
                    .unwrap_or(niniserve_scheduler::DEFAULT_PREFILL_CHUNK_TOKENS),
            },
            (SchedulerConfig::Adaptive(_), None, _) => {
                SchedulerConfig::Adaptive(adaptive.validate().map_err(|error| error.to_string())?)
            }
            (scheduler, None, false) => scheduler,
            (_, Some(_), _) => {
                return Err("--prefill-chunk-tokens requires --scheduler fixed-chunk".to_owned());
            }
            (_, None, true) => {
                return Err("adaptive tuning flags require --scheduler adaptive".to_owned());
            }
        };
        Ok(Self {
            model_path: model_path
                .ok_or("usage: niniserve --model MODEL.gguf [--port 8080] [--scheduler POLICY]")?,
            port,
            scheduler,
        })
    }
}

fn parse_positive_usize(value: String, flag: &str) -> Result<usize, String> {
    let value = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {flag}: {error}"))?;
    if value == 0 {
        return Err(format!("{flag} must be positive"));
    }
    Ok(value)
}

fn parse_positive_u128(value: String, flag: &str) -> Result<u128, String> {
    let value = value
        .parse::<u128>()
        .map_err(|error| format!("invalid {flag}: {error}"))?;
    if value == 0 {
        return Err(format!("{flag} must be positive"));
    }
    Ok(value)
}

fn parse_f64(value: String, flag: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|error| format!("invalid {flag}: {error}"))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let config = ServerConfig::parse()?;
    let backend_config = LlamaCppConfig::two_requests(&config.model_path);
    let executor = LlamaCppExecutor::load(&backend_config)?;
    let gpu_offload_supported = executor.gpu_offload_supported();
    let context_limits = executor.context_limits();
    let engine = EngineHandle::spawn(
        executor,
        EngineConfig {
            command_capacity: 16,
            pending_capacity: 16,
            event_capacity: 32,
            request_timeout: std::time::Duration::from_secs(120),
            scheduler: config.scheduler,
        },
    );
    let app = router(
        engine.clone(),
        RequestLimits {
            max_prompt_bytes: 16 * 1_024,
            max_new_tokens: 256,
        },
    );
    let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), config.port);
    let listener = tokio::net::TcpListener::bind(address).await?;
    println!(
        "NiniServe listening on http://{address}; scheduler={:?}; gpu_offload_supported={gpu_offload_supported}; n_ctx={} n_ctx_seq={} n_batch={} n_ubatch={} n_seq_max={}",
        config.scheduler,
        context_limits.0,
        context_limits.0 / backend_config.n_seq_max,
        context_limits.1,
        context_limits.2,
        backend_config.n_seq_max
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(engine.clone()))
        .await?;
    engine.drain().await?;
    Ok(())
}

async fn shutdown_signal(engine: EngineHandle) {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("failed to install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = terminate.recv() => {},
        }
    }
    #[cfg(not(unix))]
    let _ = tokio::signal::ctrl_c().await;
    engine.begin_drain();
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use niniserve_engine::SchedulerConfig;
    use niniserve_scheduler::AdaptiveConfig;

    use super::ServerConfig;

    fn parse(arguments: &[&str]) -> Result<ServerConfig, String> {
        ServerConfig::parse_arguments(arguments.iter().map(ToString::to_string))
    }

    #[test]
    fn parses_fixed_chunk_scheduler_configuration() {
        let config = parse(&[
            "--model",
            "model.gguf",
            "--scheduler",
            "fixed-chunk",
            "--prefill-chunk-tokens",
            "64",
        ])
        .expect("valid arguments");

        assert_eq!(config.model_path, PathBuf::from("model.gguf"));
        assert_eq!(
            config.scheduler,
            SchedulerConfig::FixedChunk {
                prefill_chunk_tokens: 64
            }
        );
    }

    #[test]
    fn rejects_chunk_size_for_a_non_chunked_policy() {
        assert_eq!(
            parse(&[
                "--model",
                "model.gguf",
                "--scheduler",
                "fcfs",
                "--prefill-chunk-tokens",
                "64",
            ])
            .expect_err("invalid policy combination"),
            "--prefill-chunk-tokens requires --scheduler fixed-chunk"
        );
    }

    #[test]
    fn parses_and_validates_adaptive_scheduler_configuration() {
        let config = parse(&[
            "--model",
            "model.gguf",
            "--scheduler",
            "adaptive",
            "--target-itl-ms",
            "80",
            "--adaptive-min-chunk-tokens",
            "16",
            "--adaptive-max-chunk-tokens",
            "256",
            "--adaptive-initial-chunk-tokens",
            "64",
            "--adaptive-increase-step-tokens",
            "16",
            "--adaptive-decrease-factor",
            "0.4",
            "--adaptive-ewma-alpha",
            "0.3",
            "--adaptive-low-threshold-factor",
            "0.8",
            "--adaptive-high-threshold-factor",
            "1.2",
            "--adaptive-max-prefill-wait-ms",
            "400",
        ])
        .expect("valid adaptive arguments");

        let expected = AdaptiveConfig {
            target_decode_gap_micros: 80_000,
            min_chunk_tokens: 16,
            max_chunk_tokens: 256,
            initial_chunk_tokens: 64,
            increase_step_tokens: 16,
            decrease_factor: 0.4,
            ewma_alpha: 0.3,
            low_threshold_factor: 0.8,
            high_threshold_factor: 1.2,
            max_prefill_wait_micros: 400_000,
        };
        assert_eq!(config.scheduler, SchedulerConfig::Adaptive(expected));
    }
}
