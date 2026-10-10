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
                _ => return Err(format!("unknown argument {argument:?}")),
            }
        }
        scheduler = match (scheduler, prefill_chunk_tokens) {
            (SchedulerConfig::FixedChunk { .. }, chunk) => SchedulerConfig::FixedChunk {
                prefill_chunk_tokens: chunk
                    .unwrap_or(niniserve_scheduler::DEFAULT_PREFILL_CHUNK_TOKENS),
            },
            (scheduler, None) => scheduler,
            (_, Some(_)) => {
                return Err("--prefill-chunk-tokens requires --scheduler fixed-chunk".to_owned());
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
        engine,
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
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use niniserve_engine::SchedulerConfig;

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
}
