#![forbid(unsafe_code)]

use std::{
    env,
    error::Error,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::PathBuf,
};

use niniserve_backend::llamacpp::{LlamaCppConfig, LlamaCppExecutor};
use niniserve_engine::{EngineConfig, EngineHandle};
use niniserve_protocol::RequestLimits;
use niniserve_server::router;

#[derive(Debug)]
struct ServerConfig {
    model_path: PathBuf,
    port: u16,
}

impl ServerConfig {
    fn parse() -> Result<Self, String> {
        let mut arguments = env::args().skip(1);
        let mut model_path = None;
        let mut port = 8_080;
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
                _ => return Err(format!("unknown argument {argument:?}")),
            }
        }
        Ok(Self {
            model_path: model_path.ok_or("usage: niniserve --model MODEL.gguf [--port 8080]")?,
            port,
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
        "NiniServe listening on http://{address}; gpu_offload_supported={gpu_offload_supported}; n_ctx={} n_ctx_seq={} n_batch={} n_ubatch={} n_seq_max={}",
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
