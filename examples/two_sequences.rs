use std::{env, error::Error, path::Path};

use niniserve_backend::llamacpp::{ProbeConfig, run_two_sequence_probe};

fn main() -> Result<(), Box<dyn Error>> {
    let model_path = env::args()
        .nth(1)
        .ok_or("usage: two_sequences /absolute/or/relative/model.gguf")?;
    let report = run_two_sequence_probe(Path::new(&model_path), ProbeConfig::default())?;

    println!(
        "backend gpu_offload_supported={} limits n_ctx={} n_batch={} n_ubatch={}",
        report.gpu_offload_supported, report.n_ctx, report.n_batch, report.n_ubatch
    );
    for entry in &report.trace {
        println!(
            "trace batch={} phase={} seq={} pos={} token={} logits={}",
            entry.batch_id,
            entry.phase,
            entry.sequence_id,
            entry.position,
            entry.token_id,
            entry.requested_logits
        );
    }
    println!("sequence 0 token_ids={:?}", report.sequence_a_tokens);
    println!("sequence 0 text={:?}", report.sequence_a_text);
    println!("sequence 1 token_ids={:?}", report.sequence_b_tokens);
    println!("sequence 1 text={:?}", report.sequence_b_text);
    println!(
        "cleanup seq_0={} seq_1={}",
        report.sequence_a_cleaned, report.sequence_b_cleaned
    );
    Ok(())
}
