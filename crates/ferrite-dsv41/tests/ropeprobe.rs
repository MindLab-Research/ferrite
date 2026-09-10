#[test]
fn print_rope_config() {
    let dir = std::env::var("DSV41_MODEL_DIR").unwrap_or_default();
    let txt = std::fs::read_to_string(format!("{dir}/config.json")).unwrap();
    let cfg = ferrite_dsv41::config::Dsv41Config::from_json_str(&txt).unwrap();
    println!("  original_seq_len = {}", cfg.original_seq_len);
    println!("  rope_theta       = {}", cfg.rope_theta);
    println!("  compress_rope_theta = {}", cfg.compress_rope_theta);
    println!("  rope_factor      = {}", cfg.rope_factor);
    println!("  beta_fast/beta_slow = {} / {}", cfg.beta_fast, cfg.beta_slow);
    println!("  rope_head_dim    = {}", cfg.rope_head_dim);
}
