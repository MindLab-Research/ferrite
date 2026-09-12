//! DSV41 spec-mode end-to-end verification — the `cargo test` replacement for
//! the manual `serve` → `curl` → grep loop every correctness fix used to pay
//! (≈3 minutes per iteration, three prompts, read the numbers by eye).
//!
//! [# What it checks]
//!
//! Two red lines, exactly the ones the hand loop checked:
//!
//! 1. **Text verbatim** — the spec commit path must reproduce the target's
//!    tokens. Three prompts (静夜思 / 1+1 / 出师表) are sent with `stream:false`
//!    and the decoded answers are asserted:
//!    * 静夜思 contains `床前明月光`
//!    * 出师表 contains `先帝创业未半`
//!    * 1+1 contains `2`
//!    * **no adjacent double character** in either poem (the historical
//!      `床床前` / `臣臣` / `光光` corruption — a verify block that reads the
//!      wrong causal row re-emits the token it just saw). [`has_double_char`] is
//!      the pure predicate; [`validate_report_texts`] is the rule set.
//! 2. **Accept statistics** — `mean-k > 1.0` (a real commit must beat the
//!    single-row path; `mean-k == 1.0` is the shadow/disabled fallthrough).
//!    Parsed from the serve log's `[dspark] steps=… mean-k=… tok/step=…` line.
//!
//! [# What it does NOT do]
//!
//! It does **not** launch `serve` (load + TP bring-up is far heavier than the
//! test itself). The caller starts `serve` once, tees its stderr to a log, and
//! this module *parses that log* + issues the HTTP requests. Both halves are
//! pure host code — no CUDA, no checkpoint load — which is why it is wired as
//! an `#[ignore]`d GPU-adjacent test: it only needs a *running* server.
//!
//! [# Environment]
//!
//! * `FERRITE_SPEC_PORT` — serve port (default `8199`).
//! * `FERRITE_SPEC_LOG`  — the serve log to parse (default `/tmp/spec_verify.log`).
//! * `FERRITE_SPEC_MODEL`— model name sent in the request (default
//!   `deepseek-v4.1-flash`).
//! * `FERRITE_MODEL_DIR` / `DSV41_MODEL_DIR` and `FERRITE_KERNELS` /
//!   `DSV41_KERNELS` — informational only (the `.so` is already loaded by the
//!   external serve); a missing path is a warning, never a failure.
//!
//! [# Usage]
//!
//! ```text
//! # 1. start serve ONCE (this is the slow part), tee its log:
//! DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_DSPARK_DEBUG=1 DSV41_TIMING=1 \
//!   ./target/release/ferrite-serve --model dsv41 --serve --tp 8 \
//!   --model-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash --port 8199 \
//!   > /tmp/spec_verify.log 2>&1 &
//!
//! # 2. verify (parse the log + 3 HTTP prompts):
//! FERRITE_SPEC_PORT=8199 FERRITE_SPEC_LOG=/tmp/spec_verify.log \
//!   cargo test -p ferrite-models --lib dspark_spec_verify -- --ignored --nocapture
//! ```
//!
//! `DSV41_TIMING=1` is required for the `[dspark] steps=…` line to exist at all
//! (it prints every 50 spec steps); `DSV41_DSPARK_DEBUG=1` is optional and only
//! adds the per-step trace.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::time::{Duration, Instant};

use ferrite_types::{FerriteError, Result};

/// Per-request wall-clock ceiling. A long 出师表 answer at ~20-40 ms/token is
/// tens of seconds; 240 s is the same bound the one-shot bash script used and it
/// must stay well under the ssh/curl layer's own patience.
const REQ_TIMEOUT: Duration = Duration::from_secs(240);

/// TCP connect timeout (the listener is already bound; this only guards a
/// half-dead host).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// The three verification prompts: `(label, prompt, max_tokens)`.
///
/// The labels are what [`validate_report_texts`] keys its rules on, so they must
/// stay in sync with it.
const PROMPTS: [(&str, &str, usize); 3] = [
    ("静夜思", "请背诵《静夜思》", 80),
    ("1+1", "1+1=?", 32),
    (
        "出师表",
        "请完整背诵《出师表》全文，从先帝创业未半而中道崩殂开始。",
        250,
    ),
];

/// The parsed outcome of one end-to-end spec run.
///
/// `texts` is `Vec<(label, decoded answer)>` — the same order as [`PROMPTS`].
/// The timing fields are the per-step averages from the serve log's last
/// `[dspark] steps=…` line; `faults` counts rank step errors / wedged-collective
/// lines seen anywhere in the log.
#[derive(Debug, Clone)]
pub struct SpecReport {
    pub texts: Vec<(String, String)>,
    pub mean_k: f32,
    pub tok_per_step: f32,
    pub draft_ms: f32,
    pub verify_ms: f32,
    pub commit_ms: f32,
    pub faults: usize,
}

impl std::fmt::Display for SpecReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "[spec e2e] mean-k={:.3} tok/step={:.3} draft={:.2}ms verify={:.2}ms \
             commit={:.2}ms faults={}",
            self.mean_k, self.tok_per_step, self.draft_ms, self.verify_ms, self.commit_ms, self.faults
        )?;
        for (label, text) in &self.texts {
            writeln!(
                f,
                "  [{label}] len={} {}",
                text.chars().count(),
                snippet(text, 240)
            )?;
        }
        Ok(())
    }
}

/// Run the end-to-end spec verification against an **already running** serve.
///
/// `model_dir` / `so` are informational (used only to warn when the paths do not
/// exist — the server owns the real ones); `port` is where to send the three
/// chat-completion requests. The serve log is read from `FERRITE_SPEC_LOG`
/// (default `/tmp/spec_verify.log`).
pub fn spec_serve_verify(model_dir: &str, so: &str, port: u16) -> Result<SpecReport> {
    preflight(model_dir, so);

    let model = std::env::var("FERRITE_SPEC_MODEL")
        .unwrap_or_else(|_| "deepseek-v4.1-flash".to_string());

    let mut texts: Vec<(String, String)> = Vec::with_capacity(PROMPTS.len());
    for (label, prompt, max_tokens) in PROMPTS {
        let body = chat_body(&model, prompt, max_tokens);
        let t0 = Instant::now();
        let raw = http_post(port, "/v1/chat/completions", &body)?;
        let content = extract_content(&raw).ok_or_else(|| {
            FerriteError::Config(format!(
                "[{label}] no `choices[].message.content` in response: {}",
                snippet(&raw, 200)
            ))
        })?;
        eprintln!(
            "[spec e2e] {label}: {} chars in {:.1}s",
            content.chars().count(),
            t0.elapsed().as_secs_f64()
        );
        texts.push((label.to_string(), content));
    }

    let log_path = spec_log_path();
    let log = std::fs::read_to_string(&log_path)
        .map_err(|e| FerriteError::Config(format!("read serve log {log_path}: {e}")))?;
    let stats = parse_dspark_stats(&log)
        .ok_or_else(|| {
            FerriteError::Config(format!(
                "no `[dspark] steps=… mean-k=…` line in {log_path} — was `DSV41_TIMING=1` set \
                 and the server given ≥50 spec steps?"
            ))
        })?;
    let faults = count_faults(&log);

    Ok(SpecReport {
        texts,
        mean_k: stats.mean_k,
        tok_per_step: stats.tok_per_step,
        draft_ms: stats.draft_ms,
        verify_ms: stats.verify_ms,
        commit_ms: stats.commit_ms,
        faults,
    })
}

/// The text red lines over a report's answers, as human-readable failures
/// (empty ⇒ all passed).
///
/// Rules are keyed on the [`PROMPTS`] labels: 静夜思 must contain `床前明月光`,
/// 出师表 must contain `先帝创业未半`, 1+1 must contain `2`; every answer must be
/// free of an adjacent double character.
pub fn validate_report_texts(texts: &[(String, String)]) -> Vec<String> {
    let mut fails = Vec::new();
    for (label, text) in texts {
        if let Some((idx, ctx)) = has_double_char(text) {
            fails.push(format!(
                "[{label}] adjacent double char at char {idx}: {ctx}"
            ));
        }
        let needle = if label.contains("静夜思") {
            Some("床前明月光")
        } else if label.contains("出师表") {
            Some("先帝创业未半")
        } else {
            None
        };
        if let Some(needle) = needle {
            if !text.contains(needle) {
                fails.push(format!(
                    "[{label}] missing \"{needle}\" — got: {}",
                    snippet(text, 160)
                ));
            }
        }
        if label.contains("1+1") && !text.contains('2') {
            fails.push(format!("[{label}] answer contains no '2': {}", snippet(text, 160)));
        }
    }
    fails
}

/// The first **adjacent duplicate** in `text`: `Some((char index, context))`.
///
/// The index is the position of the second (repeated) character; `context` is a
/// short window around it (`…ab`[X][X]`cd…`). Whitespace repeats (`\n\n` from a
/// markdown blank line) are NOT corruption and are skipped — the bug this
/// guards (`床床`/`光光`) is always a non-whitespace token re-emission.
pub fn has_double_char(text: &str) -> Option<(usize, String)> {
    let chars: Vec<char> = text.chars().collect();
    for i in 1..chars.len() {
        if chars[i] == chars[i - 1] && !chars[i].is_whitespace() {
            let lo = i.saturating_sub(6);
            let hi = (i + 6).min(chars.len());
            let ctx: String = chars[lo..hi].iter().collect();
            return Some((i, format!("…{ctx}…")));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Response / log parsing (std only).
// ---------------------------------------------------------------------------

/// The per-step averages parsed out of the serve log.
struct DsparkStats {
    mean_k: f32,
    tok_per_step: f32,
    draft_ms: f32,
    verify_ms: f32,
    commit_ms: f32,
}

/// The LAST `[dspark] steps=… mean-k=…` line wins: the line is printed every 50
/// steps with **cumulative** averages, so the final one is the whole run.
fn parse_dspark_stats(log: &str) -> Option<DsparkStats> {
    let line = log
        .lines()
        .filter(|l| l.contains("[dspark] steps=") && l.contains("mean-k="))
        .last()?;
    Some(DsparkStats {
        mean_k: field_f32(line, "mean-k=")?,
        tok_per_step: field_f32(line, "tok/step=")?,
        draft_ms: field_f32(line, "draft=")?,
        verify_ms: field_f32(line, "verify=")?,
        commit_ms: field_f32(line, "commit=")?,
    })
}

/// The `f32` following `key` up to the first non-numeric byte (so `draft=4.99ms`
/// yields `4.99`). `None` when the key is absent.
fn field_f32(line: &str, key: &str) -> Option<f32> {
    let start = line.find(key)? + key.len();
    let rest = &line[start..];
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e'))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Count the log's fault lines: a rank step error (`[dsv41] rank N … err …`),
/// the wedged-collective message, or a literal `fault` word.
fn count_faults(log: &str) -> usize {
    log.lines()
        .filter(|l| {
            (l.contains("rank") && l.contains(" err"))
                || l.contains("did not answer")
                || contains_word(l, "fault")
        })
        .count()
}

/// Whole-word `word` search (`fault` must not match `default`).
fn contains_word(hay: &str, word: &str) -> bool {
    let mut from = 0;
    while let Some(i) = hay[from..].find(word) {
        let p = from + i;
        let before_ok = p == 0 || !hay[..p].chars().next_back().unwrap().is_alphanumeric();
        let after = p + word.len();
        let after_ok = after >= hay.len() || !hay[after..].chars().next().unwrap().is_alphanumeric();
        if before_ok && after_ok {
            return true;
        }
        from = p + 1;
    }
    false
}

// ---------------------------------------------------------------------------
// Minimal std HTTP/1.1 client (no external crates).
// ---------------------------------------------------------------------------

fn chat_body(model: &str, prompt: &str, max_tokens: usize) -> String {
    format!(
        "{{\"model\":\"{}\",\"messages\":[{{\"role\":\"user\",\"content\":\"{}\"}}],\
         \"max_tokens\":{},\"stream\":false}}",
        json_escape(model),
        json_escape(prompt),
        max_tokens
    )
}

/// Escape a JSON string body (`"` `\` and control bytes); CJK passes through as
/// raw UTF-8 (the same form serde_json produces, and what `extract_content`
/// round-trips).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// POST `body` to `path` on localhost:`port`, returning the response body.
///
/// `Connection: close` + a bounded read loop: a server that keeps the socket
/// open (or a read timeout mid-body) yields what has arrived rather than losing
/// it, and the total read is capped at the response anyway.
fn http_post(port: u16, path: &str, body: &str) -> Result<String> {
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost:{port}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    );
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let mut s = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)
        .map_err(|e| FerriteError::Config(format!("connect 127.0.0.1:{port}: {e}")))?;
    // Best-effort: a platform that refuses the timeout still has the connect
    // bound above.
    let _ = s.set_read_timeout(Some(REQ_TIMEOUT));
    let _ = s.set_write_timeout(Some(REQ_TIMEOUT));
    s.write_all(req.as_bytes())
        .map_err(|e| FerriteError::Config(format!("write to 127.0.0.1:{port}: {e}")))?;
    let _ = s.flush();

    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 16 * 1024];
    loop {
        match s.read(&mut chunk) {
            Ok(0) => break, // clean close
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            // A read timeout after the body is complete is not an error: keep
            // what we have (see the `Connection: close` note above).
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                break
            }
            Err(e) => {
                return Err(FerriteError::Config(format!(
                    "read from 127.0.0.1:{port}: {e}"
                )))
            }
        }
    }
    parse_http_response(&buf)
}

/// Split status/headers/body, fail on a non-200, de-chunk when needed.
fn parse_http_response(buf: &[u8]) -> Result<String> {
    let split = find_subslice(buf, b"\r\n\r\n")
        .ok_or_else(|| FerriteError::Config("malformed HTTP response (no header terminator)".into()))?;
    let head = String::from_utf8_lossy(&buf[..split]).into_owned();
    let body = &buf[split + 4..];

    let status = head.lines().next().unwrap_or_default();
    if !status.contains(" 200") {
        return Err(FerriteError::Config(format!(
            "HTTP {status} — body: {}",
            snippet(&String::from_utf8_lossy(body), 300)
        )));
    }

    let chunked = head.to_ascii_lowercase().contains("transfer-encoding: chunked");
    let raw = if chunked { dechunk(body) } else { body.to_vec() };
    Ok(String::from_utf8_lossy(&raw).into_owned())
}

/// Minimal chunked-transfer decoder (axum sends a full JSON body, so this is a
/// safety net rather than the common path).
fn dechunk(body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut rest = body;
    loop {
        let line_end = match find_subslice(rest, b"\r\n") {
            Some(i) => i,
            None => break,
        };
        let size_str = String::from_utf8_lossy(&rest[..line_end]);
        let size = match usize::from_str_radix(size_str.trim().split(';').next().unwrap_or(""), 16)
        {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        let start = line_end + 2;
        let end = (start + size).min(rest.len());
        out.extend_from_slice(&rest[start..end]);
        // Skip the chunk's trailing CRLF.
        rest = if end + 2 <= rest.len() { &rest[end + 2..] } else { &[] };
    }
    out
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Pull `choices[0].message.content` out of the response JSON without a JSON
/// dependency: find the `"content"` key, then unescape the string that follows.
fn extract_content(json: &str) -> Option<String> {
    let key = "\"content\"";
    let k = json.find(key)?;
    let after = &json[k + key.len()..];
    let after = after[after.find(':')? + 1..].trim_start();
    let after = after.strip_prefix('"')?;

    let mut out = String::new();
    let mut it = after.chars();
    while let Some(c) = it.next() {
        match c {
            '"' => return Some(out),
            '\\' => match it.next()? {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                '/' => out.push('/'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'b' => out.push('\u{8}'),
                'f' => out.push('\u{c}'),
                'u' => {
                    let hex: String = (0..4).filter_map(|_| it.next()).collect();
                    if let Ok(cp) = u32::from_str_radix(&hex, 16) {
                        if let Some(ch) = char::from_u32(cp) {
                            out.push(ch);
                        }
                    }
                }
                other => out.push(other),
            },
            other => out.push(other),
        }
    }
    None
}

fn snippet(text: &str, max: usize) -> String {
    let mut s: String = text.chars().take(max).collect();
    if text.chars().count() > max {
        s.push('…');
    }
    s
}

fn spec_log_path() -> String {
    std::env::var("FERRITE_SPEC_LOG").unwrap_or_else(|_| "/tmp/spec_verify.log".to_string())
}

/// `model_dir` / `so` are informational here (the external serve owns the real
/// paths); a miss is a warning, not a failure.
fn preflight(model_dir: &str, so: &str) {
    if !model_dir.is_empty() && !Path::new(model_dir).is_dir() {
        eprintln!("[spec e2e] note: model_dir `{model_dir}` not found (serve is external — informational)");
    }
    if !so.is_empty() && !Path::new(so).is_file() {
        eprintln!("[spec e2e] note: kernels `{so}` not found (already loaded by serve — informational)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pure text red line — no GPU, no serve, always run.
    #[test]
    fn double_char_predicate() {
        assert_eq!(has_double_char("床前明月光"), None);
        let (i, ctx) = has_double_char("床床前明月光").expect("double");
        assert_eq!(i, 1);
        assert!(ctx.contains("床床"), "context {ctx}");
        // Whitespace repeats (markdown blank line) are not corruption.
        assert_eq!(has_double_char("a\n\nb"), None);
        assert_eq!(has_double_char("臣臣"), Some((1, "…臣臣…".to_string())));
    }

    /// The rule set over a synthetic (passing and failing) report.
    #[test]
    fn text_rules() {
        let good = vec![
            ("静夜思".to_string(), "床前明月光，疑是地上霜。".to_string()),
            ("1+1".to_string(), "1+1=2".to_string()),
            ("出师表".to_string(), "先帝创业未半而中道崩殂。".to_string()),
        ];
        assert!(validate_report_texts(&good).is_empty());

        let bad = vec![
            ("静夜思".to_string(), "床床前明月光".to_string()),
            ("1+1".to_string(), "1+1=3".to_string()),
            ("出师表".to_string(), "全文全文".to_string()),
        ];
        let fails = validate_report_texts(&bad);
        assert_eq!(fails.len(), 3, "{fails:?}");
    }

    /// Log parsing: last cumulative line wins, faults are counted.
    #[test]
    fn log_parsing() {
        let log = "\
[dsv41] rank0: chain ready, serving\n\
[dsv41] rank 3 spec step err at pos 12: boom\n\
[dspark] steps=50 mean-k=0.100 tok/step=1.100 draft=9.00ms verify=40.00ms commit=0.50ms (per step)\n\
[dspark] steps=100 mean-k=0.520 tok/step=1.520 draft=4.99ms verify=38.85ms commit=0.19ms (per step)\n";
        let st = parse_dspark_stats(log).expect("stats");
        assert_eq!(st.mean_k, 0.520);
        assert_eq!(st.tok_per_step, 1.520);
        assert_eq!(st.draft_ms, 4.99);
        assert_eq!(st.commit_ms, 0.19);
        assert_eq!(count_faults(log), 1);
        // `default` must not be mistaken for the `fault` word.
        assert_eq!(count_faults("using the default path"), 0);
    }

    /// Response decoding (status + content + chunked safety net).
    #[test]
    fn http_response_decoding() {
        let body = "{\"id\":\"x\",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\
                    \"message\":{\"role\":\"assistant\",\"content\":\"床前明月光，\\n疑是地上霜。\"},\
                    \"finish_reason\":\"stop\"}]}";
        let resp = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{body}");
        assert_eq!(
            extract_content(&parse_http_response(resp.as_bytes()).unwrap()).unwrap(),
            "床前明月光，\n疑是地上霜。"
        );

        let chunked = format!(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{}\r\n0\r\n\r\n",
            body.len(),
            body
        );
        assert_eq!(
            extract_content(&parse_http_response(chunked.as_bytes()).unwrap()).unwrap(),
            "床前明月光，\n疑是地上霜。"
        );

        let err = "HTTP/1.1 500 Internal Server Error\r\n\r\nboom";
        assert!(parse_http_response(err.as_bytes()).is_err());
    }

    /// End-to-end against a serve the operator started (see the module docs).
    #[test]
    #[ignore = "needs a running `serve` (DSV41_SPEC=1 DSV41_TIMING=1) + a GPU; run with --ignored"]
    fn dspark_spec_verify_gpu() {
        let (dir, so) = spec_env();
        let port: u16 = std::env::var("FERRITE_SPEC_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8199);
        let rep = spec_serve_verify(&dir, &so, port).expect("spec_serve_verify");
        // The full report is what `--nocapture` is for; the asserts are the gate.
        println!("{rep}");

        let fails = validate_report_texts(&rep.texts);
        assert!(
            fails.is_empty(),
            "spec text red line FAILED:\n  {}",
            fails.join("\n  ")
        );
        assert!(
            rep.mean_k > 1.0,
            "accept below the red line: mean-k={:.3} tok/step={:.3} (is DSV41_SPEC=1 on?)",
            rep.mean_k,
            rep.tok_per_step
        );
    }

    /// `(model_dir, kernel .so)` — informational for this test; the same
    /// `FERRITE_*` → `DSV41_*` fallback order `dspark_parity` uses.
    fn spec_env() -> (String, String) {
        let dir = std::env::var("FERRITE_MODEL_DIR")
            .or_else(|_| std::env::var("DSV41_MODEL_DIR"))
            .unwrap_or_else(|_| "/opt/dlami/nvme/models/DeepSeek-V4.1-Flash".to_string());
        let so = std::env::var("FERRITE_KERNELS")
            .or_else(|_| std::env::var("DSV41_KERNELS"))
            .unwrap_or_else(|_| "kernels/cuda/libferrite_kernels.so".to_string());
        (dir, so)
    }
}
