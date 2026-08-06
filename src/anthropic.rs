//! The Anthropic backend: the Messages API over raw HTTP (there is no
//! official Rust SDK), streamed as SSE and accumulated back into a complete
//! message.
//!
//! Content blocks are kept as raw `serde_json::Value` end to end so thinking
//! blocks (signatures included) and server compaction blocks round-trip
//! unchanged into the next request. This is the harness's native message
//! shape — the journal is written in it — so this backend does no
//! translation, only transport.

use crate::api::{Backend, Client, Failure, FinalMessage, Tier, Turn, Usage};
use crate::ui::{self, Tag};
use anyhow::{Context, Result, anyhow, bail};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::collections::HashMap;

const API_VERSION: &str = "2023-06-01";
const MAX_TOKENS: u64 = 64_000;

const BETA_FALLBACK: &str = "server-side-fallback-2026-07-01";
const BETA_COMPACT: &str = "compact-2026-01-12";
const BETA_OAUTH: &str = "oauth-2025-04-20";

enum Auth {
    ApiKey(String),
    Bearer(String),
}

/// Which optional request features a model actually accepts. Sending one a
/// model rejects fails the whole turn, so the body is built per model rather
/// than assuming the newest surface everywhere.
#[derive(Clone, Copy)]
pub(crate) struct Caps {
    adaptive: bool,
    effort: bool,
    fallbacks: bool,
    pub(crate) compaction: bool,
}

impl Caps {
    /// A starting guess from the model id. Unknown models get the modern
    /// surface and are corrected by the first rejection rather than being
    /// permanently downgraded on a name we failed to recognize.
    fn guess(model: &str) -> Caps {
        let modern = Caps {
            adaptive: true,
            effort: true,
            fallbacks: false,
            compaction: true,
        };
        match model {
            m if m.starts_with("claude-fable-") || m.starts_with("claude-mythos-") => Caps {
                fallbacks: true,
                ..modern
            },
            m if m.starts_with("claude-opus-5") => Caps {
                fallbacks: true,
                ..modern
            },
            m if m.starts_with("claude-haiku-") => Caps {
                adaptive: false,
                effort: false,
                fallbacks: false,
                compaction: false,
            },
            m if m.starts_with("claude-opus-4-5") || m.starts_with("claude-sonnet-4-5") => Caps {
                adaptive: false,
                ..modern
            },
            _ => modern,
        }
    }
}

pub struct Anthropic {
    auth: Auth,
    base_url: String,
    /// Per-model feature flags, seeded by `Caps::guess` and narrowed whenever
    /// the API says a parameter is unsupported. One rejected turn teaches the
    /// backend permanently instead of failing every turn after it.
    caps: std::sync::Mutex<HashMap<String, Caps>>,
}

impl Anthropic {
    pub fn from_env() -> Result<Anthropic> {
        let auth = if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            Auth::ApiKey(key)
        } else if let Ok(token) = std::env::var("ANTHROPIC_AUTH_TOKEN") {
            Auth::Bearer(token)
        } else {
            bail!(
                "No Anthropic credentials found. Either:\n  \
                 export ANTHROPIC_API_KEY=sk-ant-...\n  \
                 or: ant auth login && eval \"$(ant auth print-credentials --env)\""
            );
        };
        Ok(Anthropic {
            auth,
            base_url: std::env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com".into()),
            caps: std::sync::Mutex::new(HashMap::new()),
        })
    }

    /// Callers pass whatever they have — a tier alias ("small") or a concrete
    /// id — but `Caps::guess` only recognizes concrete ids, so every lookup
    /// resolves through the tier first. Keying the cache on the alias would
    /// make every tier fall through to `Caps::guess`'s modern default and
    /// relearn the same rejections on the model's first turn, every run.
    pub(crate) fn caps(&self, model: &str) -> Caps {
        let model = Tier::parse(model).map(Tier::anthropic).unwrap_or(model);
        *self
            .caps
            .lock()
            .unwrap()
            .entry(model.to_string())
            .or_insert_with(|| Caps::guess(model))
    }

    /// Read a rejection and turn off whatever the server named. Returns true
    /// when something changed, meaning a retry can succeed.
    fn learn_from(&self, model: &str, text: &str, tag: &Tag) -> bool {
        let model = Tier::parse(model).map(Tier::anthropic).unwrap_or(model);
        let mut caps = self.caps.lock().unwrap();
        let entry = caps
            .entry(model.to_string())
            .or_insert_with(|| Caps::guess(model));
        let before = (
            entry.adaptive,
            entry.effort,
            entry.fallbacks,
            entry.compaction,
        );
        if text.contains("thinking") {
            entry.adaptive = false;
        }
        if text.contains("fallbacks") {
            entry.fallbacks = false;
        }
        if text.contains("effort") || text.contains("output_config") {
            entry.effort = false;
        }
        if text.contains("anthropic-beta") {
            entry.fallbacks = false;
            entry.compaction = false;
        }
        if text.contains("context_management")
            || text.contains("compact_20260112")
            || text.contains(BETA_COMPACT)
        {
            entry.compaction = false;
        }
        let changed = before
            != (
                entry.adaptive,
                entry.effort,
                entry.fallbacks,
                entry.compaction,
            );
        if changed {
            ui::warn(
                tag,
                &format!("{model} rejected a request parameter ({text}); retrying without it"),
            );
        }
        changed
    }

    fn build_body(&self, client: &Client, turn: &Turn<'_>) -> Value {
        let caps = self.caps(turn.model);
        let tier = Tier::parse(turn.model).unwrap_or(Tier::Large);
        let mut body = json!({
            "model": tier.anthropic(),
            "max_tokens": MAX_TOKENS,
            "stream": true,
            // Second breakpoint, auto-placed on the last cacheable block, so
            // this process's growing conversation gets incremental hits turn
            // to turn. The first breakpoint is inside `system` and covers the
            // tools + shared-preamble prefix that every process has in common.
            "cache_control": {"type": "ephemeral"},
            "system": turn.system,
            "tools": turn.tools,
            "messages": turn.messages,
        });
        if caps.adaptive {
            body["thinking"] = json!({"type": "adaptive"});
        }
        if caps.fallbacks {
            // If safety classifiers decline a request, re-serve it on the
            // recommended fallback model inside the same call.
            body["fallbacks"] = json!("default");
        }
        if let Some(effort) = turn.effort.filter(|_| caps.effort) {
            body["output_config"] = json!({"effort": effort});
        }
        if client.compaction_enabled() && caps.compaction {
            // Server-side compaction. The API watches the prompt size and,
            // as it approaches the trigger threshold, summarizes the earlier
            // part of the conversation into a `compaction` block that replaces
            // it on subsequent requests. We echo the block back verbatim as
            // part of the assistant turn, which is what keeps the state alive.
            body["context_management"] = json!({"edits": [{"type": "compact_20260112"}]});
        }
        body
    }

    /// `None` when this model needs no betas at all. An empty `anthropic-beta`
    /// header is not the same as no header — the API rejects it — and a model
    /// that supports neither fallbacks nor compaction needs none.
    fn beta_header(&self, client: &Client, model: &str) -> Option<String> {
        let caps = self.caps(model);
        let mut betas = Vec::new();
        if caps.fallbacks {
            betas.push(BETA_FALLBACK);
        }
        if client.compaction_enabled() && caps.compaction {
            betas.push(BETA_COMPACT);
        }
        if matches!(self.auth, Auth::Bearer(_)) {
            betas.push(BETA_OAUTH);
        }
        (!betas.is_empty()).then(|| betas.join(","))
    }
}

impl Backend for Anthropic {
    async fn attempt(
        &self,
        client: &Client,
        turn: &Turn<'_>,
        tag: &Tag,
    ) -> Result<FinalMessage, Failure> {
        // Rebuilt per attempt so it reflects capabilities latched off by a
        // rejection in between.
        let body = self.build_body(client, turn);
        let mut req = client
            .http()
            .post(format!("{}/v1/messages", self.base_url))
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json");
        if let Some(betas) = self.beta_header(client, turn.model) {
            req = req.header("anthropic-beta", betas);
        }
        req = match &self.auth {
            Auth::ApiKey(key) => req.header("x-api-key", key),
            Auth::Bearer(token) => req.bearer_auth(token),
        };

        let resp = req
            .json(&body)
            .send()
            .await
            .context("request failed")
            .map_err(Failure::plain)?;
        let status = resp.status();
        if !status.is_success() {
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .map(String::from);
            let text = resp.text().await.unwrap_or_default();
            let msg = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| v["error"]["message"].as_str().map(String::from))
                .unwrap_or(text);
            return Err(Failure {
                error: anyhow!("HTTP {status}: {msg}"),
                retry_after,
            });
        }
        consume_stream(resp, tag).await.map_err(Failure::plain)
    }

    async fn recover(
        &self,
        _client: &Client,
        failure: &Failure,
        turn: &Turn<'_>,
        tag: &Tag,
    ) -> bool {
        let text = failure.error.to_string();
        text.contains("HTTP 400") && self.learn_from(turn.model, &text, tag)
    }

    fn context_window(&self, tier: Tier) -> u64 {
        match tier {
            Tier::Small => 200_000,
            Tier::Medium | Tier::Large => 1_000_000,
        }
    }
}

async fn consume_stream(resp: reqwest::Response, tag: &Tag) -> Result<FinalMessage> {
    let mut content: Vec<Value> = Vec::new();
    let mut tool_json: HashMap<usize, String> = HashMap::new();
    let mut stop_reason = String::new();
    // The four-way split, straight from the server: prompt counters at
    // message_start, output at message_delta (overwritten, not summed — the
    // server reports it cumulatively). Handed back on the message and folded
    // into the run's counters once by the driver, so a request can never
    // double-count.
    let mut usage = Usage::default();
    // Buffer streamed text so we only print whole lines.
    let mut line_buf = String::new();

    let mut buf = String::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("stream interrupted")?;
        buf.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(pos) = buf.find("\n\n") {
            let frame: String = buf.drain(..pos + 2).collect();
            for line in frame.lines() {
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let event: Value = serde_json::from_str(data.trim()).context("bad SSE payload")?;
                apply_event(
                    &event,
                    &mut content,
                    &mut tool_json,
                    &mut stop_reason,
                    &mut usage,
                    &mut line_buf,
                    tag,
                )?;
            }
        }
    }
    if !line_buf.is_empty() {
        ui::say(tag, &line_buf);
    }

    Ok(FinalMessage {
        // Anthropic has no server-side thread; the conversation is resent.
        thread: None,
        content,
        stop_reason,
        input_tokens: usage.prompt(),
        billable: usage.billable(),
        usage,
    })
}

#[allow(clippy::too_many_arguments)]
fn apply_event(
    event: &Value,
    content: &mut Vec<Value>,
    tool_json: &mut HashMap<usize, String>,
    stop_reason: &mut String,
    usage: &mut Usage,
    line_buf: &mut String,
    tag: &Tag,
) -> Result<()> {
    match event["type"].as_str().unwrap_or("") {
        "message_start" => {
            let reported = &event["message"]["usage"];
            // The prompt arrives as three disjoint counters, each billed at a
            // different rate: `input_tokens` is the uncached remainder, not the
            // total. They are kept apart here and summed only where a total is
            // what's wanted, because collapsing them is how a cost figure
            // becomes fiction.
            usage.uncached_input = reported["input_tokens"].as_u64().unwrap_or(0);
            usage.cache_write = reported["cache_creation_input_tokens"]
                .as_u64()
                .unwrap_or(0);
            usage.cache_read = reported["cache_read_input_tokens"].as_u64().unwrap_or(0);
        }
        "content_block_start" => {
            let index = event["index"].as_u64().unwrap_or(0) as usize;
            let block = event["content_block"].clone();
            match block["type"].as_str().unwrap_or("") {
                "tool_use" => {
                    tool_json.insert(index, String::new());
                }
                "compaction" => ui::trace(tag, "⟳ server compacted earlier context"),
                _ => {}
            }
            while content.len() <= index {
                content.push(Value::Null);
            }
            content[index] = block;
        }
        "content_block_delta" => {
            let index = event["index"].as_u64().unwrap_or(0) as usize;
            let delta = &event["delta"];
            match delta["type"].as_str().unwrap_or("") {
                "text_delta" => {
                    let text = delta["text"].as_str().unwrap_or("");
                    append_str(&mut content[index], "text", text);
                    line_buf.push_str(text);
                    while let Some(nl) = line_buf.find('\n') {
                        let line: String = line_buf.drain(..=nl).collect();
                        ui::say(tag, line.trim_end_matches('\n'));
                    }
                }
                "input_json_delta" => {
                    if let Some(partial) = tool_json.get_mut(&index) {
                        partial.push_str(delta["partial_json"].as_str().unwrap_or(""));
                    }
                }
                // Every other delta follows the same shape: each string
                // field names the block field it extends (thinking_delta →
                // thinking, signature_delta → signature, and whatever a
                // compaction summary streams as). Handling it structurally
                // rather than by name means an unrecognized block type is
                // still accumulated correctly and can be echoed back intact.
                _ => {
                    if let Some(fields) = delta.as_object() {
                        for (key, value) in fields {
                            if key == "type" {
                                continue;
                            }
                            if let Some(text) = value.as_str() {
                                append_str(&mut content[index], key, text);
                            }
                        }
                    }
                }
            }
        }
        "content_block_stop" => {
            let index = event["index"].as_u64().unwrap_or(0) as usize;
            if let Some(partial) = tool_json.remove(&index) {
                let input: Value = if partial.trim().is_empty() {
                    json!({})
                } else {
                    serde_json::from_str(&partial).context("bad tool input JSON")?
                };
                content[index]["input"] = input;
            }
        }
        "message_delta" => {
            if let Some(reason) = event["delta"]["stop_reason"].as_str() {
                *stop_reason = reason.to_string();
            }
            // Cumulative, so overwrite rather than add.
            if let Some(out) = event["usage"]["output_tokens"].as_u64() {
                usage.output = out;
            }
        }
        "error" => {
            let kind = event["error"]["type"].as_str().unwrap_or("unknown");
            let msg = event["error"]["message"].as_str().unwrap_or("");
            return Err(anyhow!("stream error ({kind}): {msg}"));
        }
        // message_stop / ping need no handling.
        _ => {}
    }
    Ok(())
}

fn append_str(block: &mut Value, key: &str, add: &str) {
    if add.is_empty() {
        return;
    }
    let mut current = block[key].as_str().unwrap_or("").to_string();
    current.push_str(add);
    block[key] = Value::String(current);
}
