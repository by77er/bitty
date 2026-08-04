//! Minimal Anthropic Messages API client over raw HTTP (there is no official
//! Rust SDK). Always streams (SSE) so long agentic turns can't hit HTTP
//! timeouts, and accumulates the events back into a complete message.
//!
//! Content blocks are kept as raw `serde_json::Value` end to end so thinking
//! blocks (signatures included) and server compaction blocks round-trip
//! unchanged into the next request.

use crate::ui::{self, Tag};
use anyhow::{Context, Result, anyhow, bail};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

const API_VERSION: &str = "2023-06-01";

/// How much model a process gets, independent of who is serving it.
///
/// The harness talks in tiers and each provider names its own model, so a
/// topology written against one backend runs unchanged on another and a
/// journaled session survives a provider switch. Concrete ids are still
/// accepted — an old journal is full of them — and resolve to the tier they
/// belonged to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tier {
    Small,
    Medium,
    Large,
}

impl Tier {
    /// What Anthropic calls each tier.
    pub fn anthropic(self) -> &'static str {
        match self {
            Tier::Small => "claude-haiku-4-5",
            Tier::Medium => "claude-sonnet-5",
            Tier::Large => "claude-opus-5",
        }
    }

    pub const NAMES: [&'static str; 3] = ["small", "medium", "large"];

    pub fn parse(name: &str) -> Option<Tier> {
        let name = name.trim();
        match name {
            "small" => Some(Tier::Small),
            "medium" => Some(Tier::Medium),
            "large" => Some(Tier::Large),
            // Legacy concrete ids, so journals and prompts written before
            // tiers existed still resolve.
            m if m.starts_with("claude-haiku") || m.ends_with("-luna") => Some(Tier::Small),
            m if m.starts_with("claude-sonnet") || m.ends_with("-terra") => Some(Tier::Medium),
            m if m.starts_with("claude-opus")
                || m.starts_with("claude-fable")
                || m.starts_with("claude-mythos")
                || m.ends_with("-sol") => Some(Tier::Large),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Small => "small",
            Tier::Medium => "medium",
            Tier::Large => "large",
        }
    }
}
/// What the root process comes up as. Everything it spawns inherits its model
/// unless the spawn names one, so this is the ceiling of the whole tree rather
/// than only the first process — put a cheaper model on the workers.
const DEFAULT_MODEL: &str = "large";
/// Root's reasoning effort. Spawned processes inherit it the same way.
pub const DEFAULT_EFFORT: &str = "high";
const MAX_TOKENS: u64 = 64_000;
const MAX_ATTEMPTS: u32 = 4;

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
struct Caps {
    adaptive: bool,
    effort: bool,
    fallbacks: bool,
    compaction: bool,
}

impl Caps {
    /// A starting guess from the model id. Unknown models get the modern
    /// surface and are corrected by the first rejection rather than being
    /// permanently downgraded on a name we failed to recognize.
    fn guess(model: &str) -> Caps {
        let modern = Caps { adaptive: true, effort: true, fallbacks: false, compaction: true };
        match model {
            m if m.starts_with("claude-fable-") || m.starts_with("claude-mythos-") => {
                Caps { fallbacks: true, ..modern }
            }
            m if m.starts_with("claude-opus-5") => Caps { fallbacks: true, ..modern },
            m if m.starts_with("claude-haiku-") => {
                Caps { adaptive: false, effort: false, fallbacks: false, compaction: false }
            }
            m if m.starts_with("claude-opus-4-5") || m.starts_with("claude-sonnet-4-5") => {
                Caps { adaptive: false, ..modern }
            }
            _ => modern,
        }
    }
}

/// Which backend a turn is sent to. The rest of the harness is unaware: it
/// hands over a `Turn` and receives a `FinalMessage` either way.
pub enum Provider {
    Anthropic,
    /// OpenAI's Responses API, authenticated with the Codex CLI's stored
    /// ChatGPT credentials.
    Codex(tokio::sync::Mutex<crate::codex::Auth>),
}

pub struct Client {
    http: reqwest::Client,
    provider: Provider,
    auth: Auth,
    base_url: String,
    pub model: String,
    /// Server-side compaction: on by default, latched off if the server
    /// rejects it (unsupported model, account without the beta) so a whole
    /// run doesn't die on a feature we can degrade without.
    compaction: AtomicBool,
    /// Per-model feature flags, seeded by `Caps::guess` and narrowed whenever
    /// the API tells us a parameter is unsupported. One rejected turn teaches
    /// the client permanently instead of failing every turn after it.
    caps: std::sync::Mutex<HashMap<String, Caps>>,
    /// Codex only: per-process (last response id, messages already sent).
    /// Threading is what makes a long conversation cheap — without it every
    /// turn re-transmits the whole thing.
    threads: std::sync::Mutex<HashMap<String, (String, usize)>>,
}

/// One request's worth of per-process configuration. Model and effort vary by
/// process; `tools` deliberately does not — it renders before `system` in the
/// cache prefix, so any variation there would fork the shared prefix at
/// position zero and cost more than the tokens it saved.
pub struct Turn<'a> {
    /// Which process this turn belongs to, so a server-side thread can be kept
    /// per process rather than per client.
    pub process: &'a str,
    pub system: &'a Value,
    pub messages: &'a [Value],
    pub tools: &'a Value,
    pub model: &'a str,
    pub effort: Option<&'a str>,
}

pub struct FinalMessage {
    /// Codex only: the id to thread the next turn onto.
    pub thread: Option<String>,
    pub content: Vec<Value>,
    pub stop_reason: String,
    /// Total prompt size for this turn: uncached + cache-write + cache-read.
    /// This is the number compaction watches, so it's what we surface.
    pub input_tokens: u64,
}

impl Client {
    pub fn from_env() -> Result<Self> {
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
        // Codex when asked for, or whenever its credentials are present and no
        // Anthropic key is: the common case is a machine set up for one or the
        // other, and guessing wrong costs a failed turn rather than money.
        let wants_codex = matches!(std::env::var("BITTY_PROVIDER").as_deref(), Ok("codex"));
        let provider = if wants_codex {
            Provider::Codex(tokio::sync::Mutex::new(crate::codex::Auth::load()?))
        } else {
            Provider::Anthropic
        };
        Ok(Client {
            provider,
            http: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(30))
                .build()?,
            auth,
            base_url: std::env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com".into()),
            model: std::env::var("BITTY_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into()),
            compaction: AtomicBool::new(
                !matches!(std::env::var("BITTY_COMPACTION").as_deref(), Ok("off")),
            ),
            caps: std::sync::Mutex::new(HashMap::new()),
            threads: std::sync::Mutex::new(HashMap::new()),
        })
    }

    pub fn compaction_enabled(&self) -> bool {
        self.compaction.load(Ordering::Relaxed)
    }

    fn caps(&self, model: &str) -> Caps {
        *self
            .caps
            .lock()
            .unwrap()
            .entry(model.to_string())
            .or_insert_with(|| Caps::guess(model))
    }

    /// Read a rejection and turn off whatever the server named. Returns true
    /// when something changed, meaning the caller should retry.
    fn learn_from(&self, model: &str, text: &str, tag: &Tag) -> bool {
        let mut caps = self.caps.lock().unwrap();
        let entry = caps.entry(model.to_string()).or_insert_with(|| Caps::guess(model));
        let before = (entry.adaptive, entry.effort, entry.fallbacks, entry.compaction);
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
        let changed = before != (entry.adaptive, entry.effort, entry.fallbacks, entry.compaction);
        if changed {
            ui::warn(
                tag,
                &format!("{model} rejected a request parameter ({text}); retrying without it"),
            );
        }
        changed
    }

    /// One model turn: send the conversation, stream the response (printing
    /// text live under `tag`), return the accumulated message.
    pub async fn message(&self, turn: Turn<'_>, tag: &Tag) -> Result<FinalMessage> {
        if let Provider::Codex(auth) = &self.provider {
            return self.codex_message(auth, turn, tag).await;
        }
        let mut delay = Duration::from_secs(1);
        for attempt in 1..=MAX_ATTEMPTS {
            // Rebuilt per attempt so it reflects a latched-off compaction flag.
            let body = self.build_body(&turn);
            match self.attempt(turn.model, &body, tag).await {
                Ok(msg) => return Ok(msg),
                Err(e)
                    if e.to_string().contains("HTTP 400")
                        && self.learn_from(turn.model, &e.to_string(), tag) =>
                {
                    continue;
                }
                Err(e) if attempt < MAX_ATTEMPTS && is_retryable(&e) => {
                    ui::warn(tag, &format!("API error ({e}); retrying in {delay:?}"));
                    tokio::time::sleep(delay).await;
                    delay *= 2;
                }
                Err(e) => return Err(e),
            }
        }
        unreachable!()
    }

    /// One turn against the Responses API. A 401 means the stored access token
    /// has aged out, so refresh once and retry rather than failing a turn for
    /// something the harness can fix itself.
    async fn codex_message(
        &self,
        auth: &tokio::sync::Mutex<crate::codex::Auth>,
        turn: Turn<'_>,
        tag: &Tag,
    ) -> Result<FinalMessage> {
        let tier = Tier::parse(turn.model).unwrap_or(Tier::Large);
        // Resume this process's thread if the server still has it, sending only
        // what it has not seen. A thread that is unknown or stale is rebuilt
        // from the full history, which the journal always has.
        let resumed = self.threads.lock().unwrap().get(turn.process).cloned();
        let (previous, already) = match resumed {
            Some((id, sent)) if sent <= turn.messages.len() => (Some(id), sent),
            _ => (None, 0),
        };
        let fresh = &turn.messages[already..];
        let body = crate::codex::body(
            tier,
            turn.effort,
            turn.system,
            if previous.is_some() { fresh } else { turn.messages },
            turn.tools,
            previous.as_deref(),
        );
        for attempt in 1..=3 {
            let (token, account) = {
                let auth = auth.lock().await;
                (auth.access_token.clone(), auth.account_id.clone())
            };
            let resp = self
                .http
                .post(crate::codex::endpoint())
                .bearer_auth(&token)
                .header("ChatGPT-Account-Id", &account)
                .header("OpenAI-Beta", "responses=experimental")
                .header("originator", "codex_cli_rs")
                .header("accept", "text/event-stream")
                .json(&body)
                .send()
                .await
                .context("request failed")?;
            let status = resp.status();
            if status == reqwest::StatusCode::UNAUTHORIZED && attempt == 1 {
                ui::trace(tag, "  … Codex token expired; refreshing");
                auth.lock().await.refresh(&self.http).await?;
                continue;
            }
            if !status.is_success() {
                let text = resp.text().await.unwrap_or_default();
                // The server forgot this thread: drop it and let the next
                // attempt send the conversation in full.
                if previous.is_some() && (status == reqwest::StatusCode::NOT_FOUND
                    || text.contains("previous_response_id"))
                {
                    ui::trace(tag, "  … thread expired; resending the conversation");
                    self.threads.lock().unwrap().remove(turn.process);
                    return Box::pin(self.codex_message(auth, turn, tag)).await;
                }
                bail!("HTTP {status}: {text}");
            }
            let done = self.consume_codex_stream(resp, tag).await?;
            if let Some(id) = &done.thread {
                self.threads
                    .lock()
                    .unwrap()
                    .insert(turn.process.to_string(), (id.clone(), turn.messages.len()));
            }
            return Ok(done);
        }
        bail!("could not authenticate against Codex")
    }

    async fn consume_codex_stream(
        &self,
        resp: reqwest::Response,
        tag: &Tag,
    ) -> Result<FinalMessage> {
        let mut acc = crate::codex::Stream::default();
        let mut line_buf = String::new();
        let mut buf = String::new();
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            buf.push_str(&String::from_utf8_lossy(&chunk.context("stream interrupted")?));
            while let Some(cut) = buf.find('\n') {
                let line = buf[..cut].trim().to_string();
                buf.drain(..=cut);
                let Some(payload) = line.strip_prefix("data:") else {
                    continue;
                };
                let payload = payload.trim();
                if payload.is_empty() || payload == "[DONE]" {
                    continue;
                }
                let Ok(event) = serde_json::from_str::<Value>(payload) else {
                    continue;
                };
                if let Some(text) = acc.apply(&event) {
                    line_buf.push_str(&text);
                    while let Some(cut) = line_buf.find('\n') {
                        ui::say(tag, &line_buf[..cut]);
                        line_buf.drain(..=cut);
                    }
                }
            }
        }
        if !line_buf.is_empty() {
            ui::say(tag, &line_buf);
        }
        let done = acc.finish()?;
        Ok(FinalMessage {
            thread: done.id,
            content: done.content,
            stop_reason: done.stop_reason,
            input_tokens: done.input_tokens,
        })
    }

    fn build_body(&self, turn: &Turn<'_>) -> Value {
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
        if self.compaction_enabled() && caps.compaction {
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
    fn beta_header(&self, model: &str) -> Option<String> {
        let caps = self.caps(model);
        let mut betas = Vec::new();
        if caps.fallbacks {
            betas.push(BETA_FALLBACK);
        }
        if self.compaction_enabled() && caps.compaction {
            betas.push(BETA_COMPACT);
        }
        if matches!(self.auth, Auth::Bearer(_)) {
            betas.push(BETA_OAUTH);
        }
        (!betas.is_empty()).then(|| betas.join(","))
    }

    async fn attempt(&self, model: &str, body: &Value, tag: &Tag) -> Result<FinalMessage> {
        let mut req = self
            .http
            .post(format!("{}/v1/messages", self.base_url))
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json");
        if let Some(betas) = self.beta_header(model) {
            req = req.header("anthropic-beta", betas);
        }
        req = match &self.auth {
            Auth::ApiKey(key) => req.header("x-api-key", key),
            Auth::Bearer(token) => req.bearer_auth(token),
        };

        let resp = req.json(body).send().await.context("request failed")?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            let msg = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| v["error"]["message"].as_str().map(String::from))
                .unwrap_or(text);
            bail!("HTTP {status}: {msg}");
        }

        self.consume_stream(resp, tag).await
    }

    async fn consume_stream(&self, resp: reqwest::Response, tag: &Tag) -> Result<FinalMessage> {
        let mut content: Vec<Value> = Vec::new();
        let mut tool_json: HashMap<usize, String> = HashMap::new();
        let mut stop_reason = String::new();
        let mut input_tokens: u64 = 0;
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
                    let event: Value =
                        serde_json::from_str(data.trim()).context("bad SSE payload")?;
                    self.apply_event(
                        &event,
                        &mut content,
                        &mut tool_json,
                        &mut stop_reason,
                        &mut input_tokens,
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
            input_tokens,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_event(
        &self,
        event: &Value,
        content: &mut Vec<Value>,
        tool_json: &mut HashMap<usize, String>,
        stop_reason: &mut String,
        input_tokens: &mut u64,
        line_buf: &mut String,
        tag: &Tag,
    ) -> Result<()> {
        match event["type"].as_str().unwrap_or("") {
            "message_start" => {
                let usage = &event["message"]["usage"];
                // The prompt total is split across three counters depending on
                // cache state; compaction cares about their sum.
                *input_tokens = ["input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"]
                    .iter()
                    .filter_map(|k| usage[k].as_u64())
                    .sum();
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
}

fn append_str(block: &mut Value, key: &str, add: &str) {
    if add.is_empty() {
        return;
    }
    let mut current = block[key].as_str().unwrap_or("").to_string();
    current.push_str(add);
    block[key] = Value::String(current);
}

fn is_retryable(e: &anyhow::Error) -> bool {
    let text = e.to_string();
    text.contains("HTTP 429")
        || text.contains("HTTP 5")
        || text.contains("overloaded_error")
        || text.contains("request failed")
        || text.contains("stream interrupted")
}
