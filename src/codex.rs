//! The Codex backend: OpenAI's Responses API, authenticated with the ChatGPT
//! OAuth credentials the Codex CLI already stores.
//!
//! Bitty's conversations stay in Anthropic shape everywhere — history, journal,
//! checkpoints, restore — and are translated to Responses shape per request.
//! That is deliberate: the journal format is the durable artifact, and making
//! it depend on whichever provider happened to be configured would mean a
//! session could not be resumed after a switch.
//!
//! Three things differ on the wire and are handled here: tools carry
//! `parameters` rather than `input_schema`, tool results are `input` items
//! rather than blocks inside a user message, and the streaming events have
//! their own names.

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

const RESPONSES_URL: &str = "https://chatgpt.com/backend-api/codex/responses";
const REFRESH_URL: &str = "https://auth.openai.com/oauth/token";
/// The Codex CLI's own client id, which is what the stored refresh token was
/// issued to — a refresh presented under any other id is rejected.
const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

/// What this provider calls each tier.
pub fn model_for(tier: crate::api::Tier) -> &'static str {
    match tier {
        crate::api::Tier::Large => "gpt-5.6-sol",
        crate::api::Tier::Medium => "gpt-5.6-terra",
        crate::api::Tier::Small => "gpt-5.6-luna",
    }
}

/// The credentials the Codex CLI stores, plus enough to refresh them.
pub struct Auth {
    path: std::path::PathBuf,
    pub access_token: String,
    pub account_id: String,
    refresh_token: String,
}

impl Auth {
    pub fn load() -> Result<Auth> {
        let path = dirs_home()
            .ok_or_else(|| anyhow!("no home directory"))?
            .join(".codex/auth.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let parsed: Value = serde_json::from_str(&text)?;
        let tokens = &parsed["tokens"];
        let read = |key: &str| -> Result<String> {
            tokens[key]
                .as_str()
                .map(String::from)
                .ok_or_else(|| anyhow!("{} has no tokens.{key}", path.display()))
        };
        Ok(Auth {
            access_token: read("access_token")?,
            account_id: read("account_id")?,
            refresh_token: read("refresh_token")?,
            path,
        })
    }

    /// Exchange the refresh token for a new access token and write it back, so
    /// the CLI and this harness stay on the same credential rather than each
    /// invalidating the other's.
    pub async fn refresh(&mut self, http: &reqwest::Client) -> Result<()> {
        let response = http
            .post(REFRESH_URL)
            .json(&json!({
                "client_id": CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
                "scope": "openid profile email",
            }))
            .send()
            .await?;
        if !response.status().is_success() {
            bail!("refreshing the Codex token failed: HTTP {}", response.status());
        }
        let body: Value = response.json().await?;
        let Some(access) = body["access_token"].as_str() else {
            bail!("the refresh response carried no access_token");
        };
        self.access_token = access.to_string();
        if let Some(refresh) = body["refresh_token"].as_str() {
            self.refresh_token = refresh.to_string();
        }

        // Merge into the file rather than rewriting it: it holds fields this
        // harness does not own, and clobbering them would break the CLI.
        if let Ok(text) = std::fs::read_to_string(&self.path) {
            if let Ok(mut stored) = serde_json::from_str::<Value>(&text) {
                stored["tokens"]["access_token"] = json!(self.access_token);
                stored["tokens"]["refresh_token"] = json!(self.refresh_token);
                if let Ok(serialized) = serde_json::to_string_pretty(&stored) {
                    let _ = std::fs::write(&self.path, serialized);
                }
            }
        }
        Ok(())
    }
}

fn dirs_home() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

/// Translate a Bitty (Anthropic-shaped) conversation into Responses `input`.
///
/// Assistant tool calls and their results are separate top-level items here,
/// not blocks nested in messages, so one Anthropic message can fan out into
/// several items.
pub fn to_input(messages: &[Value]) -> Vec<Value> {
    let mut input = Vec::new();
    for message in messages {
        let role = message["role"].as_str().unwrap_or("user");
        let blocks = match message["content"].as_array() {
            Some(blocks) => blocks.clone(),
            // A bare string body is legal in Anthropic shape.
            None => vec![json!({"type": "text", "text": message["content"].as_str().unwrap_or("")})],
        };

        let mut text_parts = Vec::new();
        for block in &blocks {
            match block["type"].as_str().unwrap_or("") {
                "text" => text_parts.push(block["text"].as_str().unwrap_or("").to_string()),
                "tool_use" => {
                    input.push(json!({
                        "type": "function_call",
                        "call_id": block["id"],
                        "name": block["name"],
                        "arguments": block["input"].to_string(),
                    }));
                }
                "tool_result" => {
                    // Content may be a string or a block list; flatten either.
                    let output = match &block["content"] {
                        Value::String(text) => text.clone(),
                        Value::Array(parts) => parts
                            .iter()
                            .filter_map(|p| p["text"].as_str())
                            .collect::<Vec<_>>()
                            .join("\n"),
                        other => other.to_string(),
                    };
                    input.push(json!({
                        "type": "function_call_output",
                        "call_id": block["tool_use_id"],
                        "output": output,
                    }));
                }
                // Thinking blocks are signed for a different provider entirely;
                // there is nothing meaningful to send.
                _ => {}
            }
        }

        if !text_parts.is_empty() {
            let joined = text_parts.join("\n");
            let kind = if role == "assistant" { "output_text" } else { "input_text" };
            input.push(json!({
                "type": "message",
                "role": role,
                "content": [{"type": kind, "text": joined}],
            }));
        }
    }
    input
}

/// Anthropic tool definitions to Responses function tools.
pub fn to_tools(tools: &Value) -> Vec<Value> {
    tools
        .as_array()
        .map(|list| {
            list.iter()
                .map(|tool| {
                    json!({
                        "type": "function",
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Fold the Responses event stream back into Anthropic-shaped content blocks,
/// which is what the rest of the harness — and the journal — expects.
pub struct Accumulated {
    pub content: Vec<Value>,
    pub stop_reason: String,
    pub input_tokens: u64,
    /// This response's id, to thread the next turn onto.
    pub id: Option<String>,
}

#[derive(Default)]
pub struct Stream {
    text: String,
    calls: Vec<Value>,
    input_tokens: u64,
    id: Option<String>,
    failed: Option<String>,
}

impl Stream {
    /// Apply one decoded SSE data payload. Returns any text to print live.
    pub fn apply(&mut self, event: &Value) -> Option<String> {
        match event["type"].as_str().unwrap_or("") {
            "response.output_text.delta" => {
                let delta = event["delta"].as_str().unwrap_or("");
                self.text.push_str(delta);
                return Some(delta.to_string());
            }
            "response.output_item.done" => {
                let item = &event["item"];
                if item["type"] == "function_call" {
                    // Arguments arrive as a JSON string; the harness wants the
                    // parsed object, and a malformed one should surface as an
                    // empty input rather than killing the turn.
                    let input: Value = item["arguments"]
                        .as_str()
                        .and_then(|a| serde_json::from_str(a).ok())
                        .unwrap_or_else(|| json!({}));
                    self.calls.push(json!({
                        "type": "tool_use",
                        "id": item["call_id"],
                        "name": item["name"],
                        "input": input,
                    }));
                }
            }
            "response.completed" => {
                self.input_tokens = event["response"]["usage"]["input_tokens"]
                    .as_u64()
                    .unwrap_or(0);
                self.id = event["response"]["id"].as_str().map(String::from);
            }
            "response.failed" | "error" => {
                self.failed = Some(
                    event["response"]["error"]["message"]
                        .as_str()
                        .or_else(|| event["message"].as_str())
                        .unwrap_or("the response failed")
                        .to_string(),
                );
            }
            _ => {}
        }
        None
    }

    pub fn finish(self) -> Result<Accumulated> {
        if let Some(why) = self.failed {
            bail!("{why}");
        }
        let mut content = Vec::new();
        if !self.text.is_empty() {
            content.push(json!({"type": "text", "text": self.text}));
        }
        let stop_reason = if self.calls.is_empty() { "end_turn" } else { "tool_use" };
        content.extend(self.calls);
        Ok(Accumulated {
            content,
            stop_reason: stop_reason.to_string(),
            input_tokens: self.input_tokens,
            id: self.id,
        })
    }
}

/// The request body for one turn.
/// One turn's request.
///
/// `previous` is the id of the last response on this process's thread. When it
/// is set the server already holds everything before it, so `messages` is only
/// what is new — which is the whole point: a process with a large conversation
/// stops re-transmitting it every turn and sends a couple of tool results
/// instead. When it is `None` the full history goes, which is what happens on
/// the first turn and after the server forgets a thread.
pub fn body(
    tier: crate::api::Tier,
    effort: Option<&str>,
    system: &Value,
    messages: &[Value],
    tools: &Value,
    previous: Option<&str>,
) -> Value {
    // Bitty's system prompt is a list of cacheable blocks; Responses takes one
    // string, so the cache_control breakpoints simply have nowhere to go.
    let instructions = match system {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| b["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n\n"),
        other => other.to_string(),
    };
    let mut body = json!({
        "model": model_for(tier),
        "instructions": instructions,
        "input": to_input(messages),
        "tools": to_tools(tools),
        "stream": true,
        // Server-side threading: the point of the exercise. Storing lets the
        // next turn reference this one instead of resending the conversation.
        "store": true,
    });
    if let Some(previous) = previous {
        body["previous_response_id"] = json!(previous);
    }
    if let Some(effort) = effort {
        body["reasoning"] = json!({"effort": effort});
    }
    body
}

pub fn endpoint() -> String {
    std::env::var("BITTY_CODEX_URL").unwrap_or_else(|_| RESPONSES_URL.to_string())
}
