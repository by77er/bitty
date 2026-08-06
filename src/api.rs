//! The provider contract and the one retry driver.
//!
//! A backend (`anthropic.rs`, `codex.rs`) sends a single attempt and reports
//! failures; it never sleeps and never loops. All retry policy — backoff,
//! failure classification, attempt budgets, billable accounting — lives in
//! `Client::drive`, written once. That split exists because it was violated
//! once: the Codex path had its own retry loop and quietly lacked the
//! mid-stream-failure case the Anthropic loop had always handled, so every
//! network blip surfaced as a failed turn. A new backend implements
//! `Backend` and cannot repeat that mistake.
//!
//! The internal message shape is Anthropic's, as raw `serde_json::Value`
//! blocks: the journal is written in it, thinking-block signatures and
//! server-compaction blocks round-trip through it untouched, and other
//! providers translate per request (see codex.rs). Adopting a third-party
//! provider crate would put a typed model exactly where that invariant
//! lives, which is why there isn't one.

use crate::ui::{self, Tag};
use anyhow::Result;
use serde_json::Value;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

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
/// Root's reasoning effort; spawned processes default to low instead.
pub const DEFAULT_EFFORT: &str = "high";
/// Attempts per turn, shared by every backend.
const MAX_ATTEMPTS: u32 = 5;

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
    /// What the turn actually cost: uncached input + cache writes + output.
    /// Cache reads are excluded — counting them makes a long-running system
    /// exhaust any budget by re-reading its own context. Filled by the
    /// backend, accumulated by the driver.
    pub billable: u64,
}

/// A failed attempt: the error, plus any pacing the server volunteered.
pub struct Failure {
    pub error: anyhow::Error,
    /// Server-provided delay from a 429, if any.
    pub retry_after: Option<String>,
}

impl Failure {
    pub fn plain(error: anyhow::Error) -> Failure {
        Failure { error, retry_after: None }
    }
}

/// One provider. Implementations send exactly one attempt per call and
/// report failures; the driver owns every retry decision.
pub trait Backend {
    /// One request: build, send, consume the stream. No retries, no sleeping.
    async fn attempt(
        &self,
        client: &Client,
        turn: &Turn<'_>,
        tag: &Tag,
    ) -> Result<FinalMessage, Failure>;

    /// Provider-specific repair after a failed attempt — drop a parameter the
    /// server rejected, refresh an aged-out token. True means something
    /// changed and an immediate retry is worthwhile; false hands the decision
    /// back to the driver's classification.
    async fn recover(&self, client: &Client, failure: &Failure, turn: &Turn<'_>, tag: &Tag)
    -> bool;

    /// Roughly how many prompt tokens this provider's model for `tier`
    /// accepts.
    fn context_window(&self, tier: Tier) -> u64;
}

/// The configured backend. New providers add a variant here and an arm in
/// the three matches below — everything else (retry, backoff, budgets,
/// accounting) is inherited from the driver.
enum Backends {
    Anthropic(crate::anthropic::Anthropic),
    Codex(crate::codex::Codex),
}

/// Exponential backoff, capped, with jitter, and deferring to `Retry-After`
/// when the server sends one.
///
/// Jitter matters more here than in most clients: a tree of processes shares
/// one account, so they hit the same limit at the same moment and would
/// otherwise wake together and collide again. Spreading the retries is what
/// turns a thundering herd back into a queue.
fn backoff(attempt: u32, retry_after: Option<&str>) -> Duration {
    // The server knows better than we do.
    if let Some(secs) = retry_after.and_then(|v| v.trim().parse::<u64>().ok()) {
        return Duration::from_secs(secs.clamp(1, 120));
    }
    let seconds = 1u64 << attempt.clamp(1, 6);
    // Up to a quarter of the interval, from the clock rather than a dependency.
    let spread = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0)
        % (seconds * 250).max(1);
    Duration::from_millis(seconds * 1000 + spread)
}

pub struct Client {
    http: reqwest::Client,
    backend: Backends,
    pub model: String,
    /// Server-side compaction: on by default, off via BITTY_COMPACTION=off.
    /// Per-model support is the backend's own knowledge; this is the global
    /// switch.
    compaction: AtomicBool,
    /// Cumulative billable tokens this run, across every process.
    billable: AtomicU64,
}

impl Client {
    pub fn from_env() -> Result<Self> {
        // Codex when asked for; Anthropic otherwise. Each backend loads only
        // its own credentials — a Codex run no longer needs a dummy Anthropic
        // key just to start.
        let backend = if matches!(std::env::var("BITTY_PROVIDER").as_deref(), Ok("codex")) {
            Backends::Codex(crate::codex::Codex::from_env()?)
        } else {
            Backends::Anthropic(crate::anthropic::Anthropic::from_env()?)
        };
        Ok(Client {
            backend,
            http: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(30))
                .build()?,
            model: std::env::var("BITTY_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into()),
            compaction: AtomicBool::new(
                !matches!(std::env::var("BITTY_COMPACTION").as_deref(), Ok("off")),
            ),
            billable: AtomicU64::new(0),
        })
    }

    pub(crate) fn http(&self) -> &reqwest::Client {
        &self.http
    }

    /// Billable tokens spent so far this run, for `--max-tokens`.
    pub fn billable_spent(&self) -> u64 {
        self.billable.load(Ordering::Relaxed)
    }

    pub fn compaction_enabled(&self) -> bool {
        self.compaction.load(Ordering::Relaxed)
    }

    /// Whether the provider will compact *this model's* context for us.
    /// Anthropic rewrites the context server-side and hands back blocks that
    /// round-trip — but only for models whose caps accept the beta; Codex has
    /// no equivalent. When this is false the harness summarises the
    /// conversation itself before it outgrows the window.
    pub fn compacts_for_us(&self, model: &str) -> bool {
        match &self.backend {
            Backends::Anthropic(b) => self.compaction_enabled() && b.caps(model).compaction,
            Backends::Codex(_) => false,
        }
    }

    /// Roughly how many prompt tokens this model accepts. The client-side
    /// compaction trigger works in real tokens against this, minus a reserve
    /// for output and the compaction turn itself.
    pub fn context_window(&self, model: &str) -> u64 {
        if let Ok(v) = std::env::var("BITTY_CONTEXT_WINDOW") {
            if let Ok(n) = v.parse() {
                return n;
            }
        }
        let tier = Tier::parse(model).unwrap_or(Tier::Large);
        match &self.backend {
            Backends::Anthropic(b) => b.context_window(tier),
            Backends::Codex(b) => b.context_window(tier),
        }
    }

    /// One model turn: send the conversation, stream the response (printing
    /// text live under `tag`), return the accumulated message.
    pub async fn message(&self, turn: Turn<'_>, tag: &Tag) -> Result<FinalMessage> {
        match &self.backend {
            Backends::Anthropic(b) => self.drive(b, turn, tag).await,
            Backends::Codex(b) => self.drive(b, turn, tag).await,
        }
    }

    /// The retry driver every backend runs under.
    async fn drive<B: Backend>(&self, backend: &B, turn: Turn<'_>, tag: &Tag) -> Result<FinalMessage> {
        for attempt in 1..=MAX_ATTEMPTS {
            let failure = match backend.attempt(self, &turn, tag).await {
                Ok(msg) => {
                    self.billable.fetch_add(msg.billable, Ordering::Relaxed);
                    return Ok(msg);
                }
                Err(failure) => failure,
            };
            if attempt == MAX_ATTEMPTS {
                return Err(failure.error);
            }
            // Provider-specific repair first: a dropped parameter or a fresh
            // token makes an immediate retry worthwhile, no backoff needed.
            if backend.recover(self, &failure, &turn, tag).await {
                continue;
            }
            if classify(&failure.error).retryable() {
                let delay = backoff(attempt, failure.retry_after.as_deref());
                ui::warn(tag, &format!("API error ({:#}); retrying in {delay:?}", failure.error));
                tokio::time::sleep(delay).await;
                continue;
            }
            return Err(failure.error);
        }
        unreachable!()
    }
}

/// What kind of failure a turn died of. Decisions — retry, compact, give up —
/// are made on the kind, not by string-matching at each call site.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FailureKind {
    /// 429 — back off and retry.
    RateLimit,
    /// The provider is melting; same treatment as a 5xx.
    Overloaded,
    /// 5xx — transient server error.
    Server,
    /// The request never completed: connect failure, dropped stream.
    Network,
    /// The prompt no longer fits the model's context window. Not retryable
    /// as-is — the recovery is to compact and try again.
    Overflow,
    /// Credentials rejected. Retrying cannot help (though a backend may be
    /// able to refresh a token in `recover`).
    Auth,
    /// The request itself was malformed or refused for shape. Not retryable.
    Invalid,
    Unknown,
}

impl FailureKind {
    pub fn retryable(self) -> bool {
        matches!(
            self,
            FailureKind::RateLimit | FailureKind::Overloaded | FailureKind::Server | FailureKind::Network
        )
    }
}

/// Classify an error by its text. The overflow patterns cover both providers'
/// phrasings — the cost of missing one is a process that stalls instead of
/// compacting, so the match is deliberately loose.
pub fn classify(e: &anyhow::Error) -> FailureKind {
    let text = e.to_string();
    let lower = text.to_lowercase();
    const OVERFLOW: [&str; 6] = [
        "prompt is too long",
        "context_length_exceeded",
        "maximum context length",
        "exceeds the context window",
        "input length and `max_tokens` exceed",
        "context window exceed",
    ];
    if OVERFLOW.iter().any(|p| lower.contains(p)) {
        return FailureKind::Overflow;
    }
    if text.contains("HTTP 429") {
        return FailureKind::RateLimit;
    }
    if lower.contains("overloaded_error") || text.contains("HTTP 529") {
        return FailureKind::Overloaded;
    }
    if text.contains("HTTP 5") {
        return FailureKind::Server;
    }
    if text.contains("request failed") || text.contains("stream interrupted") {
        return FailureKind::Network;
    }
    if text.contains("HTTP 401") || text.contains("HTTP 403") || lower.contains("authentication") {
        return FailureKind::Auth;
    }
    if text.contains("HTTP 4") {
        return FailureKind::Invalid;
    }
    FailureKind::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    /// The kinds that drive control flow: overflow compacts, 429/5xx retry,
    /// auth and shape errors give up.
    #[test]
    fn failures_classify() {
        let cases = [
            ("HTTP 400: prompt is too long: 1053000 tokens > 1000000 maximum", FailureKind::Overflow),
            ("HTTP 400: This model's maximum context length is 400000 tokens", FailureKind::Overflow),
            ("HTTP 400: context_length_exceeded", FailureKind::Overflow),
            ("HTTP 429: rate limited", FailureKind::RateLimit),
            ("HTTP 529: overloaded", FailureKind::Overloaded),
            ("stream error (overloaded_error): busy", FailureKind::Overloaded),
            ("HTTP 500: internal", FailureKind::Server),
            ("request failed", FailureKind::Network),
            ("stream interrupted", FailureKind::Network),
            ("HTTP 401: bad key", FailureKind::Auth),
            ("HTTP 400: tools.3: unknown field", FailureKind::Invalid),
        ];
        for (text, want) in cases {
            let got = classify(&anyhow!("{text}"));
            assert_eq!(got, want, "{text}");
        }
    }

    /// Overflow must not be retried verbatim — the caller compacts instead.
    #[test]
    fn overflow_is_not_retryable() {
        assert!(!FailureKind::Overflow.retryable());
        assert!(FailureKind::RateLimit.retryable());
        assert!(!FailureKind::Invalid.retryable());
    }

    /// Each attempt must wait longer than the last, or it is not backoff.
    #[test]
    fn backoff_grows() {
        let mut last = Duration::ZERO;
        for attempt in 1..=5 {
            let wait = backoff(attempt, None);
            assert!(wait > last, "attempt {attempt} waited {wait:?}, not longer than {last:?}");
            last = wait;
        }
    }

    /// And must stop growing, so a long outage does not park a process for an
    /// hour.
    #[test]
    fn backoff_is_capped() {
        assert!(backoff(30, None) <= Duration::from_secs(80));
    }

    /// The server's own advice wins over our guess, within reason.
    #[test]
    fn retry_after_is_honored() {
        assert_eq!(backoff(1, Some("7")), Duration::from_secs(7));
        assert_eq!(backoff(5, Some("  3 ")), Duration::from_secs(3));
        // Absurd values are clamped rather than trusted.
        assert_eq!(backoff(1, Some("99999")), Duration::from_secs(120));
        // Anything unparseable falls back to the exponential schedule.
        assert!(backoff(2, Some("in a bit")) >= Duration::from_secs(4));
    }

    /// Jitter has to actually spread, or a tree of processes retries in
    /// lockstep and collides again.
    #[test]
    fn backoff_is_jittered() {
        let base = Duration::from_secs(8);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..50 {
            let wait = backoff(3, None);
            assert!(wait >= base && wait < base + Duration::from_secs(3));
            seen.insert(wait.as_millis());
            std::thread::sleep(Duration::from_micros(50));
        }
        assert!(seen.len() > 1, "every retry waited exactly {base:?}");
    }
}
