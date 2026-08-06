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
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
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
                || m.ends_with("-sol") =>
            {
                Some(Tier::Large)
            }
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

/// The provider's own token split for one request, in the four categories that
/// are billed at different rates. Filled by the backend straight from the
/// response's usage block — every number here is reported, never inferred.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    /// Prompt tokens that missed the cache and were charged in full.
    pub uncached_input: u64,
    /// Prompt tokens written into the cache — dearer than uncached input.
    /// Always 0 on the Codex path, which reports no cache-write count and
    /// charges no surcharge for one.
    pub cache_write: u64,
    /// Prompt tokens served from the cache. Cheap — around a tenth of the
    /// input rate — but never free, which is why they are priced rather than
    /// dropped.
    pub cache_read: u64,
    pub output: u64,
}

impl Usage {
    /// Everything the model read this turn: the number compaction watches.
    pub fn prompt(&self) -> u64 {
        self.uncached_input + self.cache_write + self.cache_read
    }

    /// The rough "paid full price" gauge `--max-tokens` spends against, which
    /// deliberately ignores cache reads. Not a cost basis: cache reads are
    /// billed, just cheaply. Money is computed by `Rates::cost` instead.
    pub fn billable(&self) -> u64 {
        self.uncached_input + self.cache_write + self.output
    }

    fn add(&mut self, other: &Usage) {
        self.uncached_input += other.uncached_input;
        self.cache_write += other.cache_write;
        self.cache_read += other.cache_read;
        self.output += other.output;
    }
}

/// How much to trust a cost figure. Ordered worst-last, so folding many
/// requests together is `max`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Confidence {
    /// Every rate came from `BITTY_PRICES` or from a fresh OpenRouter
    /// snapshot: a published number, not this file's recollection of one.
    #[default]
    Measured,
    /// At least one rate came from `BAKED` below, or from a snapshot older
    /// than `STALE_AFTER`, or stood in for a component a snapshot omitted.
    /// Never present this as an actual cost.
    Estimated,
    /// At least one model had no rates anywhere, so its tokens contributed no
    /// dollars at all and the figure is incomplete.
    Unknown,
}

impl Confidence {
    pub fn as_str(self) -> &'static str {
        match self {
            Confidence::Measured => "measured",
            Confidence::Estimated => "estimated",
            Confidence::Unknown => "unknown",
        }
    }
}

/// USD per million tokens for one model, split the four ways a provider bills.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rates {
    pub input: f64,
    pub cache_write: f64,
    pub cache_read: f64,
    pub output: f64,
}

impl Rates {
    /// What one request cost, each component at its own rate.
    ///
    /// Not derived from `Usage::billable`, and never a blended rate: a cached
    /// conversation — which is most of them here — is mostly cache reads at a
    /// tenth of the input rate, so collapsing the split misprices nearly every
    /// turn the harness makes.
    fn cost(&self, usage: &Usage) -> f64 {
        (usage.uncached_input as f64 * self.input
            + usage.cache_write as f64 * self.cache_write
            + usage.cache_read as f64 * self.cache_read
            + usage.output as f64 * self.output)
            / 1_000_000.0
    }
}

/// Cumulative money, for one process or for a whole run.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spend {
    /// US dollars. `f64` rather than a decimal crate: no new dependencies, and
    /// a run's total is many orders of magnitude away from f64's precision
    /// limit, so the accumulated error stays far below a cent.
    pub usd: f64,
    pub confidence: Confidence,
    /// The tokens behind `usd`, so a caller can show how much of a process's
    /// context was cheap cache hits.
    pub usage: Usage,
}

impl Spend {
    fn add(&mut self, other: Spend) {
        self.usd += other.usd;
        // The weakest link decides: one estimated request makes the total
        // estimated.
        self.confidence = self.confidence.max(other.confidence);
        self.usage.add(&other.usage);
    }
}

/// Baked-in fallback prices, USD per million tokens, transcribed 2026-08-06.
///
/// This table is a point-in-time copy, not a live one — it is stale the moment
/// it is written. That is why anything priced from it reports
/// `Confidence::Estimated` and must never be labelled an actual cost, and why
/// two better sources take precedence over it:
///
///   * `BITTY_PRICES` — a JSON object of per-MILLION-token rates keyed by
///     concrete model id, e.g.
///     `{"claude-opus-5":{"input":5,"cache_write":6.25,"cache_read":0.5,"output":25}}`.
///     All four keys are required. Always wins, and counts as `Measured`:
///     whoever typed it knows their contract better than this file does.
///   * the OpenRouter snapshot (see `refresh_prices`), fetched once per
///     session in the background and cached in `.bitty/prices.json`. Disable
///     with `BITTY_PRICE_FETCH=off`.
///
/// To update the fallback, edit the rows below — nothing else prices anything.
/// Provenance:
///
///   * Every row but `claude-haiku-4-5` was read off the OpenRouter public
///     list on 2026-08-06 — the same source `refresh_prices` pulls, and the
///     only place these models are priced at all, since neither provider
///     returns a cost in its responses or publishes rates through its own API.
///     They are still `Estimated` here: a transcribed copy of a third party's
///     list ages, and we are billed by the provider, not by the aggregator.
///   * `claude-haiku-4-5` — Anthropic's published $1 / $5 per Mtok with the
///     documented cache multipliers (write x1.25 for the 5-minute TTL, read
///     x0.1). OpenRouter carries it only under the dotted spelling
///     `claude-haiku-4.5`, at exactly those four rates, which is a useful
///     cross-check; `openrouter_ids` tries that spelling so this row can be
///     measured rather than assumed.
///
/// KNOWN GAP, deliberately not modelled: long-context surcharges. OpenRouter
/// lists a second gpt-5.6-* rate card for prompts over 272k tokens at roughly
/// double, and Anthropic prices its own >200k tier higher. A very long prompt
/// is therefore under-priced here. Fixing it means rates keyed on prompt size,
/// which is a bigger change than a row edit.
const BAKED: &[(&str, Rates)] = &[
    // Published Anthropic list price; cross-checked against OpenRouter's
    // `claude-haiku-4.5`.
    (
        "claude-haiku-4-5",
        Rates {
            input: 1.00,
            cache_write: 1.25,
            cache_read: 0.10,
            output: 5.00,
        },
    ),
    (
        "claude-sonnet-5",
        Rates {
            input: 2.00,
            cache_write: 2.50,
            cache_read: 0.20,
            output: 10.00,
        },
    ),
    (
        "claude-opus-5",
        Rates {
            input: 5.00,
            cache_write: 6.25,
            cache_read: 0.50,
            output: 25.00,
        },
    ),
    (
        "gpt-5.6-luna",
        Rates {
            input: 0.10,
            cache_write: 0.125,
            cache_read: 0.01,
            output: 0.60,
        },
    ),
    (
        "gpt-5.6-terra",
        Rates {
            input: 1.00,
            cache_write: 1.25,
            cache_read: 0.10,
            output: 6.00,
        },
    ),
    (
        "gpt-5.6-sol",
        Rates {
            input: 5.00,
            cache_write: 6.25,
            cache_read: 0.50,
            output: 30.00,
        },
    ),
];

/// The public model list, which carries per-token prices as strings and needs
/// no key. Neither provider returns cost in a response, and neither publishes
/// prices through its own API, so this is the only runtime price source there
/// is.
const OPENROUTER_MODELS: &str = "https://openrouter.ai/api/v1/models";
/// Where a fetched snapshot is kept, so a later run — offline or not — starts
/// out measured instead of estimating.
const PRICE_CACHE: &str = ".bitty/prices.json";
/// Past this, a snapshot is still used (it beats the baked table) but is
/// reported as an estimate rather than a measurement.
const STALE_AFTER: u64 = 7 * 24 * 60 * 60;

fn baked(id: &str) -> Option<Rates> {
    BAKED
        .iter()
        .find(|(model, _)| *model == id)
        .map(|(_, rates)| *rates)
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Rates plus how much they can be trusted.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Priced {
    rates: Rates,
    confidence: Confidence,
}

/// The price table for this run, in precedence order: `BITTY_PRICES`, then the
/// OpenRouter snapshot, then `BAKED`.
#[derive(Default)]
struct Prices {
    overrides: HashMap<String, Rates>,
    fetched: HashMap<String, Priced>,
    /// Unix seconds at which `fetched` was pulled.
    fetched_at: u64,
}

impl Prices {
    /// Environment overrides plus whatever snapshot is already on disk.
    /// Deliberately does no network I/O: startup never waits on openrouter.ai,
    /// and a run with no cache simply begins by estimating.
    fn load() -> Prices {
        let mut prices = Prices::default();
        if let Ok(json) = std::env::var("BITTY_PRICES") {
            prices.overrides = parse_overrides(&json);
        }
        if let Ok(text) = std::fs::read_to_string(PRICE_CACHE)
            && let Ok(cached) = serde_json::from_str::<Value>(&text)
        {
            prices.fetched = parse_snapshot(&cached["rates"]);
            prices.fetched_at = cached["fetched_at"].as_u64().unwrap_or(0);
        }
        prices
    }

    fn stale(&self) -> bool {
        unix_now().saturating_sub(self.fetched_at) > STALE_AFTER
    }

    fn lookup(&self, id: &str) -> Option<Priced> {
        if let Some(rates) = self.overrides.get(id) {
            return Some(Priced {
                rates: *rates,
                confidence: Confidence::Measured,
            });
        }
        if let Some(priced) = self.fetched.get(id) {
            // An aged snapshot still beats the baked table, but it is no
            // longer a measurement of anything current.
            if self.stale() {
                return Some(Priced {
                    confidence: Confidence::Estimated,
                    ..*priced
                });
            }
            return Some(*priced);
        }
        baked(id).map(|rates| Priced {
            rates,
            confidence: Confidence::Estimated,
        })
    }

    /// Price one request. An id with no row anywhere contributes no dollars
    /// and flips the figure to `Unknown`: a visible gap is honest, whereas a
    /// silent zero presented as a cost is not.
    fn spend(&self, id: &str, usage: &Usage) -> Spend {
        match self.lookup(id) {
            Some(priced) => Spend {
                usd: priced.rates.cost(usage),
                confidence: priced.confidence,
                usage: *usage,
            },
            None => Spend {
                usd: 0.0,
                confidence: Confidence::Unknown,
                usage: *usage,
            },
        }
    }
}

/// Money spent this run. A mutex rather than atomics: dollars are f64, and the
/// total and its per-process parts have to move together.
#[derive(Default)]
struct Ledger {
    total: Spend,
    by_process: HashMap<String, Spend>,
}

/// Parse `BITTY_PRICES`. Per-million-token rates, the same units as `BAKED`.
/// A malformed entry is skipped rather than fatal — a typo in an env var
/// should not stop a run — but it is also not silently half-applied: all four
/// components must be present or the row is ignored.
fn parse_overrides(json: &str) -> HashMap<String, Rates> {
    let mut out = HashMap::new();
    let Ok(value) = serde_json::from_str::<Value>(json) else {
        return out;
    };
    let Some(rows) = value.as_object() else {
        return out;
    };
    for (id, row) in rows {
        if let (Some(input), Some(cache_write), Some(cache_read), Some(output)) = (
            row["input"].as_f64(),
            row["cache_write"].as_f64(),
            row["cache_read"].as_f64(),
            row["output"].as_f64(),
        ) {
            out.insert(
                id.clone(),
                Rates {
                    input,
                    cache_write,
                    cache_read,
                    output,
                },
            );
        }
    }
    out
}

/// Read back the `rates` object of our own cache file.
fn parse_snapshot(rates: &Value) -> HashMap<String, Priced> {
    let mut out = HashMap::new();
    let Some(rows) = rates.as_object() else {
        return out;
    };
    for (id, row) in rows {
        if let (Some(input), Some(cache_write), Some(cache_read), Some(output)) = (
            row["input"].as_f64(),
            row["cache_write"].as_f64(),
            row["cache_read"].as_f64(),
            row["output"].as_f64(),
        ) {
            out.insert(
                id.clone(),
                Priced {
                    rates: Rates {
                        input,
                        cache_write,
                        cache_read,
                        output,
                    },
                    confidence: match row["estimated"].as_bool() {
                        Some(true) => Confidence::Estimated,
                        _ => Confidence::Measured,
                    },
                },
            );
        }
    }
    out
}

/// OpenRouter quotes USD per SINGLE token, as strings. Scaling to per-million
/// is the whole conversion, and the easy mistake: read as per-million, every
/// figure is out by a factor of a million and still looks plausible on screen.
fn per_million(pricing: &Value, key: &str) -> Option<f64> {
    let per_token: f64 = pricing[key].as_str()?.trim().parse().ok()?;
    Some(per_token * 1_000_000.0)
}

/// What the public list might call one of our models: the concrete id behind a
/// provider prefix, and the same id with its version punctuated the other way
/// — `claude-haiku-4-5` here is `claude-haiku-4.5` there, and that one
/// spelling is the difference between measuring the small tier and guessing at
/// it. Anything we cannot name is left to the baked table.
fn openrouter_ids(model: &str) -> Vec<String> {
    let prefix = if model.starts_with("claude") {
        "anthropic/"
    } else if model.starts_with("gpt") {
        "openai/"
    } else {
        return Vec::new();
    };
    let mut names = vec![model.to_string()];
    if let Some(cut) = model.rfind('-') {
        let (head, tail) = model.split_at(cut);
        let minor = &tail[1..];
        if head.ends_with(|c: char| c.is_ascii_digit())
            && !minor.is_empty()
            && minor.chars().all(|c| c.is_ascii_digit())
        {
            names.push(format!("{head}.{minor}"));
        }
    }
    names.iter().map(|name| format!("{prefix}{name}")).collect()
}

fn from_openrouter(body: &Value) -> HashMap<String, Priced> {
    let mut out = HashMap::new();
    let Some(models) = body["data"].as_array() else {
        return out;
    };
    for (id, baked) in BAKED {
        let wanted = openrouter_ids(id);
        let Some(model) = models.iter().find(|m| {
            m["id"]
                .as_str()
                .is_some_and(|listed| wanted.iter().any(|w| w == listed))
        }) else {
            continue;
        };
        let pricing = &model["pricing"];
        let Some(input) = per_million(pricing, "prompt") else {
            continue;
        };
        let Some(output) = per_million(pricing, "completion") else {
            continue;
        };
        let cache_read = per_million(pricing, "input_cache_read");
        let cache_write = per_million(pricing, "input_cache_write");
        // A component the list omits falls back to the baked row, and the row
        // is downgraded with it: three quarters measured is not measured.
        let confidence = if cache_read.is_some() && cache_write.is_some() {
            Confidence::Measured
        } else {
            Confidence::Estimated
        };
        out.insert(
            (*id).to_string(),
            Priced {
                rates: Rates {
                    input,
                    cache_write: cache_write.unwrap_or(baked.cache_write),
                    cache_read: cache_read.unwrap_or(baked.cache_read),
                    output,
                },
                confidence,
            },
        );
    }
    out
}

/// Cache a fetched snapshot next to the session data. Best effort: an
/// unwritable directory is not a reason to disturb a turn.
fn save_snapshot(rates: &HashMap<String, Priced>, at: u64) {
    let rows: serde_json::Map<String, Value> = rates
        .iter()
        .map(|(id, priced)| {
            (
                id.clone(),
                serde_json::json!({
                    "input": priced.rates.input,
                    "cache_write": priced.rates.cache_write,
                    "cache_read": priced.rates.cache_read,
                    "output": priced.rates.output,
                    "estimated": priced.confidence != Confidence::Measured,
                }),
            )
        })
        .collect();
    let body = serde_json::json!({
        "source": OPENROUTER_MODELS,
        "fetched_at": at,
        "units": "usd per million tokens",
        "rates": rows,
    });
    if let Some(dir) = std::path::Path::new(PRICE_CACHE).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(PRICE_CACHE, format!("{body:#}\n"));
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
    /// What the turn actually cost: uncached input + cache writes + output.
    /// Cache reads are excluded — counting them makes a long-running system
    /// exhaust any budget by re-reading its own context. Filled by the
    /// backend, accumulated by the driver.
    pub billable: u64,
    /// The provider's own four-way token split for this turn — the only thing
    /// money can honestly be computed from. `input_tokens` and `billable` are
    /// both derived from it, and kept because compaction and the token budget
    /// read them on the hot path.
    pub usage: Usage,
}

/// A failed attempt: the error, plus any pacing the server volunteered.
pub struct Failure {
    pub error: anyhow::Error,
    /// Server-provided delay from a 429, if any.
    pub retry_after: Option<String>,
}

impl Failure {
    pub fn plain(error: anyhow::Error) -> Failure {
        Failure {
            error,
            retry_after: None,
        }
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
    /// Rates for this run. Shared with the background price fetch, which is
    /// the only writer after startup.
    prices: Arc<Mutex<Prices>>,
    /// One price fetch per session, however many requests go out.
    fetched: AtomicBool,
    /// Dollars, per process and in total.
    ledger: Mutex<Ledger>,
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
            compaction: AtomicBool::new(!matches!(
                std::env::var("BITTY_COMPACTION").as_deref(),
                Ok("off")
            )),
            billable: AtomicU64::new(0),
            prices: Arc::new(Mutex::new(Prices::load())),
            fetched: AtomicBool::new(false),
            ledger: Mutex::new(Ledger::default()),
        })
    }

    pub(crate) fn http(&self) -> &reqwest::Client {
        &self.http
    }

    /// Billable tokens spent so far this run, for `--max-tokens`.
    pub fn billable_spent(&self) -> u64 {
        self.billable.load(Ordering::Relaxed)
    }

    /// What this run has cost so far, and how much to trust the figure.
    pub fn spend_total(&self) -> Spend {
        self.ledger.lock().unwrap().total
    }

    /// What one process has cost so far. A process that has never called a
    /// model is a real, measured zero.
    pub fn spend_for(&self, process: &str) -> Spend {
        self.ledger
            .lock()
            .unwrap()
            .by_process
            .get(process)
            .copied()
            .unwrap_or_default()
    }

    /// Price one completed request and add it to both ledgers. Called once per
    /// *successful* attempt by `drive`, so a retried turn cannot be counted
    /// twice and a failed one is not counted at all.
    pub(crate) fn charge(&self, process: &str, model: &str, usage: &Usage) {
        // Price the model the request was actually served by: the backends
        // resolve a tier — or an unrecognised id — to a concrete model, and the
        // bill follows that rather than the string the process asked for.
        let tier = Tier::parse(model).unwrap_or(Tier::Large);
        let id = match &self.backend {
            Backends::Anthropic(_) => tier.anthropic(),
            Backends::Codex(_) => crate::codex::model_for(tier),
        };
        let cost = self.prices.lock().unwrap().spend(id, usage);
        let mut ledger = self.ledger.lock().unwrap();
        ledger.total.add(cost);
        ledger
            .by_process
            .entry(process.to_string())
            .or_default()
            .add(cost);
    }

    /// Pull the public OpenRouter price list once per session, in the
    /// background, because neither provider will tell us what anything costs.
    ///
    /// Off the hot path by construction: nothing awaits it, a failure is
    /// ignored, and it is fired only after a request has already gone out, so a
    /// run that never calls a model never calls out either. It sends no key and
    /// no user data — an unauthenticated GET of a public list — and announces
    /// itself once rather than reaching the network silently.
    /// `BITTY_PRICE_FETCH=off` leaves the baked table in charge.
    fn refresh_prices(&self) {
        if matches!(std::env::var("BITTY_PRICE_FETCH").as_deref(), Ok("off")) {
            return;
        }
        if self.fetched.swap(true, Ordering::Relaxed) {
            return;
        }
        {
            // A fresh snapshot on disk is already the best answer available.
            let prices = self.prices.lock().unwrap();
            if !prices.fetched.is_empty() && !prices.stale() {
                return;
            }
        }
        let http = self.http.clone();
        let prices = self.prices.clone();
        tokio::spawn(async move {
            let Ok(resp) = http
                .get(OPENROUTER_MODELS)
                .timeout(Duration::from_secs(20))
                .send()
                .await
            else {
                return;
            };
            let Ok(body) = resp.json::<Value>().await else {
                return;
            };
            let rates = from_openrouter(&body);
            if rates.is_empty() {
                return;
            }
            let at = unix_now();
            {
                let mut prices = prices.lock().unwrap();
                prices.fetched = rates.clone();
                prices.fetched_at = at;
            }
            save_snapshot(&rates, at);
            ui::system(&format!(
                "prices: {} model rates from openrouter.ai, cached in {PRICE_CACHE}",
                rates.len()
            ));
        });
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
        if let Ok(v) = std::env::var("BITTY_CONTEXT_WINDOW")
            && let Ok(n) = v.parse()
        {
            return n;
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
    async fn drive<B: Backend>(
        &self,
        backend: &B,
        turn: Turn<'_>,
        tag: &Tag,
    ) -> Result<FinalMessage> {
        for attempt in 1..=MAX_ATTEMPTS {
            let failure = match backend.attempt(self, &turn, tag).await {
                Ok(msg) => {
                    self.billable.fetch_add(msg.billable, Ordering::Relaxed);
                    self.charge(turn.process, turn.model, &msg.usage);
                    // After charging, never before: the next turn gets the
                    // better rates, this one is never delayed for them.
                    self.refresh_prices();
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
                ui::warn(
                    tag,
                    &format!("API error ({:#}); retrying in {delay:?}", failure.error),
                );
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
            FailureKind::RateLimit
                | FailureKind::Overloaded
                | FailureKind::Server
                | FailureKind::Network
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
            (
                "HTTP 400: prompt is too long: 1053000 tokens > 1000000 maximum",
                FailureKind::Overflow,
            ),
            (
                "HTTP 400: This model's maximum context length is 400000 tokens",
                FailureKind::Overflow,
            ),
            ("HTTP 400: context_length_exceeded", FailureKind::Overflow),
            ("HTTP 429: rate limited", FailureKind::RateLimit),
            ("HTTP 529: overloaded", FailureKind::Overloaded),
            (
                "stream error (overloaded_error): busy",
                FailureKind::Overloaded,
            ),
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
            assert!(
                wait > last,
                "attempt {attempt} waited {wait:?}, not longer than {last:?}"
            );
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

    /// A realistic cached turn, priced component by component. Hand-computed
    /// against the baked claude-opus-5 row (5 / 6.25 / 0.50 / 25 per Mtok):
    ///   2_000 x 5 + 8_000 x 6.25 + 40_000 x 0.50 + 1_500 x 25
    ///   = 10_000 + 50_000 + 20_000 + 37_500 = 117_500 -> $0.1175
    #[test]
    fn cost_prices_every_component_at_its_own_rate() {
        let usage = Usage {
            uncached_input: 2_000,
            cache_write: 8_000,
            cache_read: 40_000,
            output: 1_500,
        };
        let spend = Prices::default().spend("claude-opus-5", &usage);
        assert!(
            (spend.usd - 0.1175).abs() < 1e-12,
            "got {}, want 0.1175",
            spend.usd
        );
        // Baked rates are never a measurement.
        assert_eq!(spend.confidence, Confidence::Estimated);
        assert_eq!(spend.usage, usage);
    }

    /// Cache reads are cheap, not free. A turn that is almost entirely cache
    /// hits still costs something, and costs a tenth of what the same tokens
    /// would have cost uncached — asserting zero here would enshrine a bug.
    #[test]
    fn cache_reads_are_cheap_but_never_free() {
        let prices = Prices::default();
        let cached = prices.spend(
            "claude-opus-5",
            &Usage {
                cache_read: 100_000,
                ..Usage::default()
            },
        );
        let uncached = prices.spend(
            "claude-opus-5",
            &Usage {
                uncached_input: 100_000,
                ..Usage::default()
            },
        );
        // 100_000 cache reads at $0.50/Mtok = $0.05, against $0.50 had the
        // same tokens missed the cache: a tenth, and emphatically not zero.
        assert!((cached.usd - 0.05).abs() < 1e-12, "got {}", cached.usd);
        assert!(cached.usd > 0.0);
        assert!((uncached.usd - 0.50).abs() < 1e-12, "got {}", uncached.usd);
        assert!((uncached.usd / cached.usd - 10.0).abs() < 1e-9);
    }

    /// An unknown model contributes no dollars and says so. Pricing it at zero
    /// and calling the total actual is the failure this flag exists to prevent.
    #[test]
    fn unknown_model_is_unpriced_rather_than_free() {
        let usage = Usage {
            uncached_input: 1_000,
            output: 1_000,
            ..Usage::default()
        };
        let prices = Prices::default();
        assert_eq!(prices.lookup("claude-opus-9"), None);
        let spend = prices.spend("claude-opus-9", &usage);
        assert_eq!(spend.usd, 0.0);
        assert_eq!(spend.confidence, Confidence::Unknown);
        // And it poisons a total it is folded into, rather than hiding in it.
        let mut total = Spend::default();
        total.add(Prices::default().spend("claude-haiku-4-5", &usage));
        assert_eq!(total.confidence, Confidence::Estimated);
        total.add(spend);
        assert_eq!(total.confidence, Confidence::Unknown);
    }

    /// BITTY_PRICES wins over the baked table and counts as measured. This is
    /// exactly the string `Prices::load` feeds to `parse_overrides`.
    #[test]
    fn env_override_wins_over_the_baked_table() {
        let prices = Prices {
            overrides: parse_overrides(
                r#"{"claude-opus-5":{"input":10,"cache_write":12.5,"cache_read":1,"output":50},
                    "broken":{"input":1}}"#,
            ),
            ..Prices::default()
        };
        let spend = prices.spend(
            "claude-opus-5",
            &Usage {
                uncached_input: 1_000,
                cache_write: 1_000,
                cache_read: 1_000,
                output: 1_000,
            },
        );
        // (10 + 12.5 + 1 + 50) per Mtok on 1_000 tokens each = $0.0735.
        assert!((spend.usd - 0.0735).abs() < 1e-12, "got {}", spend.usd);
        assert_eq!(spend.confidence, Confidence::Measured);
        // A half-specified row is ignored, not half-applied.
        assert!(!prices.overrides.contains_key("broken"));
    }

    /// OpenRouter quotes USD per single token, as strings. These are the exact
    /// values the live endpoint returns; per-million is a factor of 1e6 away.
    #[test]
    fn openrouter_rates_convert_from_per_token_strings() {
        let body = serde_json::json!({"data": [
            {
                "id": "anthropic/claude-opus-5",
                "pricing": {
                    "prompt": "0.00000125",
                    "completion": "0.00000425",
                    "input_cache_read": "0.00000015",
                    "web_search": "0.0025",
                },
            },
            // The small tier is listed under a differently punctuated version,
            // and is the one model with no other measurable source.
            {
                "id": "anthropic/claude-haiku-4.5",
                "pricing": {
                    "prompt": "0.000001",
                    "completion": "0.000005",
                    "input_cache_read": "0.0000001",
                    "input_cache_write": "0.00000125",
                },
            },
        ]});
        let fetched = from_openrouter(&body);
        let priced = fetched.get("claude-opus-5").expect("row parsed");
        assert!((priced.rates.input - 1.25).abs() < 1e-12);
        assert!((priced.rates.output - 4.25).abs() < 1e-12);
        assert!((priced.rates.cache_read - 0.15).abs() < 1e-12);
        // No cache-write rate in this row, so that one component comes from
        // the baked table and the whole row drops to estimated.
        assert!((priced.rates.cache_write - 6.25).abs() < 1e-12);
        assert_eq!(priced.confidence, Confidence::Estimated);
        // A model we are never billed for is not collected at all.
        assert_eq!(fetched.len(), 2);

        // Matched across the `-4-5` / `-4.5` spelling, and complete, so it
        // counts as measured.
        let haiku = fetched.get("claude-haiku-4-5").expect("alias matched");
        assert!((haiku.rates.input - 1.0).abs() < 1e-12);
        assert!((haiku.rates.cache_write - 1.25).abs() < 1e-12);
        assert!((haiku.rates.cache_read - 0.1).abs() < 1e-12);
        assert!((haiku.rates.output - 5.0).abs() < 1e-12);
        assert_eq!(haiku.confidence, Confidence::Measured);
    }

    /// A snapshot takes precedence over the baked table while it is fresh, and
    /// is still used once stale — but stops claiming to be a measurement.
    #[test]
    fn stale_snapshot_is_used_but_downgraded() {
        let row = Priced {
            rates: Rates {
                input: 2.0,
                cache_write: 2.5,
                cache_read: 0.2,
                output: 10.0,
            },
            confidence: Confidence::Measured,
        };
        let mut prices = Prices {
            fetched: HashMap::from([("claude-opus-5".to_string(), row)]),
            fetched_at: unix_now(),
            ..Prices::default()
        };
        let usage = Usage {
            uncached_input: 1_000_000,
            ..Usage::default()
        };
        let fresh = prices.spend("claude-opus-5", &usage);
        assert!((fresh.usd - 2.0).abs() < 1e-12, "got {}", fresh.usd);
        assert_eq!(fresh.confidence, Confidence::Measured);

        prices.fetched_at = unix_now() - STALE_AFTER - 1;
        let stale = prices.spend("claude-opus-5", &usage);
        assert!((stale.usd - 2.0).abs() < 1e-12);
        assert_eq!(stale.confidence, Confidence::Estimated);
    }

    /// The four counters are what a provider reports; the two derived numbers
    /// the rest of the harness reads must stay what they always were.
    #[test]
    fn usage_totals_keep_their_old_meanings() {
        let usage = Usage {
            uncached_input: 100,
            cache_write: 200,
            cache_read: 4_000,
            output: 50,
        };
        assert_eq!(usage.prompt(), 4_300);
        // billable still excludes cache reads — it is a budget gauge, not cost.
        assert_eq!(usage.billable(), 350);
    }
}
