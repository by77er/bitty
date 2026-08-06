//! The actor system: a registry of processes, plus spawn/send/list/stop — the
//! same operations exposed to each agent as tools.
//!
//! Processes can be spawned one at a time, or as a *topology*: a group wired
//! together at birth, each with its own role, its own starting context, and an
//! allowlist of peers it may message. Permissions are resolved from symbolic
//! names to process ids once the whole group's ids are known.

use crate::api::{self, Confidence, Spend, Usage};
use crate::durable::{Event, Journal, MailArtifactRecord, MailRecord, NoJournal, ProcessRecord};
use crate::grants::{Capability, Grant, Grants, PathGrant};
use crate::ui::{self, Tag};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio::task::JoinHandle;

/// Upper bound on one topology, so a runaway plan can't fork-bomb the system.
pub const MAX_GROUP: usize = 16;

/// How many stopped processes stay in the registry. Tombstones are not dead
/// weight — they are what makes "proc-7 has been stopped" possible instead of
/// "no such process", keep re-stopping idempotent, and let `/graph` show what
/// happened. But they are unbounded without a cap, so keep the recent ones and
/// let the rest go. Ids are never recycled (the counter is monotonic), so
/// reaping can never make a stale reference alias a live process.
pub const MAX_TOMBSTONES: usize = 64;

/// Accepted `effort` values, validated at spawn so a typo fails there rather
/// than as a 400 on the process's first turn.
pub const EFFORT_LEVELS: [&str; 5] = ["low", "medium", "high", "xhigh", "max"];

/// What a spawned process runs at when the spawn names no effort. Deliberately
/// NOT inherited from the spawner: a worker is assumed mechanical until stated
/// otherwise, and higher intelligence is an explicit request — the opposite
/// default quietly runs every leaf at the coordinator's effort, which is the
/// most expensive possible mistake to make silently.
pub const DEFAULT_SPAWN_EFFORT: &str = "low";

/// How urgently a message needs to be read.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Priority {
    /// Wake an idle recipient immediately. The default.
    High,
    /// Never *cause* a turn. Held until the recipient is awake for some other
    /// reason, then delivered alongside whatever woke it. Costs nothing beyond
    /// the tokens of the message itself.
    Low,
}

/// A free-form message sitting in a process's mailbox.
pub struct Mail {
    /// Position in the recipient's mailbox log, assigned on delivery.
    pub seq: u64,
    pub from: String,
    pub from_name: Option<String>,
    pub body: String,
    pub priority: Priority,
    /// Long bodies are stored out of context; this identifies the recipient's
    /// durable copy and states its original character count.
    pub artifact_id: Option<String>,
    pub artifact_chars: Option<usize>,
    /// Set when the sender is blocked waiting for an answer. Whatever the
    /// recipient produces is routed back to that specific waiting caller.
    pub reply_to: Option<String>,
}

impl Mail {
    /// Harness-generated mail (exit signals, console input) is always urgent.
    pub fn system(from: &str, body: String) -> Mail {
        Mail {
            from: from.into(),
            from_name: None,
            body,
            priority: Priority::High,
            artifact_id: None,
            artifact_chars: None,
            reply_to: None,
            seq: 0,
        }
    }

    pub fn from_record(record: MailRecord) -> Mail {
        Mail {
            seq: record.seq,
            from: record.from,
            from_name: record.from_name,
            body: record.body,
            priority: if record.low_priority {
                Priority::Low
            } else {
                Priority::High
            },
            artifact_id: record.artifact_id,
            artifact_chars: record.artifact_chars,
            reply_to: record.reply_to,
        }
    }
}

/// Out-of-band instructions to a running script process, kept off the mailbox
/// so they can never be confused with the free-form messages it handles.
pub enum Control {
    /// Replace the running code, keeping the process's identity and mailbox.
    Replace(String),
}

/// A caller blocked inside `call_process`, waiting on one reply.
struct PendingCall {
    caller: String,
    target: String,
    tx: tokio::sync::oneshot::Sender<Result<String, String>>,
}

/// What a process actually runs. Both kinds are full actors — same mailbox,
/// links, grants and namespace — they differ only in what decides their
/// behavior. A script costs no API tokens.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub enum Kind {
    Agent,
    /// TypeScript source, run on an embedded Deno runtime.
    Script(String),
}

impl Kind {
    fn label(&self, model: &str, effort: &Option<String>) -> String {
        match self {
            Kind::Script(_) => "script".into(),
            Kind::Agent => match effort {
                Some(effort) => format!("{model}/{effort}"),
                None => model.to_string(),
            },
        }
    }
}

/// A process that died or stalled, and why — the payload of an exit signal.
struct Exit {
    id: String,
    label: String,
    reason: String,
    /// False for a stall: the process is idle and can still be woken.
    terminal: bool,
}

/// How a new process starts out: what it's told to do, who it is, how much of
/// the spawner's conversation it can see, and who it may talk to.
pub struct NodeSpec {
    pub instructions: String,
    pub name: Option<String>,
    /// Extra system-prompt text describing this process's role. Composed with
    /// the harness scaffolding, never replacing it.
    pub persona: Option<String>,
    /// A rendered snapshot of the spawner's conversation, seeded into the
    /// child's first user turn. `None` = the child starts with a clean slate.
    pub inherited: Option<String>,
    /// Requested capabilities, in symbolic form — targets are sibling node
    /// names, `parent`, `self`, `user`, or the id of an already-running
    /// process. Resolved and then attenuated against the spawner's own grants.
    pub wants: GrantSpec,
    /// Link this process to its spawner, as `spawn_link` does: if either exits
    /// abnormally, the other is sent an exit signal. On by default.
    pub link: bool,
    /// Model for this process. `None` inherits the spawner's, so a cheap
    /// worker's own helpers stay cheap.
    pub model: Option<String>,
    /// Reasoning effort. `None` inherits the spawner's.
    pub effort: Option<String>,
    /// Agent by default; a script when TypeScript source is supplied.
    pub kind: Kind,
    /// Tools this process should see that are really calls to other processes.
    pub aliases: Vec<ToolAlias>,
}

/// A named, schema-typed tool that a process sees in its own tool list but
/// which is really a synchronous call to another process. The point is to give
/// a subagent a real contract — a name, a description, validated arguments —
/// instead of asking it to compose free text and hope.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolAlias {
    pub name: String,
    pub description: String,
    /// JSON Schema for the arguments, checked before anything is delivered.
    pub input_schema: serde_json::Value,
    /// Symbolic before resolution, a process id afterward.
    pub target: String,
}

/// A capability request, before name resolution. `None` on a field means
/// "inherit whatever the spawner holds" rather than "deny".
#[derive(Default, Clone)]
pub struct GrantSpec {
    pub send: Option<Vec<String>>,
    pub stop: Option<Vec<String>>,
    pub spawn: Option<bool>,
    pub run: Option<Vec<String>>,
    pub net: Option<Vec<String>>,
    pub env: Option<Vec<String>>,
    pub sys: Option<Vec<String>>,
    pub read: Option<Vec<String>>,
    pub write: Option<Vec<String>>,
}

impl Default for NodeSpec {
    fn default() -> Self {
        NodeSpec {
            instructions: String::new(),
            name: None,
            persona: None,
            inherited: None,
            wants: GrantSpec::default(),
            model: None,
            effort: None,
            kind: Kind::Agent,
            aliases: Vec::new(),
            // Linked by default: an unlinked spawn is the deliberate choice,
            // matching spawn_link being the common case in practice.
            link: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Running,
    Idle,
    Stopped,
}

/// One process as seen by the human dashboard. This is deliberately a value
/// snapshot rather than a set of shared cells: rendering never holds registry
/// or process locks.
#[derive(Clone, Debug)]
pub struct ProcessSnapshot {
    pub id: String,
    pub name: Option<String>,
    pub parent: String,
    pub status: Status,
    /// "script", or the resolved model/effort pair for an agent.
    pub runs: String,
    /// Prompt tokens in this process's most recent turn — this process's
    /// CURRENT context size, not any kind of peak.
    pub tokens: u64,
    /// What this process has cost so far, with the token split behind it and
    /// how much the price table can be trusted.
    pub spend: crate::api::Spend,
}

/// Coherent system state for one dashboard frame.
#[derive(Clone, Debug)]
pub struct SystemSnapshot {
    pub processes: Vec<ProcessSnapshot>,
    pub billable: u64,
    /// Largest live context in the system. A whole-system maximum, so it says
    /// nothing about any one process — for that, read `ProcessSnapshot::tokens`.
    pub peak_context: u64,
    /// The run's total cost, including processes that have since stopped.
    pub spend: crate::api::Spend,
    pub settled: bool,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Status::Running => "running",
            Status::Idle => "idle",
            Status::Stopped => "stopped",
        }
    }
}

struct Entry {
    id: String,
    name: Option<String>,
    parent: String,
    /// Taken when the process stops: a tombstone should not pin a channel.
    sender: Mutex<Option<UnboundedSender<Mail>>>,
    status: Arc<Mutex<Status>>,
    handle: Mutex<Option<JoinHandle<()>>>,
    context_tokens: Arc<AtomicU64>,
    /// Whether this process is linked to its spawner. Links are only ever
    /// parent↔child, so one flag captures both directions.
    linked: bool,
    /// This process's capabilities — also the ceiling for anything it spawns.
    grants: Grants,
    /// Model and effort, inherited by anything this process spawns. Shared
    /// with the running process's `Meta` rather than copied, so switching a
    /// model in flight is seen by the next turn instead of the next restart.
    model: Arc<Mutex<String>>,
    effort: Arc<Mutex<Option<String>>>,
    /// "script", or "model/effort" for an agent — what /graph shows.
    runs: String,
    /// Script processes only: the channel for code replacement.
    control: Mutex<Option<UnboundedSender<Control>>>,
    /// Next position in this process's mailbox log.
    seq: AtomicU64,
    /// Agent mail may be paged to avoid context blow-ups. Scripts consume
    /// bytes directly and therefore keep receiving complete bodies.
    artifact_mail: bool,
}

/// Everything a process knows about itself.
#[derive(Clone)]
pub struct Meta {
    pub id: String,
    pub name: Option<String>,
    pub parent: String,
    pub tag: Tag,
    pub status: Arc<Mutex<Status>>,
    pub persona: Option<String>,
    /// What this process is permitted to do, already resolved and attenuated.
    pub grants: Grants,
    /// Human labels for every id named in `grants`, for prompts and errors.
    pub labels: HashMap<String, String>,
    /// Prompt size of this process's last turn — what compaction watches.
    pub context_tokens: Arc<AtomicU64>,
    /// Resolved aliases, rendered into this process's tool list.
    pub aliases: Vec<ToolAlias>,
    /// Read per turn, not captured at spawn — see `Entry::model`.
    pub model: Arc<Mutex<String>>,
    pub effort: Arc<Mutex<Option<String>>>,
}

impl Meta {
    /// The model this process should use for its next turn.
    pub fn model(&self) -> String {
        self.model.lock().unwrap().clone()
    }

    pub fn effort(&self) -> Option<String> {
        self.effort.lock().unwrap().clone()
    }

    /// Returns true if this was an actual transition, so callers can log
    /// state changes without narrating every turn.
    pub fn set_status(&self, status: Status) -> bool {
        let mut current = self.status.lock().unwrap();
        let changed = *current != status;
        *current = status;
        changed
    }

    pub fn is_stopped(&self) -> bool {
        *self.status.lock().unwrap() == Status::Stopped
    }

    /// "proc-2 (worker)" or just "proc-2".
    pub fn label(&self) -> String {
        match &self.name {
            Some(name) => format!("{} ({})", self.id, name),
            None => self.id.clone(),
        }
    }

    /// The grant-only permission check. Stop (and anything that needs stop's
    /// authority, like patch_script) should go through `System::may` instead
    /// — a static grant can't cover a process's own future children, which
    /// this alone knows nothing about.
    pub fn may(&self, cap: Capability, target: &str) -> bool {
        self.grants.get(cap).permits(target)
    }

    /// Turn an id into "proc-3 (editor)" using whatever labels this process
    /// was given, falling back to the bare id.
    pub fn label_of(&self, id: &str) -> String {
        match self.labels.get(id) {
            Some(label) => format!("{id} ({label})"),
            None => id.to_string(),
        }
    }

    /// Human-readable list of what a capability permits, for rejections.
    pub fn permitted(&self, cap: Capability) -> String {
        match self.grants.get(cap) {
            Grant::All => "any process".into(),
            Grant::Nobody => "no one".into(),
            Grant::Ids(ids) => {
                let mut named: Vec<String> = ids.iter().map(|id| self.label_of(id)).collect();
                named.sort();
                named.join(", ")
            }
        }
    }

    /// Everything a capability names, for expanding `"*"`.
    pub fn granted_ids(&self, cap: Capability) -> Option<Vec<String>> {
        self.grants.get(cap).ids().map(|ids| {
            let mut out: Vec<String> = ids.iter().cloned().collect();
            out.sort();
            out
        })
    }
}

pub struct System {
    /// The main runtime. Scripts run on dedicated threads (V8 isolates are not
    /// Send), so spawning must name a runtime explicitly rather than relying
    /// on an ambient one.
    rt: tokio::runtime::Handle,
    procs: Mutex<Vec<Entry>>,
    counter: AtomicU64,
    pub api: api::Client,
    /// Latch so a system-wide quiesce is announced once, not once per process.
    quiesce_announced: AtomicBool,
    /// Held for the whole of a spawn so ids can be handed back if validation
    /// fails. Without it two concurrent spawns could reserve overlapping ids.
    spawning: Mutex<()>,
    /// Processes whose model changed since their last turn.
    switched: Mutex<HashSet<String>>,
    /// In-flight synchronous calls, keyed by correlation id.
    pending: Mutex<HashMap<String, PendingCall>>,
    calls: AtomicU64,
    /// Live cache of long mail. Durable journals mirror these records; the
    /// cache keeps no-journal and current-session reads equally cheap.
    mail_artifacts: Mutex<HashMap<String, MailArtifactRecord>>,
    /// Recent send times per (sender, recipient), for the flood limit.
    flood: Mutex<Flood>,
    /// Where each process's life is recorded, so it can be brought back.
    pub journal: Arc<dyn Journal>,
}

/// The longest message one process may put in another's mailbox. Everything
/// past the cap is truncated with a note: a mailbox is for coordination, and
/// a payload this size belongs in a file or a session value, delivered by
/// reference. Agent recipients additionally page bodies above
/// `INLINE_MAIL_CHARS`, so the bounded complete copy does not land verbatim in
/// their context.
pub const MAX_MAIL_CHARS: usize = 32_000;
/// Bodies above this size are replaced in an agent's prompt by a preview and
/// a recipient-scoped handle. This mirrors `run_script`'s large-result limit.
pub const INLINE_MAIL_CHARS: usize = 8_000;
const MAIL_PREVIEW_CHARS: usize = 2_000;
const MAIL_PAGE_CHARS: usize = 8_000;

/// Per-pair flood limit: this many messages in the window, then sends fail
/// until the window drains. Prime-agent uses a comparable token bucket for
/// the same reason — a looping sender must hit a wall, not an inbox.
const FLOOD_LIMIT: usize = 30;
const FLOOD_WINDOW: std::time::Duration = std::time::Duration::from_secs(10);

/// Sliding-window send counter per (sender, recipient) pair. Pure so it can
/// be tested without a running system.
#[derive(Default)]
struct Flood {
    recent: HashMap<(String, String), std::collections::VecDeque<std::time::Instant>>,
}

impl Flood {
    /// Record an attempt at `now`; false when the pair is over its budget.
    fn allow(&mut self, from: &str, to: &str, now: std::time::Instant) -> bool {
        let key = (from.to_string(), to.to_string());
        let window = self.recent.entry(key).or_default();
        while window
            .front()
            .is_some_and(|t| now.duration_since(*t) > FLOOD_WINDOW)
        {
            window.pop_front();
        }
        if window.len() >= FLOOD_LIMIT {
            return false;
        }
        window.push_back(now);
        true
    }
}

/// Enforce the mail body cap: text past the limit is dropped, with a note
/// saying so and naming the better channel.
fn cap_body(body: String) -> String {
    if body.chars().count() <= MAX_MAIL_CHARS {
        return body;
    }
    let total = body.chars().count();
    let kept: String = body.chars().take(MAX_MAIL_CHARS).collect();
    format!(
        "{kept}\n[truncated by the harness: this message was {total} characters and the mailbox \
         cap is {MAX_MAIL_CHARS}. A payload this size should be written to a file (or kept in a \
         session value) and sent by reference.]"
    )
}

fn artifact_preview(body: &str, id: &str, chars: usize) -> String {
    let preview: String = body.chars().take(MAIL_PREVIEW_CHARS).collect();
    format!(
        "{preview}\n\n[long message stored as mailbox artifact {id}: {chars} characters. Use the \
         mailbox tool with action=read and id={id:?} to page it without loading all of it into \
         context.]"
    )
}

impl System {
    pub fn new(api: api::Client) -> Self {
        System {
            rt: tokio::runtime::Handle::current(),
            procs: Mutex::new(Vec::new()),
            counter: AtomicU64::new(0),
            api,
            quiesce_announced: AtomicBool::new(false),
            flood: Mutex::new(Flood::default()),
            spawning: Mutex::new(()),
            switched: Mutex::new(HashSet::new()),
            pending: Mutex::new(HashMap::new()),
            calls: AtomicU64::new(0),
            mail_artifacts: Mutex::new(HashMap::new()),
            journal: Arc::new(NoJournal),
        }
    }

    pub fn with_journal(mut self, journal: Arc<dyn Journal>) -> Self {
        self.journal = journal;
        self
    }

    fn remember_mail_artifact(&self, artifact: MailArtifactRecord) {
        self.journal.store_mail_artifact(&artifact);
        self.mail_artifacts
            .lock()
            .unwrap()
            .insert(artifact.id.clone(), artifact);
    }

    fn store_long_mail(
        &self,
        recipient: &str,
        from: &str,
        id: String,
        body: &str,
    ) -> Option<(String, usize, String)> {
        let chars = body.chars().count();
        if chars <= INLINE_MAIL_CHARS {
            return None;
        }
        self.remember_mail_artifact(MailArtifactRecord {
            id: id.clone(),
            recipient: recipient.to_string(),
            from: from.to_string(),
            chars,
            body: body.to_string(),
        });
        Some((id.clone(), chars, artifact_preview(body, &id, chars)))
    }

    /// Call ids restart with the harness, while their artifacts outlive it.
    /// Preserve an older handle rather than silently changing what it points
    /// at when a post-resume call reuses the same correlation id.
    fn unused_mail_artifact_id(&self, recipient: &str, base: String) -> String {
        if self.mail_artifact(recipient, &base).is_none() {
            return base;
        }
        for suffix in 2u64.. {
            let candidate = format!("{base}-{suffix}");
            if self.mail_artifact(recipient, &candidate).is_none() {
                return candidate;
            }
        }
        unreachable!("an unbounded artifact suffix search must find a free id")
    }

    fn uses_artifact_mail(&self, process: &str) -> bool {
        self.procs
            .lock()
            .unwrap()
            .iter()
            .find(|entry| entry.id == process)
            .is_some_and(|entry| entry.artifact_mail)
    }

    fn restore_mail(&self, recipient: &str, record: MailRecord, artifact_mail: bool) -> Mail {
        let mut mail = Mail::from_record(record);
        if artifact_mail {
            let id = mail
                .artifact_id
                .clone()
                .unwrap_or_else(|| format!("mail-{recipient}-{}", mail.seq));
            if let Some((id, chars, preview)) =
                self.store_long_mail(recipient, &mail.from, id, &mail.body)
            {
                mail.body = preview;
                mail.artifact_id = Some(id);
                mail.artifact_chars = Some(chars);
            } else {
                mail.artifact_id = None;
                mail.artifact_chars = None;
            }
        }
        mail
    }

    fn mail_artifact(&self, recipient: &str, id: &str) -> Option<MailArtifactRecord> {
        if let Some(artifact) = self.mail_artifacts.lock().unwrap().get(id).cloned() {
            return (artifact.recipient == recipient).then_some(artifact);
        }
        let artifact = self.journal.read_mail_artifact(recipient, id)?;
        self.mail_artifacts
            .lock()
            .unwrap()
            .insert(id.to_string(), artifact.clone());
        Some(artifact)
    }

    /// Metadata only: listing a mailbox must not itself pull every long body
    /// into the model's context.
    pub fn list_mail_artifacts(&self, recipient: &str) -> String {
        let mut artifacts: HashMap<String, MailArtifactRecord> = self
            .journal
            .list_mail_artifacts(recipient)
            .into_iter()
            .map(|artifact| (artifact.id.clone(), artifact))
            .collect();
        for artifact in self.mail_artifacts.lock().unwrap().values() {
            if artifact.recipient == recipient {
                artifacts.insert(artifact.id.clone(), artifact.clone());
            }
        }
        let mut artifacts: Vec<MailArtifactRecord> = artifacts.into_values().collect();
        artifacts.sort_by(|left, right| left.id.cmp(&right.id));
        if artifacts.is_empty() {
            return "No stored mailbox artifacts.".into();
        }
        artifacts
            .into_iter()
            .map(|artifact| {
                format!(
                    "{} · {} chars · from {}",
                    artifact.id, artifact.chars, artifact.from
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn read_mail_artifact(
        &self,
        recipient: &str,
        id: &str,
        offset: usize,
        limit: usize,
    ) -> Result<String, String> {
        let Some(artifact) = self.mail_artifact(recipient, id) else {
            return Err(format!(
                "No mailbox artifact '{id}' belongs to {recipient}."
            ));
        };
        let offset = offset.min(artifact.chars);
        let limit = limit.clamp(1, MAIL_PAGE_CHARS);
        let page: String = artifact.body.chars().skip(offset).take(limit).collect();
        let end = offset + page.chars().count();
        Ok(format!(
            "mailbox artifact {id} · from {} · characters {offset}..{end} of {}\n{page}",
            artifact.from, artifact.chars
        ))
    }

    pub fn discard_mail_artifact(&self, recipient: &str, id: &str) -> Result<String, String> {
        let removed_memory = {
            let mut artifacts = self.mail_artifacts.lock().unwrap();
            if artifacts
                .get(id)
                .is_some_and(|artifact| artifact.recipient == recipient)
            {
                artifacts.remove(id);
                true
            } else {
                false
            }
        };
        let removed_disk = self.journal.discard_mail_artifact(recipient, id);
        if removed_memory || removed_disk {
            Ok(format!("Discarded mailbox artifact {id}."))
        } else {
            Err(format!(
                "No mailbox artifact '{id}' belongs to {recipient}."
            ))
        }
    }

    /// Resume ids above everything already used, so a restored process never
    /// collides with a new one.
    pub fn resume_ids_after(&self, highest: u64) {
        self.counter.fetch_max(highest, Ordering::Relaxed);
    }

    /// Rebuild a process from its journal: same id, same grants, same links,
    /// and for an agent the conversation it had. A script restarts from its
    /// source with an empty heap — a V8 isolate cannot be serialized, which
    /// matches what patch_script already promises.
    pub fn restore(
        self: &Arc<Self>,
        record: ProcessRecord,
        history: Vec<serde_json::Value>,
        pending: Vec<MailRecord>,
        mail_cursor: u64,
    ) {
        let (sender, receiver) = mpsc::unbounded_channel::<Mail>();
        let (control_tx, control_rx) = mpsc::unbounded_channel::<Control>();
        let status = Arc::new(Mutex::new(Status::Running));
        let context_tokens = Arc::new(AtomicU64::new(0));
        let artifact_mail = matches!(&record.kind, Kind::Agent);
        // Everything the log says was never consumed is owed to this process,
        // in the order it originally arrived.
        for mail in pending {
            let _ = sender.send(self.restore_mail(&record.id, mail, artifact_mail));
        }
        let label = match &record.name {
            Some(name) => format!("{} {name}", record.id),
            None => record.id.clone(),
        };

        let model_cell = Arc::new(Mutex::new(record.model.clone()));
        let effort_cell = Arc::new(Mutex::new(record.effort.clone()));
        self.procs.lock().unwrap().push(Entry {
            id: record.id.clone(),
            name: record.name.clone(),
            parent: record.parent.clone(),
            sender: Mutex::new(Some(sender)),
            status: status.clone(),
            handle: Mutex::new(None),
            context_tokens: context_tokens.clone(),
            linked: record.linked,
            grants: record.grants.clone(),
            model: model_cell.clone(),
            effort: effort_cell.clone(),
            runs: record.kind.label(&record.model, &record.effort),
            control: Mutex::new(match record.kind {
                Kind::Script(_) => Some(control_tx),
                Kind::Agent => None,
            }),
            seq: AtomicU64::new(mail_cursor),
            artifact_mail,
        });

        let meta = Meta {
            id: record.id.clone(),
            name: record.name.clone(),
            parent: record.parent.clone(),
            tag: Tag::new(label, record.ordinal),
            status,
            persona: record.persona.clone(),
            grants: record.grants.clone(),
            labels: HashMap::new(),
            context_tokens,
            aliases: record.aliases.clone(),
            model: model_cell.clone(),
            effort: effort_cell.clone(),
        };

        let id = record.id.clone();
        let handle = match record.kind.clone() {
            Kind::Agent => self.rt.spawn(crate::agent::resume(
                self.clone(),
                meta,
                receiver,
                record.instructions,
                record.inherited,
                history,
            )),
            Kind::Script(source) => self.rt.spawn(crate::script::run(
                self.clone(),
                meta,
                receiver,
                control_rx,
                record.instructions,
                source,
                true,
            )),
        };
        if let Some(entry) = self.procs.lock().unwrap().iter().find(|p| p.id == id) {
            *entry.handle.lock().unwrap() = Some(handle);
        }
    }

    /// Record that a process has folded everything up to `seq` into a turn, and
    /// push the log to disk. Call it *after* recording that turn's `Input`,
    /// never before: this flushes, so a cursor advance made durable ahead of the
    /// message body it accounts for loses the message outright — `restore` reads
    /// it as already consumed and never redelivers it. In this order a crash in
    /// the window costs a duplicate delivery, which is the at-least-once
    /// semantic `send` promises.
    pub fn note_consumed(&self, process: &str, seq: u64) {
        if seq > 0 {
            self.journal
                .record(process, &Event::Consumed { through: seq });
        }
        // The turn is the durability boundary: everything buffered since the
        // last one lands in a single write before the process acts on it.
        self.journal.flush(process);
    }

    /// The main runtime, for work that must not run on a script's own thread.
    pub fn rt(&self) -> &tokio::runtime::Handle {
        &self.rt
    }

    /// Open a slot for a synchronous call and return its correlation id.
    pub fn register_call(
        &self,
        caller: &str,
        target: &str,
    ) -> (
        String,
        tokio::sync::oneshot::Receiver<Result<String, String>>,
    ) {
        let id = format!("call-{}", self.calls.fetch_add(1, Ordering::Relaxed) + 1);
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.pending.lock().unwrap().insert(
            id.clone(),
            PendingCall {
                caller: caller.to_string(),
                target: target.to_string(),
                tx,
            },
        );
        (id, rx)
    }

    /// Hand a reply back to whoever is blocked on `id`. A correlation id that
    /// is no longer pending (the caller timed out, or replied twice) is simply
    /// dropped rather than treated as an error.
    pub fn resolve_call(&self, id: &str, value: Result<String, String>) {
        if let Some(pending) = self.pending.lock().unwrap().remove(id) {
            let value = match value {
                Ok(body) if self.uses_artifact_mail(&pending.caller) => {
                    let artifact_id = self.unused_mail_artifact_id(
                        &pending.caller,
                        format!("mail-{}-{id}", pending.caller),
                    );
                    let shown = self
                        .store_long_mail(&pending.caller, &pending.target, artifact_id, &body)
                        .map(|(_, _, preview)| preview)
                        .unwrap_or(body);
                    if pending.caller != pending.target {
                        ui::arrival(&pending.caller, &pending.target, &shown);
                    }
                    Ok(shown)
                }
                Ok(body) => {
                    if pending.caller != pending.target {
                        ui::arrival(&pending.caller, &pending.target, &body);
                    }
                    Ok(body)
                }
                Err(error) => Err(error),
            };
            let _ = pending.tx.send(value);
        }
    }

    pub fn call_is_pending(&self, id: &str) -> bool {
        self.pending.lock().unwrap().contains_key(id)
    }

    /// The call `caller` is blocked on, waiting for `target` to answer.
    ///
    /// A process that answers with a plain message instead of in_reply_to is
    /// clearly still answering, and the caller would otherwise block for its
    /// whole timeout while the reply sat unread in a mailbox it cannot check.
    pub fn call_awaiting(&self, caller: &str, target: &str) -> Option<String> {
        self.pending
            .lock()
            .unwrap()
            .iter()
            .find(|(_, call)| call.caller == caller && call.target == target)
            .map(|(id, _)| id.clone())
    }

    /// Fail every call waiting on a process that has just stopped, so a caller
    /// never blocks on an answer that can no longer come.
    fn fail_calls_to(&self, targets: &[String], reason: &str) {
        let doomed: Vec<String> = {
            let pending = self.pending.lock().unwrap();
            pending
                .iter()
                .filter(|(_, call)| targets.contains(&call.target))
                .map(|(id, _)| id.clone())
                .collect()
        };
        for id in doomed {
            self.resolve_call(&id, Err(reason.to_string()));
        }
    }

    /// Replace the code of a running script, keeping its id, mailbox, links
    /// and grants. Agents have no code to replace.
    pub fn patch_script(&self, target: &str, source: String) -> Result<String, String> {
        let procs = self.procs.lock().unwrap();
        let Some(entry) = procs.iter().find(|p| p.id == target) else {
            return Err(format!("No process with id '{target}'."));
        };
        if *entry.status.lock().unwrap() == Status::Stopped {
            return Err(format!("Process {target} has been stopped."));
        }
        let control = entry.control.lock().unwrap();
        let sent = match control.as_ref() {
            None => Err(format!(
                "{target} is an agent, not a script — there is no code to replace. Change what an \
                 agent does by messaging it."
            )),
            Some(tx) => tx
                .send(Control::Replace(source.clone()))
                .map_err(|_| format!("Process {target} is no longer running.")),
        };
        drop(control);
        drop(procs);
        sent?;
        // Without this the replacement lives only in the isolate that is
        // running it, and a restart brings back the process's first draft —
        // every patch since then silently gone.
        self.journal.record(target, &Event::Patched { source });
        self.journal.flush(target);
        Ok(format!(
            "Replaced the code running in {target}. Its id, mailbox, links and permissions are \
             unchanged; any state the old code held is gone."
        ))
    }

    /// Spawn one process. Convenience wrapper over `spawn_group`.
    ///
    /// This can fail like any other spawn — a single node is still subject to
    /// the capability ceiling — so the result must be handled, not unwrapped.
    pub fn spawn(self: &Arc<Self>, parent: &str, spec: NodeSpec) -> Result<String, String> {
        Ok(self.spawn_group(parent, vec![spec])?[0].1.clone())
    }

    /// Drop the oldest tombstones beyond the retention cap. Called at spawn
    /// time — the only moment the registry grows — rather than at stop time,
    /// which would race the exit-signal lookup that still needs the entry.
    fn reap_tombstones(&self) {
        let mut procs = self.procs.lock().unwrap();
        let stopped: Vec<bool> = procs
            .iter()
            .map(|p| *p.status.lock().unwrap() == Status::Stopped)
            .collect();
        let doomed: HashSet<usize> = tombstones_to_drop(&stopped, MAX_TOMBSTONES)
            .into_iter()
            .collect();
        if doomed.is_empty() {
            return;
        }
        let mut index = 0;
        procs.retain(|_| {
            let keep = !doomed.contains(&index);
            index += 1;
            keep
        });
        ui::system(&format!(
            "reaped {} tombstone(s); {MAX_TOMBSTONES} most recent retained",
            doomed.len()
        ));
    }

    /// Spawn a wired group of processes. Ids are allocated for the whole group
    /// first, so nodes can reference each other by name; then permissions are
    /// resolved and every task is launched.
    ///
    /// Returns (name, id) pairs in input order.
    pub fn spawn_group(
        self: &Arc<Self>,
        parent: &str,
        nodes: Vec<NodeSpec>,
    ) -> Result<Vec<(String, String)>, String> {
        if nodes.is_empty() {
            return Err("No processes specified.".into());
        }
        if nodes.len() > MAX_GROUP {
            return Err(format!(
                "Too many processes: {} requested, limit is {MAX_GROUP} per topology.",
                nodes.len()
            ));
        }
        self.reap_tombstones();

        // Everything that can reject a spawn runs before any id is claimed, so
        // a refused spawn leaves no gap in the id sequence and no half-state.
        for node in &nodes {
            if let Kind::Script(source) = &node.kind {
                crate::script::precheck(node.name.as_deref().unwrap_or("script"), source)?;
            }
        }
        for node in &nodes {
            if let Some(model) = &node.model
                && crate::api::Tier::parse(model).is_none()
            {
                return Err(format!(
                    "'{model}' is not a model tier. Processes are sized, not named — use \
                     {}, or omit the field to inherit yours.",
                    crate::api::Tier::NAMES.join(", ")
                ));
            }
            if let Some(effort) = &node.effort
                && !EFFORT_LEVELS.contains(&effort.as_str())
            {
                return Err(format!(
                    "Unknown effort '{effort}'. Use one of: {}.",
                    EFFORT_LEVELS.join(", ")
                ));
            }
        }

        // Phase 1 — reserve ids so nodes can reference each other by name.
        // Reserved, not consumed: later phases can still reject the spawn, and
        // a refused spawn must not leave a hole in the sequence. The lock makes
        // the reserve-validate-commit sequence atomic against other spawns.
        let _reserving = self.spawning.lock().unwrap();
        let base = self.counter.load(Ordering::Relaxed);
        let mut ids = Vec::with_capacity(nodes.len());
        let mut ordinals = Vec::with_capacity(nodes.len());
        for offset in 1..=nodes.len() as u64 {
            ids.push(format!("proc-{}", base + offset));
            ordinals.push(base + offset);
        }
        let by_name: Vec<(Option<&str>, &str)> = nodes
            .iter()
            .zip(&ids)
            .map(|(node, id)| (node.name.as_deref(), id.as_str()))
            .collect();

        // The spawner's grants are the ceiling, and its model the default,
        // for everything below it. Effort is deliberately not inherited —
        // see DEFAULT_SPAWN_EFFORT.
        let (ceiling, inherited_model, existing) = {
            let procs = self.procs.lock().unwrap();
            let me = procs.iter().find(|p| p.id == parent);
            let ceiling = me
                .map(|p| p.grants.clone())
                .unwrap_or_else(Grants::console_authority); // parent == "user"
            let model = me
                .map(|p| p.model.lock().unwrap().clone())
                .unwrap_or_else(|| self.api.model.clone());
            let existing: Vec<(String, Option<String>)> = procs
                .iter()
                .map(|p| (p.id.clone(), p.name.clone()))
                .collect();
            (ceiling, model, existing)
        };
        if !ceiling.spawn.is_permissive() {
            return Err(format!(
                "Not permitted: {parent} does not hold the spawn capability."
            ));
        }
        // Phase 2 — resolve symbolic targets to ids, rejecting typos before
        // anything starts running, then attenuate against the ceiling.
        let mut resolved: Vec<(Grants, HashMap<String, String>, Vec<ToolAlias>)> = Vec::new();
        for (node, self_id) in nodes.iter().zip(&ids) {
            let mut labels: HashMap<String, String> = HashMap::new();

            // Resolve one symbolic target: a sibling in this batch, a keyword,
            // or the id of a process that is already running. Cross-batch ids
            // are what let a later group be wired to an earlier one.
            let mut resolve = |target: &String| -> Result<String, String> {
                match target.as_str() {
                    "user" => {
                        labels.insert("user".into(), "the human console".into());
                        Ok("user".into())
                    }
                    "parent" => {
                        labels.insert(parent.to_string(), "your spawner".into());
                        Ok(parent.to_string())
                    }
                    "self" => {
                        labels.insert(self_id.clone(), "yourself".into());
                        Ok(self_id.clone())
                    }
                    peer => {
                        if let Some((_, id)) = by_name.iter().find(|(name, _)| *name == Some(peer))
                        {
                            labels.insert(id.to_string(), peer.to_string());
                            return Ok(id.to_string());
                        }
                        if let Some((id, name)) = existing.iter().find(|(id, _)| id == peer) {
                            if let Some(name) = name {
                                labels.insert(id.clone(), name.clone());
                            }
                            return Ok(id.clone());
                        }
                        Err(format!(
                            "Unknown target '{peer}'. Valid targets are names from this spawn \
                             group, the id of a running process, or 'parent', 'self', 'user'."
                        ))
                    }
                }
            };

            let as_grant = |spec: &Option<Vec<String>>,
                            resolve: &mut dyn FnMut(&String) -> Result<String, String>|
             -> Result<Option<Grant>, String> {
                let Some(targets) = spec else { return Ok(None) };
                if targets.is_empty() {
                    return Ok(Some(Grant::Nobody));
                }
                let mut ids = HashSet::new();
                for target in targets {
                    ids.insert(resolve(target)?);
                }
                Ok(Some(Grant::Ids(ids)))
            };

            let send_req = as_grant(&node.wants.send, &mut resolve)?;
            let stop_req = as_grant(&node.wants.stop, &mut resolve)?;

            // An *explicit* request for authority the spawner lacks is an
            // error, not something to quietly trim: a coordinator that wires a
            // worker to a peer it cannot reach has a broken plan, and should
            // find out now rather than watch the worker fail to deliver later.
            // An *omitted* field is not a request — it inherits, and clamping
            // there is just the default doing its job.
            let describe_ceiling = |grant: &Grant| match grant {
                Grant::All => "anyone".to_string(),
                Grant::Nobody => "no one".to_string(),
                Grant::Ids(ids) => {
                    let mut named: Vec<String> = ids.iter().cloned().collect();
                    named.sort();
                    named.join(", ")
                }
            };
            let checked = |requested: Option<Grant>,
                           ceiling_grant: &Grant,
                           always: &[String],
                           gerund: &str,
                           verb: &str|
             -> Result<Grant, String> {
                let Some(wanted) = requested else {
                    // Inheriting still has to carry the invariants, or a
                    // process can end up unable to stop itself.
                    return Ok(ceiling_grant.clone().with(always));
                };
                let granted = wanted.clone().attenuate(ceiling_grant).with(always);
                let dropped = granted.dropped_from(&wanted);
                if !dropped.is_empty() {
                    return Err(format!(
                        "Cannot grant '{}' {gerund} access to {}: you may only {verb} {}. A \
                         process can never be granted more authority than the process that \
                         spawns it.",
                        node.name.as_deref().unwrap_or("the new process"),
                        dropped.join(", "),
                        describe_ceiling(ceiling_grant),
                    ));
                }
                Ok(granted)
            };

            // A child may always be granted permission to message the process
            // that spawned it — that's structural wiring, not authority handed
            // away, so it must not be checked against the spawner's own
            // ceiling.send (who *it* may message, an unrelated question from
            // whether its children may reach it). Widened into the ceiling
            // rather than force-added after attenuation, so an explicit `[]`
            // (isolate it entirely) still denies it — only Nobody survives
            // attenuation against Nobody regardless of what the ceiling holds.
            let send_ceiling = ceiling.send.clone().with(&[parent.to_string()]);
            let send = checked(send_req.clone(), &send_ceiling, &[], "messaging", "message")?;
            // Stop authority is granted deliberately, never drifted into.
            // Inherit it only from a spawner that holds it over everything;
            // otherwise a process defaults to stopping just itself, so a
            // grandchild never quietly acquires the power to kill its parent.
            let stop_default = if send_req.is_some() || !matches!(ceiling.stop, Grant::All) {
                Some(Grant::Ids(HashSet::from([self_id.clone()])))
            } else {
                None
            };
            let stop = checked(
                stop_req.or(stop_default),
                &ceiling.stop,
                std::slice::from_ref(self_id),
                "stopping",
                "stop",
            )?;
            let spawn = match node.wants.spawn {
                Some(true) if !ceiling.spawn.is_permissive() => {
                    return Err(format!(
                        "Cannot grant the spawn capability to '{}': you do not hold it yourself.",
                        node.name.as_deref().unwrap_or("the new process")
                    ));
                }
                Some(false) => Grant::Nobody,
                Some(true) | None => ceiling.spawn.clone(),
            };

            // Filesystem roots are canonicalized here, so a request is checked
            // against real directories rather than the text of a path.
            let paths = |requested: &Option<Vec<String>>,
                         ceiling_grant: &PathGrant,
                         label: &str|
             -> Result<PathGrant, String> {
                let Some(roots) = requested else {
                    return Ok(ceiling_grant.clone());
                };
                if roots.is_empty() {
                    return Ok(PathGrant::Nowhere);
                }
                let mut canonical = Vec::new();
                for root in roots {
                    match std::path::Path::new(root).canonicalize() {
                        Ok(path) => canonical.push(path),
                        Err(e) => {
                            return Err(format!("Cannot grant {label} on '{root}': {e}"));
                        }
                    }
                }
                let wanted = PathGrant::Under(canonical);
                let granted = wanted.clone().attenuate(ceiling_grant);
                let dropped = granted.dropped_from(&wanted);
                if !dropped.is_empty() {
                    return Err(format!(
                        "Cannot grant '{}' {label} on {}: your own {label} covers only {}. A \
                         process can never be granted more authority than the process that \
                         spawns it.",
                        node.name.as_deref().unwrap_or("the new process"),
                        dropped.join(", "),
                        ceiling_grant.describe(),
                    ));
                }
                Ok(granted)
            };
            // Programs, hosts, variables and system keys are all plain names,
            // so they need no resolution — only the same attenuation check.
            let mut verbatim = |name: &String| Ok(name.clone());
            let run = checked(
                as_grant(&node.wants.run, &mut verbatim)?,
                &ceiling.run,
                &[],
                "running",
                "run",
            )?;
            let net = checked(
                as_grant(&node.wants.net, &mut verbatim)?,
                &ceiling.net,
                &[],
                "network access to",
                "reach",
            )?;
            let env = checked(
                as_grant(&node.wants.env, &mut verbatim)?,
                &ceiling.env,
                &[],
                "environment access to",
                "read",
            )?;
            let sys = checked(
                as_grant(&node.wants.sys, &mut verbatim)?,
                &ceiling.sys,
                &[],
                "system info",
                "query",
            )?;
            let read = paths(&node.wants.read, &ceiling.read, "read access")?;
            let write = paths(&node.wants.write, &ceiling.write, "write access")?;

            let grants = Grants {
                send,
                stop,
                spawn,
                run,
                net,
                env,
                sys,
                read,
                write,
            };

            // An alias *is* a grant: authority to reach exactly one process
            // with exactly one shape of argument, and nothing else. So it is
            // bounded by the spawner's own reach rather than by the child's —
            // a process with no messaging at all can still hold tools, which
            // is the whole point of handing out a tool instead of a graph.
            //
            // What it may not do is launder authority upward. A spawner may
            // point an alias at anything it could message itself, or at a
            // process it is creating in this very spawn, and nothing else.
            let mut aliases = Vec::new();
            for alias in &node.aliases {
                let target = resolve(&alias.target)?;
                if !ceiling.send.permits(&target) && !ids.contains(&target) {
                    return Err(format!(
                        "Cannot give '{}' the tool '{}': it points at {target}, which you are \
                         not permitted to message. An alias cannot grant reach that you do not \
                         have yourself.",
                        node.name.as_deref().unwrap_or("the new process"),
                        alias.name,
                    ));
                }
                aliases.push(ToolAlias {
                    target,
                    ..alias.clone()
                });
            }
            labels
                .entry(parent.to_string())
                .or_insert_with(|| "your spawner".into());
            labels
                .entry(self_id.clone())
                .or_insert_with(|| "yourself".into());
            resolved.push((grants, labels, aliases));
        }

        // Phase 3 — nothing can refuse the spawn from here, so the ids are
        // finally consumed.
        self.counter
            .store(base + nodes.len() as u64, Ordering::Relaxed);
        let mut launched = Vec::with_capacity(nodes.len());
        for (((node, id), n), (grants, labels, aliases)) in
            nodes.into_iter().zip(ids).zip(ordinals).zip(resolved)
        {
            let label = match &node.name {
                Some(name) => format!("{id} {name}"),
                None => id.clone(),
            };
            let (sender, receiver) = mpsc::unbounded_channel::<Mail>();
            let (control_tx, control_rx) = mpsc::unbounded_channel::<Control>();
            let status = Arc::new(Mutex::new(Status::Running));
            let context_tokens = Arc::new(AtomicU64::new(0));
            // One cell, held by the registry and the running process alike.
            let spawn_model = Arc::new(Mutex::new(
                node.model
                    .clone()
                    .unwrap_or_else(|| inherited_model.clone()),
            ));
            let effort = Some(
                node.effort
                    .clone()
                    .unwrap_or_else(|| DEFAULT_SPAWN_EFFORT.to_string()),
            );
            let spawn_effort = Arc::new(Mutex::new(effort.clone()));

            self.procs.lock().unwrap().push(Entry {
                id: id.clone(),
                name: node.name.clone(),
                parent: parent.to_string(),
                sender: Mutex::new(Some(sender)),
                status: status.clone(),
                handle: Mutex::new(None),
                context_tokens: context_tokens.clone(),
                linked: node.link,
                grants: grants.clone(),
                model: spawn_model.clone(),
                effort: spawn_effort.clone(),
                control: Mutex::new(match node.kind {
                    Kind::Script(_) => Some(control_tx),
                    Kind::Agent => None,
                }),
                seq: AtomicU64::new(0),
                runs: node
                    .kind
                    .label(node.model.as_deref().unwrap_or(&inherited_model), &effort),
                artifact_mail: matches!(&node.kind, Kind::Agent),
            });

            if self.journal.enabled() {
                self.journal.record(
                    &id,
                    &Event::Spawned(ProcessRecord {
                        id: id.clone(),
                        name: node.name.clone(),
                        parent: parent.to_string(),
                        persona: node.persona.clone(),
                        instructions: node.instructions.clone(),
                        inherited: node.inherited.clone(),
                        grants: grants.clone(),
                        aliases: aliases.clone(),
                        model: node
                            .model
                            .clone()
                            .unwrap_or_else(|| inherited_model.clone()),
                        effort: effort.clone(),
                        linked: node.link,
                        kind: node.kind.clone(),
                        ordinal: n,
                    }),
                );
            }
            let meta = Meta {
                id: id.clone(),
                name: node.name.clone(),
                parent: parent.to_string(),
                tag: Tag::new(label, n),
                status,
                persona: node.persona,
                grants,
                labels,
                context_tokens,
                aliases,
                model: spawn_model.clone(),
                effort: spawn_effort.clone(),
            };

            // Initial instructions are a process's first inbound message even
            // though they are passed directly to its runner rather than
            // queued. Put them on the same observable delivery stream.
            if !node.instructions.trim().is_empty() {
                ui::arrival(&id, parent, &node.instructions);
            }

            let handle = match node.kind.clone() {
                Kind::Agent => self.rt.spawn(crate::agent::run(
                    self.clone(),
                    meta,
                    receiver,
                    node.instructions,
                    node.inherited,
                )),
                Kind::Script(source) => self.rt.spawn(crate::script::run(
                    self.clone(),
                    meta,
                    receiver,
                    control_rx,
                    node.instructions,
                    source,
                    false,
                )),
            };
            if let Some(entry) = self.procs.lock().unwrap().iter().find(|p| p.id == id) {
                *entry.handle.lock().unwrap() = Some(handle);
            }
            launched.push((node.name.unwrap_or_else(|| id.clone()), id));
        }
        Ok(launched)
    }

    /// Deliver mail to a process by id. The special id "user" prints straight
    /// to the human's console.
    pub fn send(&self, to: &str, mut mail: Mail) -> Result<String, String> {
        if to == "user" {
            let label = match &mail.from_name {
                Some(name) => format!("{} {}", mail.from, name),
                None => mail.from.clone(),
            };
            ui::mail_to_user(&label, &mail.body);
            return Ok("Delivered to the user's console.".into());
        }
        // The harness and the human are exempt from both limits: exit signals
        // and console input are never the flood, and truncating them would
        // hide the very information they exist to deliver.
        if mail.from != "system" && mail.from != "user" {
            if !self
                .flood
                .lock()
                .unwrap()
                .allow(&mail.from, to, std::time::Instant::now())
            {
                return Err(format!(
                    "Rate limited: you have sent {to} more than {FLOOD_LIMIT} messages in \
                     {FLOOD_WINDOW:?}. Batch your updates into fewer, denser messages."
                ));
            }
            mail.body = cap_body(mail.body);
        }
        let procs = self.procs.lock().unwrap();
        let Some(entry) = procs.iter().find(|process| process.id == to) else {
            return Err(format!(
                "No process with id '{to}'. Use list_processes to see valid ids."
            ));
        };
        if *entry.status.lock().unwrap() == Status::Stopped {
            return Err(format!(
                "Process {to} has been stopped; it cannot receive mail."
            ));
        }

        // Recorded before delivery, so a crash between the two costs a
        // duplicate rather than a lost message — at-least-once, which is the
        // semantic a cursor-based queue can actually honor.
        mail.seq = entry.seq.fetch_add(1, Ordering::Relaxed) + 1;
        let complete_body = mail.body.clone();
        if entry.artifact_mail {
            let artifact_id = format!("mail-{to}-{}", mail.seq);
            if let Some((id, chars, preview)) =
                self.store_long_mail(to, &mail.from, artifact_id, &complete_body)
            {
                mail.body = preview;
                mail.artifact_id = Some(id);
                mail.artifact_chars = Some(chars);
            }
        }
        self.journal.record(
            to,
            &Event::Enqueued(MailRecord {
                seq: mail.seq,
                from: mail.from.clone(),
                from_name: mail.from_name.clone(),
                artifact_id: mail.artifact_id.clone(),
                artifact_chars: mail.artifact_chars,
                body: complete_body,
                low_priority: mail.priority == Priority::Low,
                reply_to: mail.reply_to.clone(),
            }),
        );
        // One write per message, which is exactly what the "recorded before
        // delivery... at-least-once" comment above already assumes is
        // happening. Without this, low-priority mail in particular can sit
        // unflushed indefinitely: agent.rs moves it into the in-RAM
        // `deferred` list without waking the process, so both the sender's
        // and recipient's copies stay volatile for as long as it idles.
        self.journal.flush(to);

        let from = match &mail.from_name {
            Some(name) => format!("{} {name}", mail.from),
            None => mail.from.clone(),
        };
        let shown = mail.body.clone();
        match entry.sender.lock().unwrap().as_ref() {
            Some(sender) => match sender.send(mail) {
                Ok(()) => {
                    ui::arrival(to, &from, &shown);
                    Ok(format!("Delivered to {to}'s mailbox."))
                }
                Err(_) => Err(format!("Process {to} is no longer running.")),
            },
            None => Err(format!("Process {to} is no longer running.")),
        }
    }

    /// Stop a process (and, with `cascade`, every descendant it spawned).
    /// Stopping is permanent: the task is aborted mid-await, the entry stays
    /// listed as "stopped", and mail to it errors. A process may stop itself.
    ///
    /// `initiator` is the process that asked, or `None` for the human console.
    /// A process stopping *itself* is a graceful exit and stays quiet; every
    /// other death is abnormal and notifies its neighbors, because otherwise
    /// anyone waiting on it waits forever.
    /// Switch a process's model and effort while it runs. Takes effect on its
    /// next turn, because the loop reads both per turn rather than caching
    /// them at spawn.
    pub fn set_model(&self, id: &str, model: &str, effort: Option<&str>) -> Result<String, String> {
        // Same validation as spawn, so a typo fails here rather than as a 400
        // on the process's next turn.
        let Some(tier) = crate::api::Tier::parse(model) else {
            return Err(format!(
                "'{model}' is not a model tier — use one of {}.",
                crate::api::Tier::NAMES.join(", ")
            ));
        };
        let model = tier.as_str();
        if let Some(effort) = effort
            && !EFFORT_LEVELS.contains(&effort)
        {
            return Err(format!(
                "'{effort}' is not an effort level; use one of {}.",
                EFFORT_LEVELS.join(", ")
            ));
        }
        let mut procs = self.procs.lock().unwrap();
        let Some(entry) = procs.iter_mut().find(|p| p.id == id) else {
            return Err(format!("No such process: {id}."));
        };
        if matches!(*entry.status.lock().unwrap(), Status::Stopped) {
            return Err(format!("{id} has been stopped."));
        }
        if entry.control.lock().unwrap().is_some() {
            return Err(format!("{id} is a script — it runs code, not a model."));
        }
        let was = entry.model.lock().unwrap().clone();
        *entry.model.lock().unwrap() = model.to_string();
        if let Some(effort) = effort {
            *entry.effort.lock().unwrap() = Some(effort.to_string());
        }
        // What /ps and /graph show is a rendered string, so it has to be
        // re-rendered or it would keep reporting the model this process no
        // longer uses.
        let now_effort = entry.effort.lock().unwrap().clone();
        entry.runs = match &now_effort {
            Some(effort) => format!("{model}/{effort}"),
            None => model.to_string(),
        };
        drop(procs);
        self.journal.record(
            id,
            &Event::Retuned {
                model: model.to_string(),
                effort: effort.map(String::from),
            },
        );
        self.journal.flush(id);
        // A switch invalidates the prompt cache for this process, and the
        // thinking blocks in its history were signed by the old model. Both
        // are the caller's to act on, so say so rather than leaving it to be
        // discovered as a failed turn.
        self.switched.lock().unwrap().insert(id.to_string());
        Ok(format!(
            "{id}: {was} → {model}{}. Takes effect on its next turn; its prompt cache starts \
             cold, and prior thinking blocks are dropped because their signatures belong to the \
             old model.",
            effort
                .map(|e| format!(" at {e} effort"))
                .unwrap_or_default()
        ))
    }

    /// Whether this process just changed model and therefore needs its history
    /// cleaned before the next request. Consumed once.
    pub fn take_switched(&self, id: &str) -> bool {
        self.switched.lock().unwrap().remove(id)
    }

    pub fn stop(
        &self,
        targets: &[String],
        cascade: bool,
        initiator: Option<&str>,
    ) -> Result<String, String> {
        let mut abnormal: Vec<Exit> = Vec::new();
        let gone: Vec<String>;
        let summary;
        {
            let procs = self.procs.lock().unwrap();

            // Partition the request before touching anything, so one bad id
            // doesn't abort the stops that are perfectly valid.
            let mut selected: HashSet<String> = HashSet::new();
            let mut unknown: Vec<String> = Vec::new();
            for target in targets {
                if target == "user" {
                    unknown.push("user (the human console cannot be stopped)".into());
                } else if procs.iter().any(|p| &p.id == target) {
                    selected.insert(target.clone());
                } else {
                    unknown.push(target.clone());
                }
            }
            if selected.is_empty() {
                return Err(format!(
                    "Nothing stopped. Unknown: {}. Use list_processes to see valid ids.",
                    unknown.join(", ")
                ));
            }

            // Add the transitive children of every target.
            if cascade {
                loop {
                    let mut grew = false;
                    for p in procs.iter() {
                        if selected.contains(&p.parent) && !selected.contains(&p.id) {
                            selected.insert(p.id.clone());
                            grew = true;
                        }
                    }
                    if !grew {
                        break;
                    }
                }
            }

            let mut stopped: Vec<String> = Vec::new();
            let mut already: Vec<String> = Vec::new();
            for p in procs.iter().filter(|p| selected.contains(&p.id)) {
                {
                    let mut status = p.status.lock().unwrap();
                    if *status == Status::Stopped {
                        already.push(p.id.clone());
                        continue;
                    }
                    *status = Status::Stopped;
                }
                if let Some(handle) = p.handle.lock().unwrap().take() {
                    handle.abort();
                }
                // Neither is usable again; a tombstone keeps only its metadata.
                p.sender.lock().unwrap().take();
                ui::system(&format!(
                    "■ stopped {}{}",
                    p.id,
                    p.name
                        .as_ref()
                        .map(|n| format!(" ({n})"))
                        .unwrap_or_default()
                ));
                self.journal.record(
                    &p.id,
                    &Event::Stopped {
                        reason: match initiator {
                            Some(by) => format!("stopped by {by}"),
                            None => "stopped from the console".into(),
                        },
                    },
                );
                // A tombstone needs its identity and the fact that it stopped;
                // its conversation and mailbox are dead weight from here on.
                // Compacting now bounds the cost of a graph that churns.
                self.journal.compact(&p.id);
                if initiator != Some(p.id.as_str()) {
                    let reason = match initiator {
                        _ if !targets.iter().any(|t| t == &p.id) => {
                            "stopped as a descendant of a process that was stopped (cascade)"
                                .to_string()
                        }
                        Some(by) => format!("stopped by {by}"),
                        None => "stopped by the user from the console".to_string(),
                    };
                    abnormal.push(Exit {
                        id: p.id.clone(),
                        label: describe(p),
                        reason,
                        terminal: true,
                    });
                }
                stopped.push(p.id.clone());
            }

            let mut parts = Vec::new();
            if !stopped.is_empty() {
                parts.push(format!("Stopped: {}.", stopped.join(", ")));
            }
            if !already.is_empty() {
                parts.push(format!("Already stopped: {}.", already.join(", ")));
            }
            if !unknown.is_empty() {
                parts.push(format!("Unknown: {}.", unknown.join(", ")));
            }
            gone = stopped.clone();
            summary = parts.join(" ");
        } // registry lock released before any delivery

        // Anything blocked on a process that just died must be released.
        if !gone.is_empty() {
            self.fail_calls_to(&gone, "the process stopped before it replied");
        }
        self.signal_exits(&abnormal);
        Ok(summary)
    }

    /// Deliver exit signals along **links**, the way `spawn_link` does: a link
    /// is established deliberately when a process is spawned, and joins that
    /// process to its spawner in both directions. Nothing is inferred from the
    /// communication graph — a process that merely talks to another is not
    /// told when it dies. That is the supervisor's job: it holds the link,
    /// learns of the death, and decides what to relay.
    ///
    /// Unlike OTP's default, an exit signal never kills the linked process; it
    /// arrives as mail, so the recipient decides what to do. Every process
    /// here effectively traps exits.
    fn signal_exits(&self, exits: &[Exit]) {
        if exits.is_empty() {
            return;
        }
        let subjects: HashSet<&str> = exits.iter().map(|e| e.id.as_str()).collect();

        // recipient -> the exits it is linked to and should hear about
        let mut mailbag: Vec<(String, Vec<&Exit>)> = Vec::new();
        {
            let procs = self.procs.lock().unwrap();
            for exit in exits {
                let Some(entry) = procs.iter().find(|p| p.id == exit.id) else {
                    continue;
                };
                let mut linked: Vec<String> = Vec::new();
                // A link joins a process to its spawner in both directions.
                if entry.linked && entry.parent != "user" {
                    linked.push(entry.parent.clone());
                }
                for child in procs.iter() {
                    if child.parent == exit.id && child.linked {
                        linked.push(child.id.clone());
                    }
                }
                for to in linked {
                    if subjects.contains(to.as_str()) {
                        continue;
                    }
                    let live = procs
                        .iter()
                        .any(|p| p.id == to && *p.status.lock().unwrap() != Status::Stopped);
                    if !live {
                        continue;
                    }
                    match mailbag.iter_mut().find(|(id, _)| id == &to) {
                        Some((_, list)) => list.push(exit),
                        None => mailbag.push((to, vec![exit])),
                    }
                }
            }
        } // lock released before delivery

        for (to, exits) in mailbag {
            let detail = exits
                .iter()
                .map(|e| format!("- {} — {}", e.label, e.reason))
                .collect::<Vec<_>>()
                .join("\n");
            let terminal = exits.iter().any(|e| e.terminal);
            let guidance = if terminal {
                "Anything listed as stopped can no longer receive messages or reply. If you were \
                 waiting on it, stop waiting and decide: re-plan, do the work yourself, or spawn a \
                 replacement. If other processes you coordinate were depending on it, telling them \
                 is your job — they are not linked to it and have not been notified."
            } else {
                "It is idle rather than dead, so a message will wake it and it can retry. Decide \
                 whether to retry, reassign the work, or replace it — and tell anything you \
                 coordinate that was depending on it."
            };
            let body = format!(
                "<exit_signal>\n{detail}\n\nYou are linked to it, which is why you were told. \
                 {guidance}\n</exit_signal>"
            );
            let _ = self.send(&to, Mail::system("system", body));
        }
    }

    /// A process that is alive but has given up on its current task. Reported
    /// along the same links as an exit, since a supervisor waiting on it needs
    /// to know just as much.
    pub fn signal_stalled(&self, id: &str, label: &str) {
        self.signal_exits(&[Exit {
            id: id.to_string(),
            label: label.to_string(),
            reason: "stalled after repeated API failures".to_string(),
            terminal: false,
        }]);
    }

    /// Every live process id except `except`. The console's view — global,
    /// because the human is outside the namespace.
    pub fn live_ids(&self, except: &str) -> Vec<String> {
        self.procs
            .lock()
            .unwrap()
            .iter()
            .filter(|p| p.id != except && *p.status.lock().unwrap() != Status::Stopped)
            .map(|p| p.id.clone())
            .collect()
    }

    /// What a process can observe: itself, everything it spawned
    /// (transitively), its spawner, and anything named in its grants.
    ///
    /// Visibility tracks authority. A process holding `All` over messaging or
    /// stopping may act on anything, so hiding the system from it would be
    /// incoherent — it sees everything. A process confined to an allowlist is
    /// in a namespace: it sees its own subtree and its wiring, and processes
    /// outside that simply do not exist as far as it is concerned. That is why
    /// an out-of-view id reports as unknown rather than as forbidden — the
    /// latter would confirm the existence of something it should not know
    /// about.
    ///
    /// `None` means unrestricted: the caller should not filter at all.
    pub fn visible_to(&self, viewer: &Meta) -> Option<HashSet<String>> {
        let unbounded = [Capability::Send, Capability::Stop]
            .iter()
            .any(|cap| matches!(viewer.grants.get(*cap), Grant::All));
        if unbounded {
            return None;
        }

        let procs = self.procs.lock().unwrap();
        // Expand descendants from self alone. Seeding this with the parent
        // would sweep in every sibling through the shared parent edge.
        let mut seen: HashSet<String> = HashSet::from([viewer.id.clone()]);
        loop {
            let mut grew = false;
            for p in procs.iter() {
                if seen.contains(&p.parent) && !seen.contains(&p.id) {
                    seen.insert(p.id.clone());
                    grew = true;
                }
            }
            if !grew {
                break;
            }
        }
        // The spawner and any granted peer are visible as leaves — their own
        // subtrees are not.
        if viewer.parent != "user" {
            seen.insert(viewer.parent.clone());
        }
        for cap in Capability::ALL {
            if let Some(ids) = viewer.grants.get(cap).ids() {
                seen.extend(ids.iter().cloned());
            }
        }
        Some(seen)
    }

    /// True when `target` exists at all from `viewer`'s vantage point.
    pub fn is_visible(&self, viewer: &Meta, target: &str) -> bool {
        match self.visible_to(viewer) {
            None => true,
            Some(view) => view.contains(target),
        }
    }

    /// True if `target` is `ancestor` itself or descends from it through the
    /// live spawn tree, regardless of any grant.
    fn spawned_by(&self, ancestor: &str, target: &str) -> bool {
        let procs = self.procs.lock().unwrap();
        let mut current = target.to_string();
        loop {
            if current == ancestor {
                return true;
            }
            match procs.iter().find(|p| p.id == current) {
                Some(p) if p.parent != current => current = p.parent.clone(),
                _ => return false,
            }
        }
    }

    /// The permission check for a capability that targets another process.
    /// A static grant alone would leave a process unable to stop or message
    /// anything it spawns after its own grants were fixed — those ids don't
    /// exist yet at that point, so no allowlist could ever have named them.
    /// Stopping (and so, since it is at least as powerful, patching) and
    /// messaging your own descendants is always permitted on top of whatever
    /// the grant itself says.
    pub fn may(&self, viewer: &Meta, cap: Capability, target: &str) -> bool {
        viewer.may(cap, target)
            || (matches!(cap, Capability::Stop | Capability::Send)
                && self.spawned_by(&viewer.id, target))
    }

    /// Announce once when nothing is left running, so the human knows the
    /// system is waiting on them rather than wedged.
    pub fn note_quiesced(&self) {
        if self.all_settled() && !self.quiesce_announced.swap(true, Ordering::Relaxed) {
            let waiting = self
                .procs
                .lock()
                .unwrap()
                .iter()
                .filter(|p| *p.status.lock().unwrap() == Status::Idle)
                .count();
            let noun = if waiting == 1 { "process" } else { "processes" };
            ui::system(&format!(
                "— system idle · {waiting} {noun} waiting for a message"
            ));
        }
    }

    pub fn note_running(&self) {
        self.quiesce_announced.store(false, Ordering::Relaxed);
    }

    /// The console's global listing.
    pub fn list(&self) -> String {
        self.list_filtered(None)
    }

    /// What a process sees when it calls `list_processes`.
    pub fn list_for(&self, viewer: &Meta) -> String {
        let view = self.visible_to(viewer);
        self.list_filtered(view.as_ref())
    }

    fn list_filtered(&self, view: Option<&HashSet<String>>) -> String {
        // Built from the same snapshot the dashboard renders, so /ps and the
        // TUI can never disagree about what a process cost.
        let snapshot = self.snapshot();
        let mut out =
            String::from("id       name           status   context  cache  cost      parent\n");
        let mut caveat = false;
        for p in snapshot
            .processes
            .iter()
            .filter(|p| view.is_none_or(|v| v.contains(&p.id)))
        {
            caveat |= p.spend.confidence != Confidence::Measured;
            out.push_str(&format!(
                "{:<8} {:<14} {:<8} {:<8} {:<6} {:<9} {}\n",
                p.id,
                p.name.as_deref().unwrap_or("-"),
                p.status.as_str(),
                format_tokens(p.tokens),
                cache_share(&p.spend.usage),
                format_cost(&p.spend),
                p.parent,
            ));
        }
        out.push_str(&format!(
            "total {} ({} prices)\n",
            format_cost(&snapshot.spend),
            snapshot.spend.confidence.as_str()
        ));
        if caveat {
            out.push_str(
                "(~ priced from the baked-in table, ? no price for that model — see api.rs)\n",
            );
        }
        if !self.api.compaction_enabled() {
            out.push_str("(compaction is off — contexts grow unbounded)\n");
        }
        out
    }

    /// The process graph: the supervision tree (who spawned whom, and which
    /// of those edges are links) annotated with each process's capabilities.
    /// Both relationships matter and they are not the same — the tree shows
    /// who gets told when something dies, the `sends→` column shows who may
    /// talk to whom.
    pub fn graph(&self) -> String {
        let procs = self.procs.lock().unwrap();
        if procs.is_empty() {
            return "(no processes)\n".into();
        }

        // Anything whose parent isn't a live registry entry roots its own tree,
        // so a process is never hidden by an unknown or detached parent.
        let mut rows: Vec<Row> = Vec::new();
        let roots: Vec<&Entry> = procs
            .iter()
            .filter(|p| !procs.iter().any(|q| q.id == p.parent))
            .collect();
        for (i, root) in roots.iter().enumerate() {
            walk(&procs, root, "", i + 1 == roots.len(), true, &mut rows);
        }

        let width = rows
            .iter()
            .map(|r| r.tree.chars().count())
            .max()
            .unwrap_or(0);
        let mut out = String::from(
            "process graph — tree = who spawned whom, ⚯ = linked, sends→ = may message\n",
        );
        for row in rows {
            let pad = " ".repeat(width - row.tree.chars().count());
            out.push_str(&format!(
                "{}{pad}  {:<8} {:>6}  {}\n",
                row.tree, row.status, row.tokens, row.notes
            ));
        }
        out
    }

    /// Take the dashboard's complete view under one registry lock. In
    /// particular, do not call `all_settled` or `peak_context` here: both would
    /// try to acquire `procs` a second time and deadlock this snapshot.
    pub fn snapshot(&self) -> SystemSnapshot {
        let procs = self.procs.lock().unwrap();
        let mut settled = true;
        let mut peak_context = 0;
        let mut processes = Vec::with_capacity(procs.len());
        for process in procs.iter() {
            let status = *process.status.lock().unwrap();
            let tokens = process.context_tokens.load(Ordering::Relaxed);
            settled &= status != Status::Running;
            peak_context = peak_context.max(tokens);
            processes.push(ProcessSnapshot {
                id: process.id.clone(),
                name: process.name.clone(),
                parent: process.parent.clone(),
                status,
                runs: process.runs.clone(),
                tokens,
                spend: self.api.spend_for(&process.id),
            });
        }
        SystemSnapshot {
            processes,
            billable: self.api.billable_spent(),
            peak_context,
            spend: self.api.spend_total(),
            settled,
        }
    }

    /// Largest live context in the system, for the status line.
    pub fn peak_context(&self) -> u64 {
        self.procs
            .lock()
            .unwrap()
            .iter()
            .map(|p| p.context_tokens.load(Ordering::Relaxed))
            .max()
            .unwrap_or(0)
    }

    /// True when no process is running (all idle or stopped) — used by --once.
    pub fn all_settled(&self) -> bool {
        let procs = self.procs.lock().unwrap();
        procs
            .iter()
            .all(|p| *p.status.lock().unwrap() != Status::Running)
    }
}

/// Positions of the stopped entries to drop, keeping the `keep` most recent.
/// The registry is append-ordered, so a lower index is an older process.
fn tombstones_to_drop(stopped: &[bool], keep: usize) -> Vec<usize> {
    let positions: Vec<usize> = stopped
        .iter()
        .enumerate()
        .filter(|(_, is_stopped)| **is_stopped)
        .map(|(index, _)| index)
        .collect();
    if positions.len() <= keep {
        return Vec::new();
    }
    positions[..positions.len() - keep].to_vec()
}

struct Row {
    tree: String,
    status: &'static str,
    tokens: String,
    notes: String,
}

/// Render one process and then its children, drawing the usual box tree.
fn walk(procs: &[Entry], entry: &Entry, prefix: &str, last: bool, root: bool, rows: &mut Vec<Row>) {
    let connector = if root {
        String::new()
    } else {
        format!(
            "{prefix}{}{} ",
            if last { "└" } else { "├" },
            if entry.linked { "⚯" } else { "─" }
        )
    };
    let name = entry.name.as_deref().unwrap_or("-");

    let mut notes = Vec::new();
    match &entry.grants.send {
        Grant::All => {}
        Grant::Nobody => notes.push("sends→ no one".to_string()),
        Grant::Ids(ids) => {
            let mut targets: Vec<String> = ids.iter().cloned().collect();
            targets.sort();
            notes.push(format!("sends→ {}", targets.join(" ")));
        }
    }
    match &entry.grants.stop {
        Grant::All => {}
        Grant::Nobody => notes.push("stops→ no one".into()),
        Grant::Ids(ids) if ids.len() == 1 && ids.contains(&entry.id) => {
            notes.push("stops→ self".into())
        }
        Grant::Ids(ids) => {
            let mut targets: Vec<String> = ids.iter().cloned().collect();
            targets.sort();
            notes.push(format!("stops→ {}", targets.join(" ")));
        }
    }
    if !entry.grants.spawn.is_permissive() {
        notes.push("no-spawn".into());
    }
    // Filesystem reach is the authority that leaves the harness, so it is the
    // one most worth being able to audit at a glance.
    if entry.grants.run.is_permissive() {
        notes.push(format!(
            "runs→ {}",
            match &entry.grants.run {
                Grant::All => "any".to_string(),
                Grant::Ids(names) => {
                    let mut n: Vec<String> = names.iter().cloned().collect();
                    n.sort();
                    n.join(" ")
                }
                Grant::Nobody => String::new(),
            }
        ));
    }
    if entry.grants.read.is_permissive() {
        notes.push(format!("reads→ {}", entry.grants.read.describe()));
    }
    if entry.grants.write.is_permissive() {
        notes.push(format!("writes→ {}", entry.grants.write.describe()));
    }
    notes.push(entry.runs.clone());

    rows.push(Row {
        tree: format!("{connector}{} {name}", entry.id),
        status: entry.status.lock().unwrap().as_str(),
        tokens: format_tokens(entry.context_tokens.load(Ordering::Relaxed)),
        notes: notes.join("  "),
    });

    let children: Vec<&Entry> = procs.iter().filter(|p| p.parent == entry.id).collect();
    let child_prefix = if root {
        String::new()
    } else {
        format!("{prefix}{}  ", if last { " " } else { "│" })
    };
    for (i, child) in children.iter().enumerate() {
        walk(
            procs,
            child,
            &child_prefix,
            i + 1 == children.len(),
            false,
            rows,
        );
    }
}

/// "proc-2 (worker)" or just "proc-2".
fn describe(entry: &Entry) -> String {
    match &entry.name {
        Some(name) => format!("{} ({})", entry.id, name),
        None => entry.id.clone(),
    }
}

fn format_tokens(n: u64) -> String {
    match n {
        0 => "-".into(),
        n if n < 1_000 => format!("{n}"),
        n => format!("{}k", n / 1_000),
    }
}

/// A cost for a text row, carrying its own caveat: `~` when the rates behind
/// it came from the baked-in table, `?` when some model had no price at all
/// and the figure is therefore incomplete. A bare figure means measured.
fn format_cost(spend: &Spend) -> String {
    let mark = match spend.confidence {
        Confidence::Measured => "",
        Confidence::Estimated => "~",
        Confidence::Unknown => "?",
    };
    format!("{mark}${:.4}", spend.usd)
}

/// How much of a process's prompt tokens were cheap cache hits.
fn cache_share(usage: &Usage) -> String {
    match usage.prompt() {
        0 => "-".into(),
        prompt => format!("{}%", usage.cache_read * 100 / prompt),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flood limit is per pair and slides: a burst hits the wall, other
    /// pairs are unaffected, and time drains the window.
    #[test]
    fn flood_limit_is_per_pair_and_slides() {
        let mut flood = Flood::default();
        let start = std::time::Instant::now();
        for _ in 0..FLOOD_LIMIT {
            assert!(flood.allow("proc-2", "proc-3", start));
        }
        assert!(!flood.allow("proc-2", "proc-3", start), "over budget");
        // A different pair has its own budget.
        assert!(flood.allow("proc-2", "proc-4", start));
        assert!(flood.allow("proc-4", "proc-3", start));
        // And the window drains with time.
        assert!(flood.allow(
            "proc-2",
            "proc-3",
            start + FLOOD_WINDOW + std::time::Duration::from_secs(1)
        ));
    }

    /// A capped body keeps its head and says what happened; a small one is
    /// untouched.
    #[test]
    fn mail_bodies_are_capped() {
        assert_eq!(cap_body("hello".into()), "hello");
        let capped = cap_body("x".repeat(MAX_MAIL_CHARS + 5_000));
        assert!(capped.contains("truncated by the harness"));
        assert!(capped.chars().count() < MAX_MAIL_CHARS + 300);
    }

    #[test]
    fn keeps_everything_under_the_cap() {
        assert!(tombstones_to_drop(&[true, true, false], 4).is_empty());
    }

    #[test]
    fn drops_oldest_first_and_never_the_living() {
        // Positions 0,2,4 are tombstones; keeping 1 must drop the older two
        // and leave every running process untouched.
        let stopped = [true, false, true, false, true];
        assert_eq!(tombstones_to_drop(&stopped, 1), vec![0, 2]);
    }

    #[test]
    fn a_cap_of_zero_drops_every_tombstone() {
        assert_eq!(tombstones_to_drop(&[true, false, true], 0), vec![0, 2]);
    }

    #[test]
    fn all_living_is_a_no_op() {
        assert!(tombstones_to_drop(&[false, false], 0).is_empty());
    }

    #[tokio::test]
    async fn snapshot_is_coherent_and_carries_dashboard_fields() {
        let sys = System::new(dummy_client());
        let running = test_entry("proc-1", "user", Grant::Nobody);
        let mut idle = test_entry("proc-2", "proc-1", Grant::Nobody);
        idle.name = Some("worker".into());
        *idle.status.lock().unwrap() = Status::Idle;
        idle.context_tokens.store(12_345, Ordering::Relaxed);
        *sys.procs.lock().unwrap() = vec![running, idle];

        let snapshot = sys.snapshot();
        assert_eq!(snapshot.processes.len(), 2);
        assert_eq!(snapshot.processes[1].name.as_deref(), Some("worker"));
        assert_eq!(snapshot.processes[1].parent, "proc-1");
        assert_eq!(snapshot.processes[1].status, Status::Idle);
        assert_eq!(snapshot.processes[1].runs, "small");
        assert_eq!(snapshot.processes[1].tokens, 12_345);
        assert_eq!(snapshot.peak_context, 12_345);
        assert!(!snapshot.settled);

        *sys.procs.lock().unwrap()[0].status.lock().unwrap() = Status::Stopped;
        assert!(sys.snapshot().settled);
    }

    /// The context figure is per process — the status line's old `peak_context`
    /// is a whole-system maximum and says nothing about any one of them. Cost
    /// lands on the process that spent it, and on the run total, and nowhere
    /// else.
    #[tokio::test]
    async fn snapshot_context_and_cost_are_per_process() {
        let sys = System::new(dummy_client());
        let small = test_entry("proc-1", "user", Grant::Nobody);
        small.context_tokens.store(1_000, Ordering::Relaxed);
        let big = test_entry("proc-2", "proc-1", Grant::Nobody);
        big.context_tokens.store(90_000, Ordering::Relaxed);
        *sys.procs.lock().unwrap() = vec![small, big];

        let snapshot = sys.snapshot();
        assert_eq!(snapshot.processes[0].tokens, 1_000);
        assert_eq!(snapshot.processes[1].tokens, 90_000);
        assert_eq!(snapshot.peak_context, 90_000);
        // Nothing has called a model yet, so every figure is a measured zero.
        assert_eq!(snapshot.spend.usd, 0.0);
        assert_eq!(snapshot.spend.confidence, Confidence::Measured);

        sys.api.charge(
            "proc-2",
            "large",
            &Usage {
                uncached_input: 1_000,
                cache_read: 20_000,
                output: 100,
                ..Usage::default()
            },
        );
        let snapshot = sys.snapshot();
        assert_eq!(snapshot.processes[0].spend.usd, 0.0);
        assert!(snapshot.processes[1].spend.usd > 0.0);
        assert_eq!(snapshot.spend.usd, snapshot.processes[1].spend.usd);
        assert_eq!(snapshot.processes[1].spend.usage.cache_read, 20_000);
        // Contexts are untouched by charging: the gauge is the last request's
        // prompt, not a running total.
        assert_eq!(snapshot.processes[1].tokens, 90_000);
    }

    fn test_entry(id: &str, parent: &str, stop: Grant) -> Entry {
        Entry {
            id: id.to_string(),
            name: None,
            parent: parent.to_string(),
            sender: Mutex::new(None),
            status: Arc::new(Mutex::new(Status::Running)),
            handle: Mutex::new(None),
            context_tokens: Arc::new(AtomicU64::new(0)),
            linked: true,
            grants: Grants {
                send: Grant::Nobody,
                stop,
                spawn: Grant::Nobody,
                run: Grant::Nobody,
                net: Grant::Nobody,
                env: Grant::Nobody,
                sys: Grant::Nobody,
                read: PathGrant::Nowhere,
                write: PathGrant::Nowhere,
            },
            model: Arc::new(Mutex::new("small".to_string())),
            effort: Arc::new(Mutex::new(None)),
            runs: "small".to_string(),
            control: Mutex::new(None),
            seq: AtomicU64::new(0),
            artifact_mail: true,
        }
    }

    fn test_meta_for(entry: &Entry) -> Meta {
        Meta {
            id: entry.id.clone(),
            name: entry.name.clone(),
            parent: entry.parent.clone(),
            tag: Tag::new(&entry.id, 0),
            status: entry.status.clone(),
            persona: None,
            grants: entry.grants.clone(),
            labels: HashMap::new(),
            context_tokens: entry.context_tokens.clone(),
            aliases: Vec::new(),
            model: entry.model.clone(),
            effort: entry.effort.clone(),
        }
    }

    // SAFETY (test-only): no other test in this binary reads or writes
    // ANTHROPIC_API_KEY, and Client::from_env only reads it — no network call.
    fn dummy_client() -> api::Client {
        unsafe { std::env::set_var("ANTHROPIC_API_KEY", "sk-ant-test-not-real") };
        api::Client::from_env().unwrap()
    }

    #[tokio::test]
    async fn long_agent_mail_is_paged_and_recipient_scoped() {
        let sys = System::new(dummy_client());
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let mut agent = test_entry("proc-1", "user", Grant::Nobody);
        agent.sender = Mutex::new(Some(sender));
        agent.artifact_mail = true;
        *sys.procs.lock().unwrap() = vec![agent];

        let body = "α".repeat(INLINE_MAIL_CHARS + 20);
        sys.send("proc-1", Mail::system("user", body.clone()))
            .unwrap();
        let delivered = receiver.recv().await.unwrap();
        let id = delivered
            .artifact_id
            .clone()
            .expect("long mail gets a handle");
        assert_eq!(delivered.artifact_chars, Some(INLINE_MAIL_CHARS + 20));
        assert!(delivered.body.contains(&id));
        assert!(delivered.body.chars().count() < body.chars().count());
        assert!(sys.list_mail_artifacts("proc-1").contains(&id));

        let tail = sys
            .read_mail_artifact("proc-1", &id, INLINE_MAIL_CHARS, 100)
            .unwrap();
        assert!(tail.ends_with(&"α".repeat(20)));
        assert!(sys.read_mail_artifact("proc-2", &id, 0, 10).is_err());
        assert!(sys.discard_mail_artifact("proc-2", &id).is_err());
        assert!(sys.read_mail_artifact("proc-1", &id, 0, 10).is_ok());
        sys.discard_mail_artifact("proc-1", &id).unwrap();
        assert!(sys.read_mail_artifact("proc-1", &id, 0, 10).is_err());
    }

    #[tokio::test]
    async fn script_mail_keeps_the_complete_body_inline() {
        let sys = System::new(dummy_client());
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let mut script = test_entry("proc-1", "user", Grant::Nobody);
        script.sender = Mutex::new(Some(sender));
        script.artifact_mail = false;
        *sys.procs.lock().unwrap() = vec![script];

        let body = "x".repeat(INLINE_MAIL_CHARS + 20);
        sys.send("proc-1", Mail::system("user", body.clone()))
            .unwrap();
        let delivered = receiver.recv().await.unwrap();
        assert_eq!(delivered.body, body);
        assert!(delivered.artifact_id.is_none());
    }

    #[tokio::test]
    async fn resumed_call_artifacts_do_not_overwrite_an_old_handle() {
        let sys = System::new(dummy_client());
        *sys.procs.lock().unwrap() = vec![test_entry("proc-1", "user", Grant::Nobody)];
        sys.mail_artifacts.lock().unwrap().insert(
            "mail-proc-1-call-1".into(),
            MailArtifactRecord {
                id: "mail-proc-1-call-1".into(),
                recipient: "proc-1".into(),
                from: "proc-2".into(),
                chars: 3,
                body: "old".into(),
            },
        );

        let (call, response) = sys.register_call("proc-1", "proc-2");
        assert_eq!(call, "call-1");
        sys.resolve_call(&call, Ok("n".repeat(INLINE_MAIL_CHARS + 1)));
        let shown = response.await.unwrap().unwrap();
        assert!(shown.contains("mail-proc-1-call-1-2"));
        assert!(
            sys.read_mail_artifact("proc-1", "mail-proc-1-call-1", 0, 10)
                .unwrap()
                .ends_with("old")
        );
        assert!(
            sys.read_mail_artifact("proc-1", "mail-proc-1-call-1-2", 0, 10)
                .is_ok()
        );
    }

    #[tokio::test]
    async fn a_process_may_always_stop_what_it_spawned_even_without_a_matching_grant() {
        let sys = System::new(dummy_client());
        // proc-1's own stop grant names only itself — resolved long before
        // proc-2 was ever spawned, so it could never have named it.
        let coordinator = test_entry(
            "proc-1",
            "user",
            Grant::Ids(HashSet::from(["proc-1".to_string()])),
        );
        let coordinator_meta = test_meta_for(&coordinator);
        let worker = test_entry(
            "proc-2",
            "proc-1",
            Grant::Ids(HashSet::from(["proc-2".to_string()])),
        );
        let stranger = test_entry("proc-3", "user", Grant::Nobody);
        *sys.procs.lock().unwrap() = vec![coordinator, worker, stranger];

        assert!(sys.may(&coordinator_meta, Capability::Stop, "proc-2"));
        // Not spawned by it and never granted: stays out of reach.
        assert!(!sys.may(&coordinator_meta, Capability::Stop, "proc-3"));
    }

    #[tokio::test]
    async fn the_descendant_bypass_reaches_grandchildren_too() {
        let sys = System::new(dummy_client());
        let coordinator = test_entry(
            "proc-1",
            "user",
            Grant::Ids(HashSet::from(["proc-1".to_string()])),
        );
        let coordinator_meta = test_meta_for(&coordinator);
        let child = test_entry(
            "proc-2",
            "proc-1",
            Grant::Ids(HashSet::from(["proc-2".to_string()])),
        );
        let grandchild = test_entry(
            "proc-3",
            "proc-2",
            Grant::Ids(HashSet::from(["proc-3".to_string()])),
        );
        *sys.procs.lock().unwrap() = vec![coordinator, child, grandchild];

        assert!(sys.may(&coordinator_meta, Capability::Stop, "proc-3"));
    }

    fn journal_record(id: &str) -> ProcessRecord {
        ProcessRecord {
            id: id.into(),
            name: None,
            parent: "proc-1".into(),
            persona: None,
            instructions: "do the thing".into(),
            inherited: None,
            grants: crate::grants::Grants::unrestricted(),
            aliases: Vec::new(),
            model: "claude-opus-5".into(),
            effort: None,
            linked: true,
            kind: Kind::Agent,
            ordinal: 1,
        }
    }

    fn temp_journal_root(name: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "bitty-system-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    /// The invariant edit (1) in agent.rs depends on: `flush` alone must push
    /// a buffered Output to disk, with no trailing Spawned/Stopped event to
    /// trigger `record`'s own auto-flush. Fails if that auto-flush special
    /// case is ever relied on instead of an explicit flush after Output.
    #[test]
    fn an_output_is_durable_once_flushed_with_no_trailing_spawned_or_stopped() {
        let root = temp_journal_root("output-durability");
        let journal = crate::durable::FileJournal::new(&root).unwrap();
        let id = "proc-2";
        journal.record(id, &Event::Spawned(journal_record(id)));
        journal.record(
            id,
            &Event::Output {
                content: serde_json::json!([{"type": "text", "text": "hi"}]),
            },
        );
        let path = root.join(format!("{id}.jsonl"));
        let before = std::fs::read_to_string(&path).unwrap();
        assert!(
            !before.contains("\"Output\""),
            "buffered Output must not already be on disk before an explicit flush"
        );

        journal.flush(id); // exactly the call agent.rs now makes right after recording Output
        let after = std::fs::read_to_string(&path).unwrap();
        let events: Vec<Event> = after
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        assert!(events.iter().any(|e| matches!(e, Event::Output { .. })));
        assert!(!events.iter().any(|e| matches!(e, Event::Stopped { .. })));
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, Event::Spawned(_)))
                .count(),
            1,
            "durable with exactly the one Spawned already on disk, no Stopped at all"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The invariant edits (2a)/(2b)/(3) depend on: recording a turn's Input
    /// before calling `note_consumed` means its flush can never durably
    /// advance the cursor without the message body it accounts for already
    /// being in the same, about-to-be-flushed buffer. Fails if `note_consumed`
    /// is ever called before the Input it accounts for is recorded.
    #[tokio::test]
    async fn consumed_is_never_durable_without_its_input_already_on_disk() {
        let root = temp_journal_root("consumed-ordering");
        let sys = System::new(dummy_client())
            .with_journal(Arc::new(crate::durable::FileJournal::new(&root).unwrap()));
        let id = "proc-2";
        sys.journal.record(id, &Event::Spawned(journal_record(id)));

        // Body first...
        sys.journal.record(
            id,
            &Event::Input {
                content: serde_json::json!([{"type": "text", "text": "mail body"}]),
            },
        );
        let path = root.join(format!("{id}.jsonl"));
        let before = std::fs::read_to_string(&path).unwrap();
        assert!(
            !before.contains("\"Input\""),
            "the Input is still buffered, exactly like a crash landing right here"
        );

        // ...cursor second: note_consumed's flush now carries both, or neither.
        sys.note_consumed(id, 1);
        let after = std::fs::read_to_string(&path).unwrap();
        let input_at = after.find("\"Input\"").expect("Input reached disk");
        let consumed_at = after.find("\"Consumed\"").expect("Consumed reached disk");
        assert!(
            input_at < consumed_at,
            "Input must precede Consumed in the log"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn the_descendant_bypass_also_covers_messaging_what_it_spawned() {
        let sys = System::new(dummy_client());
        // Same story as stop: proc-1's own send grant is empty, resolved
        // before proc-2 existed to be named in it.
        let coordinator = test_entry("proc-1", "user", Grant::Nobody);
        let coordinator_meta = test_meta_for(&coordinator);
        let worker = test_entry("proc-2", "proc-1", Grant::Nobody);
        let stranger = test_entry("proc-3", "user", Grant::Nobody);
        *sys.procs.lock().unwrap() = vec![coordinator, worker, stranger];

        assert!(sys.may(&coordinator_meta, Capability::Send, "proc-2"));
        assert!(!sys.may(&coordinator_meta, Capability::Send, "proc-3"));
    }
}
