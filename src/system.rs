//! The actor system: a registry of processes, plus spawn/send/list/stop — the
//! same operations exposed to each agent as tools.
//!
//! Processes can be spawned one at a time, or as a *topology*: a group wired
//! together at birth, each with its own role, its own starting context, and an
//! allowlist of peers it may message. Permissions are resolved from symbolic
//! names to process ids once the whole group's ids are known.

use crate::api;
use crate::durable::{Event, Journal, MailRecord, NoJournal, ProcessRecord};
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
            priority: if record.low_priority { Priority::Low } else { Priority::High },
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

#[derive(Clone, Copy, PartialEq)]
pub enum Status {
    Running,
    Idle,
    Stopped,
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
    /// Model and effort, inherited by anything this process spawns.
    model: String,
    effort: Option<String>,
    /// "script", or "model/effort" for an agent — what /graph shows.
    runs: String,
    /// Script processes only: the channel for code replacement.
    control: Mutex<Option<UnboundedSender<Control>>>,
    /// Next position in this process's mailbox log.
    seq: AtomicU64,
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
    pub model: String,
    pub effort: Option<String>,
}

impl Meta {
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

    /// The single permission check. Every verb goes through here.
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
    /// In-flight synchronous calls, keyed by correlation id.
    pending: Mutex<HashMap<String, PendingCall>>,
    calls: AtomicU64,
    /// Where each process's life is recorded, so it can be brought back.
    pub journal: Arc<dyn Journal>,
}

impl System {
    pub fn new(api: api::Client) -> Self {
        System {
            rt: tokio::runtime::Handle::current(),
            procs: Mutex::new(Vec::new()),
            counter: AtomicU64::new(0),
            api,
            quiesce_announced: AtomicBool::new(false),
            spawning: Mutex::new(()),
            pending: Mutex::new(HashMap::new()),
            calls: AtomicU64::new(0),
            journal: Arc::new(NoJournal),
        }
    }

    pub fn with_journal(mut self, journal: Arc<dyn Journal>) -> Self {
        self.journal = journal;
        self
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
    ) {
        let (sender, receiver) = mpsc::unbounded_channel::<Mail>();
        let (control_tx, control_rx) = mpsc::unbounded_channel::<Control>();
        let status = Arc::new(Mutex::new(Status::Running));
        let context_tokens = Arc::new(AtomicU64::new(0));
        // Everything the log says was never consumed is owed to this process,
        // in the order it originally arrived.
        let highest_seq = pending.last().map(|m| m.seq).unwrap_or(0);
        for mail in pending {
            let _ = sender.send(Mail::from_record(mail));
        }
        let label = match &record.name {
            Some(name) => format!("{} {name}", record.id),
            None => record.id.clone(),
        };

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
            model: record.model.clone(),
            effort: record.effort.clone(),
            runs: record.kind.label(&record.model, &record.effort),
            control: Mutex::new(match record.kind {
                Kind::Script(_) => Some(control_tx),
                Kind::Agent => None,
            }),
            seq: AtomicU64::new(highest_seq),
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
            model: record.model.clone(),
            effort: record.effort.clone(),
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

    /// Record that a process has folded everything up to `seq` into a turn.
    /// Written alongside the turn itself, so the window in which a message
    /// could be seen twice is one append rather than a whole turn.
    pub fn note_consumed(&self, process: &str, seq: u64) {
        if seq > 0 {
            self.journal.record(process, &Event::Consumed { through: seq });
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
    ) -> (String, tokio::sync::oneshot::Receiver<Result<String, String>>) {
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
        match control.as_ref() {
            None => Err(format!(
                "{target} is an agent, not a script — there is no code to replace. Change what an \
                 agent does by messaging it."
            )),
            Some(tx) => match tx.send(Control::Replace(source)) {
                Ok(()) => Ok(format!(
                    "Replaced the code running in {target}. Its id, mailbox, links and permissions \
                     are unchanged; any state the old code held is gone."
                )),
                Err(_) => Err(format!("Process {target} is no longer running.")),
            },
        }
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
            if let Some(model) = &node.model {
                if !model.starts_with("claude-") {
                    return Err(format!(
                        "'{model}' is not a model this harness can use. Processes run on Claude \
                         models — try claude-opus-5, claude-sonnet-5 or claude-haiku-4-5, or omit \
                         the field to inherit yours."
                    ));
                }
            }
            if let Some(effort) = &node.effort {
                if !EFFORT_LEVELS.contains(&effort.as_str()) {
                    return Err(format!(
                        "Unknown effort '{effort}'. Use one of: {}.",
                        EFFORT_LEVELS.join(", ")
                    ));
                }
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

        // The spawner's grants are the ceiling, and its model/effort the
        // defaults, for everything below it.
        let (ceiling, inherited_model, inherited_effort, existing) = {
            let procs = self.procs.lock().unwrap();
            let me = procs.iter().find(|p| p.id == parent);
            let ceiling = me
                .map(|p| p.grants.clone())
                .unwrap_or_else(Grants::console_authority); // parent == "user"
            let model = me
                .map(|p| p.model.clone())
                .unwrap_or_else(|| self.api.model.clone());
            let effort = me.and_then(|p| p.effort.clone());
            let existing: Vec<(String, Option<String>)> = procs
                .iter()
                .map(|p| (p.id.clone(), p.name.clone()))
                .collect();
            (ceiling, model, effort, existing)
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
                        if let Some((_, id)) = by_name.iter().find(|(name, _)| *name == Some(peer)) {
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

            // An explicit allowlist means exactly that — messaging the spawner
            // is a policy choice, not an invariant, and the topology default
            // already grants it when the field is omitted. Self-stop is the one
            // true invariant: a process that cannot stop itself cannot exit.
            let send = checked(send_req.clone(), &ceiling.send, &[], "messaging", "message")?;
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
                &[self_id.clone()],
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
                &ceiling.run, &[], "running", "run")?;
            let net = checked(
                as_grant(&node.wants.net, &mut verbatim)?,
                &ceiling.net, &[], "network access to", "reach")?;
            let env = checked(
                as_grant(&node.wants.env, &mut verbatim)?,
                &ceiling.env, &[], "environment access to", "read")?;
            let sys = checked(
                as_grant(&node.wants.sys, &mut verbatim)?,
                &ceiling.sys, &[], "system info", "query")?;
            let read = paths(&node.wants.read, &ceiling.read, "read access")?;
            let write = paths(&node.wants.write, &ceiling.write, "write access")?;

            let grants = Grants { send, stop, spawn, run, net, env, sys, read, write };

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
            labels.entry(parent.to_string()).or_insert_with(|| "your spawner".into());
            labels.entry(self_id.clone()).or_insert_with(|| "yourself".into());
            resolved.push((grants, labels, aliases));
        }

        // Phase 3 — nothing can refuse the spawn from here, so the ids are
        // finally consumed.
        self.counter.store(base + nodes.len() as u64, Ordering::Relaxed);
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
                model: node.model.clone().unwrap_or_else(|| inherited_model.clone()),
                effort: node.effort.clone().or_else(|| inherited_effort.clone()),
                control: Mutex::new(match node.kind {
                    Kind::Script(_) => Some(control_tx),
                    Kind::Agent => None,
                }),
                seq: AtomicU64::new(0),
                runs: node.kind.label(
                    node.model.as_deref().unwrap_or(&inherited_model),
                    &node.effort.clone().or_else(|| inherited_effort.clone()),
                ),
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
                        model: node.model.clone().unwrap_or_else(|| inherited_model.clone()),
                        effort: node.effort.clone().or_else(|| inherited_effort.clone()),
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
                model: node.model.clone().unwrap_or_else(|| inherited_model.clone()),
                effort: node.effort.clone().or_else(|| inherited_effort.clone()),
            };

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
        let procs = self.procs.lock().unwrap();
        match procs.iter().find(|p| p.id == to) {
            Some(entry) => {
                if *entry.status.lock().unwrap() == Status::Stopped {
                    return Err(format!("Process {to} has been stopped; it cannot receive mail."));
                }
                // Recorded before delivery, so a crash between the two costs a
                // duplicate rather than a lost message — at-least-once, which
                // is the semantic a cursor-based queue can actually honor.
                mail.seq = entry.seq.fetch_add(1, Ordering::Relaxed) + 1;
                self.journal.record(
                    to,
                    &Event::Enqueued(MailRecord {
                        seq: mail.seq,
                        from: mail.from.clone(),
                        from_name: mail.from_name.clone(),
                        body: mail.body.clone(),
                        low_priority: mail.priority == Priority::Low,
                        reply_to: mail.reply_to.clone(),
                    }),
                );
                match entry.sender.lock().unwrap().as_ref() {
                    Some(sender) => match sender.send(mail) {
                        Ok(()) => Ok(format!("Delivered to {to}'s mailbox.")),
                        Err(_) => Err(format!("Process {to} is no longer running.")),
                    },
                    None => Err(format!("Process {to} is no longer running.")),
                }
            }
            None => Err(format!(
                "No process with id '{to}'. Use list_processes to see valid ids."
            )),
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
                    p.name.as_ref().map(|n| format!(" ({n})")).unwrap_or_default()
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
            ui::system(&format!(
                "— system idle · {waiting} process(es) waiting for a message"
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
        let procs = self.procs.lock().unwrap();
        let mut out = String::from("id       name           status   context  parent\n");
        for p in procs.iter().filter(|p| view.is_none_or(|v| v.contains(&p.id))) {
            out.push_str(&format!(
                "{:<8} {:<14} {:<8} {:<8} {}\n",
                p.id,
                p.name.as_deref().unwrap_or("-"),
                p.status.lock().unwrap().as_str(),
                format_tokens(p.context_tokens.load(Ordering::Relaxed)),
                p.parent,
            ));
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

        let width = rows.iter().map(|r| r.tree.chars().count()).max().unwrap_or(0);
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
        format!("{prefix}{}{} ", if last { "└" } else { "├" }, if entry.linked { "⚯" } else { "─" })
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
        notes.push(format!("runs→ {}", match &entry.grants.run {
            Grant::All => "any".to_string(),
            Grant::Ids(names) => {
                let mut n: Vec<String> = names.iter().cloned().collect();
                n.sort();
                n.join(" ")
            }
            Grant::Nobody => String::new(),
        }));
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
        walk(procs, child, &child_prefix, i + 1 == children.len(), false, rows);
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

#[cfg(test)]
mod tests {
    use super::tombstones_to_drop;

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
}
