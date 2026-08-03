//! Durable processes: a process's life as an append-only event journal.
//!
//! The abstraction is deliberately the *journal*, not a "storage backend".
//! Temporal is not a place to put bytes — it owns control flow, and adopting it
//! would mean restructuring the agent loop into a workflow plus activities. But
//! the state model ports cleanly: locally these events are lines in a file; on
//! Temporal the same events are the workflow's history, folded by replay. So
//! this trait makes a process's *state* portable, not its runtime.
//!
//! Events are recorded at turn boundaries, which is what makes restore
//! consistent: a process is always resumed at a point where it is about to make
//! a request.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Everything needed to bring a process back without re-deriving anything.
/// Grants are stored already resolved, so a restored process holds exactly the
/// authority it held before rather than being re-attenuated against a tree that
/// may no longer exist.
#[derive(Clone, Serialize, Deserialize)]
pub struct ProcessRecord {
    pub id: String,
    pub name: Option<String>,
    pub parent: String,
    pub persona: Option<String>,
    pub instructions: String,
    pub inherited: Option<String>,
    pub grants: crate::grants::Grants,
    pub aliases: Vec<crate::system::ToolAlias>,
    pub model: String,
    pub effort: Option<String>,
    pub linked: bool,
    pub kind: crate::system::Kind,
    pub ordinal: u64,
}

/// A message as recorded in a mailbox log.
#[derive(Clone, Serialize, Deserialize)]
pub struct MailRecord {
    pub seq: u64,
    pub from: String,
    pub from_name: Option<String>,
    pub body: String,
    pub low_priority: bool,
    pub reply_to: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "event")]
pub enum Event {
    /// A message arrived. Written by the *sender*, so mail is durable the
    /// moment it is sent rather than when it is read.
    Enqueued(MailRecord),
    /// The recipient has folded everything up to `through` into a turn.
    /// Popping is a cursor advance, not a destructive read — which is what
    /// makes the queue reconstructible instead of merely recoverable.
    Consumed { through: u64 },
    Spawned(ProcessRecord),
    /// A completed model turn: the assistant's content blocks.
    Output { content: Value },
    /// A user turn — tool results, mail, or both.
    Input { content: Value },
    Stopped { reason: String },
}

/// Where a process's events live.
pub trait Journal: Send + Sync {
    fn record(&self, process: &str, event: &Event);
    fn replay(&self, process: &str) -> Vec<Event>;
    /// Process ids with a journal, oldest first.
    fn processes(&self) -> Vec<String>;
    /// Push buffered events to disk. Called at turn boundaries — the points
    /// where a record must be durable before anything acts on it.
    fn flush(&self, _process: &str) {}
    fn enabled(&self) -> bool {
        true
    }
}

/// The default: remember nothing, cost nothing.
pub struct NoJournal;

impl Journal for NoJournal {
    fn record(&self, _process: &str, _event: &Event) {}
    fn replay(&self, _process: &str) -> Vec<Event> {
        Vec::new()
    }
    fn processes(&self) -> Vec<String> {
        Vec::new()
    }
    fn enabled(&self) -> bool {
        false
    }
}

/// One JSONL file per process. Chosen over SQLite so the record stays
/// readable, diffable and committable; move to SQLite if torn writes turn out
/// to be a real problem rather than a theoretical one.
pub struct FileJournal {
    root: std::path::PathBuf,
    /// One buffered, held-open handle per process. Recording an event used to
    /// be an open-write-close triple; now it is a memcpy, and the syscall
    /// happens once per turn instead of once per event. Easier on an SSD and
    /// considerably faster besides.
    files: std::sync::Mutex<std::collections::HashMap<String, std::io::BufWriter<std::fs::File>>>,
}

impl FileJournal {
    pub fn new(root: impl Into<std::path::PathBuf>) -> std::io::Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        Ok(FileJournal {
            root,
            files: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    fn path(&self, process: &str) -> std::path::PathBuf {
        // Ids are harness-generated ("proc-7"), but never trust one into a path.
        let safe: String = process
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        self.root.join(format!("{safe}.jsonl"))
    }
}

impl Journal for FileJournal {
    fn record(&self, process: &str, event: &Event) {
        use std::io::Write;
        let Ok(line) = serde_json::to_string(event) else {
            return;
        };
        let mut files = self.files.lock().unwrap();
        let writer = match files.entry(process.to_string()) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(slot) => {
                let file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(self.path(process));
                let Ok(file) = file else { return };
                slot.insert(std::io::BufWriter::new(file))
            }
        };
        let _ = writeln!(writer, "{line}");
        // Batching is for the high-frequency events. Spawned and Stopped are
        // neither frequent nor optional: Spawned is the only record that makes
        // a process restorable at all, and a script process has no turns, so
        // nothing else would ever flush it — its spawn would sit in this buffer
        // until the harness exited and the process would simply not come back.
        if matches!(event, Event::Spawned(_) | Event::Stopped { .. }) {
            let _ = writer.flush();
        }
    }

    fn flush(&self, process: &str) {
        use std::io::Write;
        if let Some(writer) = self.files.lock().unwrap().get_mut(process) {
            let _ = writer.flush();
        }
    }

    fn replay(&self, process: &str) -> Vec<Event> {
        // Anything still buffered for this process has to hit disk first, or a
        // replay in the same run would read a truncated log.
        self.flush(process);
        let Ok(text) = std::fs::read_to_string(self.path(process)) else {
            return Vec::new();
        };
        text.lines()
            .filter(|line| !line.trim().is_empty())
            // A torn final line is dropped rather than failing the restore.
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    }

    fn processes(&self) -> Vec<String> {
        let mut found: Vec<(u64, String)> = std::fs::read_dir(&self.root)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                if path.extension()? != "jsonl" {
                    return None;
                }
                let id = path.file_stem()?.to_string_lossy().to_string();
                let n = id.rsplit('-').next()?.parse::<u64>().ok()?;
                Some((n, id))
            })
            .collect();
        found.sort();
        found.into_iter().map(|(_, id)| id).collect()
    }
}

/// Fold a journal back into a process definition and its conversation.
///
/// An assistant turn whose tool calls have no matching results is dropped: the
/// harness died mid-turn, and re-running it is both correct and cheaper than
/// trying to reconstruct what the tools would have returned.
pub fn restore(
    events: Vec<Event>,
) -> Option<(ProcessRecord, Vec<Value>, bool, Vec<MailRecord>)> {
    let mut record = None;
    let mut history: Vec<Value> = Vec::new();
    let mut stopped = false;
    let mut enqueued: Vec<MailRecord> = Vec::new();
    let mut consumed_through = 0u64;
    for event in events {
        match event {
            Event::Spawned(spawned) => record = Some(spawned),
            Event::Enqueued(mail) => enqueued.push(mail),
            Event::Consumed { through } => consumed_through = consumed_through.max(through),
            Event::Output { content } => {
                history.push(serde_json::json!({"role": "assistant", "content": content}))
            }
            Event::Input { content } => {
                history.push(serde_json::json!({"role": "user", "content": content}))
            }
            Event::Stopped { .. } => stopped = true,
        }
    }
    if let Some(last) = history.last() {
        let unanswered = last["role"] == "assistant"
            && last["content"]
                .as_array()
                .is_some_and(|blocks| blocks.iter().any(|b| b["type"] == "tool_use"));
        if unanswered {
            history.pop();
        }
    }
    // Whatever was never consumed is still owed to this process.
    let pending: Vec<MailRecord> = enqueued
        .into_iter()
        .filter(|mail| mail.seq > consumed_through)
        .collect();
    record.map(|record| (record, history, stopped, pending))
}
