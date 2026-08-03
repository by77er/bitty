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
    /// A fold of everything before it, standing in for those events entirely.
    ///
    /// This is exactly what `restore` computes, written back out — which is
    /// what makes compaction provably behavior-preserving, and what lets a
    /// reader that starts at a checkpoint need nothing before it.
    Checkpoint {
        record: ProcessRecord,
        history: Vec<Value>,
        consumed_through: u64,
        pending: Vec<MailRecord>,
        stopped: bool,
    },
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
    /// Replace this process's log with a single checkpoint standing for it.
    /// A no-op by default: a durable-execution backend owns its own history and
    /// has no use for ours.
    fn compact(&self, _process: &str) {}
    /// Whether the log has grown enough since its last checkpoint to be worth
    /// rewriting. Asked at flush boundaries, so compaction never interrupts a
    /// turn in progress.
    fn should_compact(&self, _process: &str) -> bool {
        false
    }
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
    /// File size at each process's last checkpoint. Growth is measured from
    /// there rather than from zero, which is what keeps compaction amortized.
    marks: std::sync::Mutex<std::collections::HashMap<String, u64>>,
}

impl FileJournal {
    pub fn new(root: impl Into<std::path::PathBuf>) -> std::io::Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        Ok(FileJournal {
            root,
            files: std::sync::Mutex::new(std::collections::HashMap::new()),
            marks: std::sync::Mutex::new(std::collections::HashMap::new()),
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

    /// Rewrite the log as one checkpoint plus nothing else.
    ///
    /// The checkpoint is derived from the *file*, never from live memory. That
    /// buys two things. It cannot disagree with what a restore would have
    /// produced, because it is the same fold. And it cannot drop mail unsafely:
    /// consumed cursors are batched, so memory may know a higher one than disk
    /// does, and reading the file means we only ever discard messages whose
    /// consumption is already durable.
    fn compact(&self, process: &str) {
        use std::io::Write;

        // Held across the whole swap. An event written to the old handle after
        // the rename would land in an unlinked file and be lost.
        let mut files = self.files.lock().unwrap();
        if let Some(writer) = files.get_mut(process) {
            let _ = writer.flush();
        }

        let path = self.path(process);
        let Ok(text) = std::fs::read_to_string(&path) else {
            return;
        };
        let events: Vec<Event> = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        let before = events.len();
        let Some((record, history, stopped, pending)) = restore(events) else {
            // Nothing to fold onto — no spawn record means no process.
            return;
        };
        let consumed_through = pending
            .iter()
            .map(|mail| mail.seq)
            .min()
            .unwrap_or(0)
            .saturating_sub(1);
        let checkpoint = Event::Checkpoint {
            record,
            history,
            consumed_through,
            pending,
            stopped,
        };
        let Ok(line) = serde_json::to_string(&checkpoint) else {
            return;
        };

        // Written beside the original and renamed over it: a crash leaves one
        // intact file or the other, never a torn one.
        let staged = path.with_extension("jsonl.new");
        let wrote = (|| -> std::io::Result<()> {
            let mut file = std::fs::File::create(&staged)?;
            writeln!(file, "{line}")?;
            file.sync_all()
        })();
        if wrote.is_err() {
            let _ = std::fs::remove_file(&staged);
            return;
        }

        // Verify before committing. One extra parse turns "corruption found at
        // resume, with the original already gone" into "compaction declined".
        let verified = std::fs::read_to_string(&staged)
            .ok()
            .and_then(|text| serde_json::from_str::<Event>(text.trim()).ok())
            .map(|event| restore(vec![event]).is_some())
            .unwrap_or(false);
        if !verified || std::fs::rename(&staged, &path).is_err() {
            let _ = std::fs::remove_file(&staged);
            return;
        }

        // The old handle points at the replaced inode; reopen onto the new file.
        files.remove(process);
        if let Ok(file) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            files.insert(process.to_string(), std::io::BufWriter::new(file));
        }
        let now = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        self.marks.lock().unwrap().insert(process.to_string(), now);
        crate::ui::system(&format!("compacted {process}: {before} events → 1"));
    }

    /// Amortized doubling with a floor: compact when the log has grown past
    /// twice its size at the last checkpoint. Cost stays O(1) per event, and a
    /// log that is legitimately large is not rewritten over and over.
    fn should_compact(&self, process: &str) -> bool {
        // Overridable so a test can force compaction without writing a
        // quarter-megabyte of events first.
        let floor: u64 = std::env::var("BITTY_COMPACT_FLOOR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256 * 1024);
        let Ok(meta) = std::fs::metadata(self.path(process)) else {
            return false;
        };
        if meta.len() < floor {
            return false;
        }
        let mark = self.marks.lock().unwrap().get(process).copied().unwrap_or(0);
        meta.len() >= mark.max(floor).saturating_mul(2)
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
            // A checkpoint supersedes everything before it: reset, then keep
            // folding whatever was appended afterward.
            Event::Checkpoint {
                record: checkpointed,
                history: kept,
                consumed_through: through,
                pending,
                stopped: was_stopped,
            } => {
                record = Some(checkpointed);
                history = kept;
                consumed_through = through;
                stopped = was_stopped;
                enqueued = pending;
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &str) -> ProcessRecord {
        ProcessRecord {
            id: id.into(),
            name: Some("worker".into()),
            parent: "proc-1".into(),
            persona: None,
            instructions: "do the thing".into(),
            inherited: None,
            grants: crate::grants::Grants::unrestricted(),
            aliases: Vec::new(),
            model: "claude-opus-5".into(),
            effort: None,
            linked: true,
            kind: crate::system::Kind::Agent,
            ordinal: 2,
        }
    }

    fn mail(seq: u64) -> MailRecord {
        MailRecord {
            seq,
            from: "proc-1".into(),
            from_name: None,
            body: format!("message {seq}"),
            low_priority: false,
            reply_to: None,
        }
    }

    /// Fold the way `compact` does, so the test exercises the same derivation
    /// the compactor uses rather than a parallel one.
    fn checkpoint_of(events: Vec<Event>) -> Event {
        let (record, history, stopped, pending) = restore(events).expect("has a spawn record");
        let consumed_through = pending
            .iter()
            .map(|m| m.seq)
            .min()
            .unwrap_or(0)
            .saturating_sub(1);
        Event::Checkpoint { record, history, consumed_through, pending, stopped }
    }

    fn same(a: Vec<Event>, b: Vec<Event>) -> bool {
        let (ar, ah, as_, ap) = restore(a).unwrap();
        let (br, bh, bs, bp) = restore(b).unwrap();
        ar.id == br.id
            && ar.instructions == br.instructions
            && ah == bh
            && as_ == bs
            && ap.iter().map(|m| m.seq).eq(bp.iter().map(|m| m.seq))
    }

    fn stream() -> Vec<Event> {
        vec![
            Event::Spawned(record("proc-2")),
            Event::Input { content: serde_json::json!([{"type": "text", "text": "go"}]) },
            Event::Output { content: serde_json::json!([{"type": "text", "text": "working"}]) },
            Event::Enqueued(mail(1)),
            Event::Enqueued(mail(2)),
            Event::Consumed { through: 1 },
            Event::Enqueued(mail(3)),
            Event::Consumed { through: 2 },
        ]
    }

    /// The invariant compaction rests on: folding a log and folding its
    /// checkpoint have to produce the same process.
    #[test]
    fn checkpoint_preserves_the_fold() {
        let events = stream();
        let compacted = vec![checkpoint_of(stream())];
        assert!(same(events, compacted));
    }

    /// A checkpoint has to be a starting point, not just an ending one: events
    /// appended after it must fold on top rather than be ignored.
    #[test]
    fn events_after_a_checkpoint_still_apply() {
        let mut compacted = vec![checkpoint_of(stream())];
        compacted.push(Event::Enqueued(mail(4)));
        compacted.push(Event::Consumed { through: 3 });

        let mut full = stream();
        full.push(Event::Enqueued(mail(4)));
        full.push(Event::Consumed { through: 3 });

        assert!(same(full, compacted));
    }

    /// Compacting twice must be a no-op, or repeated checkpoints would drift.
    #[test]
    fn compaction_is_idempotent() {
        let once = vec![checkpoint_of(stream())];
        let twice = vec![checkpoint_of(vec![checkpoint_of(stream())])];
        assert!(same(once, twice));
    }

    /// Unconsumed mail is the one thing a checkpoint must never lose.
    #[test]
    fn pending_mail_survives() {
        let (_, _, _, pending) = restore(vec![checkpoint_of(stream())]).unwrap();
        assert_eq!(pending.iter().map(|m| m.seq).collect::<Vec<_>>(), vec![3]);
    }

    /// A stopped process stays stopped across a checkpoint.
    #[test]
    fn stopped_survives() {
        let mut events = stream();
        events.push(Event::Stopped { reason: "done".into() });
        let (_, _, stopped, _) = restore(vec![checkpoint_of(events)]).unwrap();
        assert!(stopped);
    }
}
