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
    /// The durable record keeps the complete (bounded) body. When this is
    /// present, the live mailbox receives only a preview and this handle.
    #[serde(default)]
    pub artifact_id: Option<String>,
    #[serde(default)]
    pub artifact_chars: Option<usize>,
    pub body: String,
    pub low_priority: bool,
    pub reply_to: Option<String>,
}

/// A long mailbox body stored outside model context and paged explicitly by
/// its recipient. The recipient field is authorization data, not decoration.
#[derive(Clone, Serialize, Deserialize)]
pub struct MailArtifactRecord {
    pub id: String,
    pub recipient: String,
    pub from: String,
    pub chars: usize,
    pub body: String,
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
    Consumed {
        through: u64,
    },
    Spawned(ProcessRecord),
    /// A completed model turn: the assistant's content blocks.
    Output {
        content: Value,
    },
    /// A user turn — tool results, mail, or both.
    Input {
        content: Value,
    },
    Stopped {
        reason: String,
    },
    /// The conversation was replaced by a summary of itself. Recorded so a
    /// resume gets the compacted form rather than replaying the turns it was
    /// summarised from — otherwise compaction would be undone by the next
    /// restart.
    Compacted {
        history: Vec<Value>,
    },
    /// A model or effort change made while the process was running. Without
    /// this the switch would live only in memory and quietly revert on the
    /// next restart.
    Retuned {
        model: String,
        effort: Option<String>,
    },
    /// `patch_script` replaced a running script's code. Without this the
    /// replacement would live only in the isolate that was running it, and a
    /// restart would bring back the process's first draft instead.
    Patched {
        source: String,
    },
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
    fn store_mail_artifact(&self, _artifact: &MailArtifactRecord) {}
    fn read_mail_artifact(&self, _recipient: &str, _id: &str) -> Option<MailArtifactRecord> {
        None
    }
    fn list_mail_artifacts(&self, _recipient: &str) -> Vec<MailArtifactRecord> {
        Vec::new()
    }
    fn discard_mail_artifact(&self, _recipient: &str, _id: &str) -> bool {
        false
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
        self.root.join(format!("{}.jsonl", safe_component(process)))
    }

    fn artifact_dir(&self, process: &str) -> std::path::PathBuf {
        self.root.join("mail").join(safe_component(process))
    }

    fn artifact_path(&self, process: &str, id: &str) -> std::path::PathBuf {
        self.artifact_dir(process)
            .join(format!("{}.json", safe_component(id)))
    }
}

fn safe_component(value: &str) -> String {
    value
        .chars()
        .filter(|character| {
            character.is_ascii_alphanumeric() || *character == '-' || *character == '_'
        })
        .collect()
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
        let Some((record, history, stopped, pending, _, mail_cursor)) = restore(events) else {
            // Nothing to fold onto — no spawn record means no process.
            return;
        };
        let consumed_through = pending
            .iter()
            .map(|mail| mail.seq)
            .min()
            .unwrap_or(mail_cursor.saturating_add(1))
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
        if let Ok(file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
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
        let mark = self
            .marks
            .lock()
            .unwrap()
            .get(process)
            .copied()
            .unwrap_or(0);
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

    fn store_mail_artifact(&self, artifact: &MailArtifactRecord) {
        use std::io::Write;

        let directory = self.artifact_dir(&artifact.recipient);
        if std::fs::create_dir_all(&directory).is_err() {
            return;
        }
        let path = self.artifact_path(&artifact.recipient, &artifact.id);
        let staged = path.with_extension("json.new");
        let Ok(serialized) = serde_json::to_vec(artifact) else {
            return;
        };
        let wrote = (|| -> std::io::Result<()> {
            let mut file = std::fs::File::create(&staged)?;
            file.write_all(&serialized)?;
            file.sync_all()
        })();
        if wrote.is_err() || std::fs::rename(&staged, &path).is_err() {
            let _ = std::fs::remove_file(&staged);
        }
    }

    fn read_mail_artifact(&self, recipient: &str, id: &str) -> Option<MailArtifactRecord> {
        let bytes = std::fs::read(self.artifact_path(recipient, id)).ok()?;
        let artifact: MailArtifactRecord = serde_json::from_slice(&bytes).ok()?;
        (artifact.recipient == recipient && artifact.id == id).then_some(artifact)
    }

    fn list_mail_artifacts(&self, recipient: &str) -> Vec<MailArtifactRecord> {
        let mut artifacts: Vec<MailArtifactRecord> =
            std::fs::read_dir(self.artifact_dir(recipient))
                .into_iter()
                .flatten()
                .flatten()
                .filter(|entry| {
                    entry
                        .path()
                        .extension()
                        .is_some_and(|extension| extension == "json")
                })
                .filter_map(|entry| std::fs::read(entry.path()).ok())
                .filter_map(|bytes| serde_json::from_slice(&bytes).ok())
                .filter(|artifact: &MailArtifactRecord| artifact.recipient == recipient)
                .collect();
        artifacts.sort_by(|left, right| left.id.cmp(&right.id));
        artifacts
    }

    fn discard_mail_artifact(&self, recipient: &str, id: &str) -> bool {
        // Validate the record before deleting. Sanitizing a hostile id keeps
        // it inside the directory, but two spellings can sanitize to the same
        // filename; only the exact recorded id may remove that file.
        if self.read_mail_artifact(recipient, id).is_none() {
            return false;
        }
        std::fs::remove_file(self.artifact_path(recipient, id)).is_ok()
    }
}

/// Fold a journal back into a process definition and its conversation.
///
/// An assistant turn whose tool calls have no matching results is dropped: the
/// harness died mid-turn, and re-running it is both correct and cheaper than
/// trying to reconstruct what the tools would have returned. The dropped
/// calls are returned so the caller can warn the process — some of them may
/// have executed before the crash (mail, for one, is durable at the recipient
/// the moment it is sent), and silently re-running them means duplicate
/// messages and duplicate workers.
pub fn restore(
    events: Vec<Event>,
) -> Option<(
    ProcessRecord,
    Vec<Value>,
    bool,
    Vec<MailRecord>,
    Vec<Value>,
    u64,
)> {
    let mut record = None;
    let mut history: Vec<Value> = Vec::new();
    let mut stopped = false;
    let mut enqueued: Vec<MailRecord> = Vec::new();
    let mut consumed_through = 0u64;
    let mut mail_cursor = 0u64;
    let mut retuned: Option<(String, Option<String>)> = None;
    let mut patched: Option<String> = None;
    for event in events {
        match event {
            Event::Spawned(spawned) => record = Some(spawned),
            Event::Enqueued(mail) => {
                mail_cursor = mail_cursor.max(mail.seq);
                enqueued.push(mail);
            }
            Event::Consumed { through } => {
                consumed_through = consumed_through.max(through);
                mail_cursor = mail_cursor.max(through);
            }
            Event::Output { content } => {
                history.push(serde_json::json!({"role": "assistant", "content": content}))
            }
            Event::Input { content } => {
                history.push(serde_json::json!({"role": "user", "content": content}))
            }
            Event::Stopped { .. } => stopped = true,
            Event::Retuned { model, effort } => retuned = Some((model, effort)),
            Event::Patched { source } => patched = Some(source),
            Event::Compacted {
                history: summarised,
            } => history = summarised,
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
                mail_cursor = enqueued
                    .iter()
                    .map(|mail| mail.seq)
                    .max()
                    .unwrap_or(through)
                    .max(through);
            }
        }
    }
    let mut dropped: Vec<Value> = Vec::new();
    if let Some(last) = history.last() {
        let unanswered = last["role"] == "assistant"
            && last["content"]
                .as_array()
                .is_some_and(|blocks| blocks.iter().any(|b| b["type"] == "tool_use"));
        if unanswered {
            if let Some(turn) = history.pop() {
                dropped = turn["content"]
                    .as_array()
                    .map(|blocks| {
                        blocks
                            .iter()
                            .filter(|b| b["type"] == "tool_use")
                            .cloned()
                            .collect()
                    })
                    .unwrap_or_default();
            }
        }
    }
    // Whatever was never consumed is still owed to this process.
    let pending: Vec<MailRecord> = enqueued
        .into_iter()
        .filter(|mail| mail.seq > consumed_through)
        .collect();
    record.map(|mut record| {
        // Applied last, so it wins over whatever the spawn recorded.
        if let Some((model, effort)) = retuned {
            record.model = model;
            if effort.is_some() {
                record.effort = effort;
            }
        }
        // Only a script has code to replace, but a stray event on an agent's
        // log would be a bug elsewhere, not something to unwrap and panic on.
        if let Some(source) = patched {
            record.kind = crate::system::Kind::Script(source);
        }
        (record, history, stopped, pending, dropped, mail_cursor)
    })
}

/// What a resumed process is told when its last turn was cut off mid-tools.
/// Uncertain work is never replayed silently: some of the dropped calls may
/// have executed before the crash (mail, for one, is durable at the recipient
/// the moment it is sent), so the process has to check before it repeats
/// anything with side effects.
pub fn restart_notice(dropped: &[Value]) -> Option<String> {
    if dropped.is_empty() {
        return None;
    }
    let calls: Vec<String> = dropped
        .iter()
        .map(|b| {
            let name = b["name"].as_str().unwrap_or("?");
            let input: String = b["input"].to_string().chars().take(200).collect();
            format!("- {name}({input})")
        })
        .collect();
    Some(format!(
        "<restart_notice>\nThe harness restarted while you were mid-turn. Your last turn was \
         lost before its tool results were recorded, so these tool calls may or may not have \
         taken effect:\n{}\nVerify before repeating anything with side effects — a process you \
         spawned may already exist (list_processes), and a message you sent may already have \
         been delivered. Then continue the work.\n</restart_notice>",
        calls.join("\n")
    ))
}

/// Put the restart notice where the resumed conversation will read it next:
/// appended to the trailing user turn when there is one, or as a user turn of
/// its own.
pub fn attach_restart_notice(history: &mut Vec<Value>, notice: &str) {
    let block = serde_json::json!({"type": "text", "text": notice});
    if let Some(last) = history.last_mut() {
        if last["role"] == "user" {
            if let Some(blocks) = last["content"].as_array_mut() {
                blocks.push(block);
                return;
            }
        }
    }
    history.push(serde_json::json!({"role": "user", "content": [block]}));
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
            artifact_id: None,
            artifact_chars: None,
            body: format!("message {seq}"),
            low_priority: false,
            reply_to: None,
        }
    }

    fn temporary_journal(name: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("bitty-{name}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn old_mail_records_default_to_inline() {
        let record: MailRecord = serde_json::from_value(serde_json::json!({
            "seq": 1,
            "from": "proc-1",
            "from_name": null,
            "body": "hello",
            "low_priority": false,
            "reply_to": null
        }))
        .unwrap();
        assert!(record.artifact_id.is_none());
        assert!(record.artifact_chars.is_none());
    }

    #[test]
    fn mailbox_artifacts_survive_reopen_and_are_recipient_scoped() {
        let root = temporary_journal("mail-artifact");
        let artifact = MailArtifactRecord {
            id: "mail-proc-2-7".into(),
            recipient: "proc-2".into(),
            from: "proc-1".into(),
            chars: 9,
            body: "kept here".into(),
        };
        {
            let journal = FileJournal::new(&root).unwrap();
            journal.store_mail_artifact(&artifact);
        }
        let journal = FileJournal::new(&root).unwrap();
        assert_eq!(
            journal
                .read_mail_artifact("proc-2", "mail-proc-2-7")
                .unwrap()
                .body,
            "kept here"
        );
        assert!(
            journal
                .read_mail_artifact("proc-1", "mail-proc-2-7")
                .is_none()
        );
        assert_eq!(journal.list_mail_artifacts("proc-2").len(), 1);
        // A different spelling that sanitizes similarly must not delete it.
        assert!(!journal.discard_mail_artifact("proc-2", "mail/-proc-2-7"));
        assert!(journal.discard_mail_artifact("proc-2", "mail-proc-2-7"));
        assert!(journal.list_mail_artifacts("proc-2").is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }

    /// Fold the way `compact` does, so the test exercises the same derivation
    /// the compactor uses rather than a parallel one.
    fn checkpoint_of(events: Vec<Event>) -> Event {
        let (record, history, stopped, pending, _, mail_cursor) =
            restore(events).expect("has a spawn record");
        let consumed_through = pending
            .iter()
            .map(|m| m.seq)
            .min()
            .unwrap_or(mail_cursor.saturating_add(1))
            .saturating_sub(1);
        Event::Checkpoint {
            record,
            history,
            consumed_through,
            pending,
            stopped,
        }
    }

    fn same(a: Vec<Event>, b: Vec<Event>) -> bool {
        let (ar, ah, as_, ap, _, ac) = restore(a).unwrap();
        let (br, bh, bs, bp, _, bc) = restore(b).unwrap();
        ar.id == br.id
            && ar.instructions == br.instructions
            && ah == bh
            && as_ == bs
            && ap.iter().map(|m| m.seq).eq(bp.iter().map(|m| m.seq))
            && ac == bc
    }

    fn stream() -> Vec<Event> {
        vec![
            Event::Spawned(record("proc-2")),
            Event::Input {
                content: serde_json::json!([{"type": "text", "text": "go"}]),
            },
            Event::Output {
                content: serde_json::json!([{"type": "text", "text": "working"}]),
            },
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
        let (_, _, _, pending, _, _) = restore(vec![checkpoint_of(stream())]).unwrap();
        assert_eq!(pending.iter().map(|m| m.seq).collect::<Vec<_>>(), vec![3]);
    }

    #[test]
    fn a_fully_consumed_mail_cursor_survives_compaction() {
        let events = vec![
            Event::Spawned(record("proc-2")),
            Event::Enqueued(mail(1)),
            Event::Enqueued(mail(2)),
            Event::Consumed { through: 2 },
        ];
        let checkpoint = checkpoint_of(events);
        let (_, _, _, pending, _, cursor) = restore(vec![checkpoint]).unwrap();
        assert!(pending.is_empty());
        assert_eq!(cursor, 2);
    }

    /// A stopped process stays stopped across a checkpoint.
    #[test]
    fn stopped_survives() {
        let mut events = stream();
        events.push(Event::Stopped {
            reason: "done".into(),
        });
        let (_, _, stopped, _, _, _) = restore(vec![checkpoint_of(events)]).unwrap();
        assert!(stopped);
    }

    fn script_record(id: &str, source: &str) -> ProcessRecord {
        ProcessRecord {
            kind: crate::system::Kind::Script(source.into()),
            ..record(id)
        }
    }

    /// The bug this event exists to fix: patch_script's replacement has to
    /// win over the source the process was first spawned with, or a restart
    /// brings back the first draft.
    #[test]
    fn a_patch_replaces_the_spawned_source_on_restore() {
        let events = vec![
            Event::Spawned(script_record("proc-2", "v1")),
            Event::Patched {
                source: "v2".into(),
            },
        ];
        let (record, ..) = restore(events).unwrap();
        assert!(matches!(record.kind, crate::system::Kind::Script(s) if s == "v2"));
    }

    /// Compaction folds through `restore` (see `checkpoint_of`), so a patch
    /// has to survive being checkpointed the same way a retune does.
    #[test]
    fn a_patch_survives_being_folded_into_a_checkpoint() {
        let events = vec![
            Event::Spawned(script_record("proc-2", "v1")),
            Event::Patched {
                source: "v2".into(),
            },
        ];
        let (record, ..) = restore(vec![checkpoint_of(events)]).unwrap();
        assert!(matches!(record.kind, crate::system::Kind::Script(s) if s == "v2"));
    }

    /// A log ending mid-turn: the unanswered assistant turn is dropped AND its
    /// tool calls are reported, so the resumed process can be warned that they
    /// may already have run.
    fn mid_turn_events() -> Vec<Event> {
        vec![
            Event::Spawned(record("proc-2")),
            Event::Input {
                content: serde_json::json!([{"type": "text", "text": "go"}]),
            },
            Event::Output {
                content: serde_json::json!([
                    {"type": "text", "text": "spawning"},
                    {"type": "tool_use", "id": "t1", "name": "spawn_process",
                     "input": {"instructions": "count things"}}
                ]),
            },
        ]
    }

    #[test]
    fn a_dropped_turn_reports_its_tool_calls() {
        let (_, history, _, _, dropped, _) = restore(mid_turn_events()).unwrap();
        assert_eq!(history.len(), 1, "the unanswered turn is gone");
        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0]["name"], "spawn_process");
        let notice = restart_notice(&dropped).expect("a dropped call warrants a notice");
        assert!(notice.contains("spawn_process"));
        assert!(notice.contains("may or may not have taken effect"));
    }

    /// The corrective Compacted event a resume writes has to converge: a
    /// second restore of [original events + correction] must see the notice
    /// and no unanswered turn — without the correction, the dropped Output
    /// stays in the log and the next restart replays it mid-history with its
    /// tool calls forever unanswered, which the API rejects.
    #[test]
    fn the_resume_correction_converges() {
        let mut events = mid_turn_events();
        let (_, mut history, _, _, dropped, _) = restore(mid_turn_events()).unwrap();
        let notice = restart_notice(&dropped).unwrap();
        attach_restart_notice(&mut history, &notice);
        events.push(Event::Compacted {
            history: history.clone(),
        });
        // The resumed process then runs a full turn.
        events.push(Event::Output {
            content: serde_json::json!([{"type": "text", "text": "verified; continuing"}]),
        });

        let (_, replayed, _, _, dropped_again, _) = restore(events).unwrap();
        assert!(
            dropped_again.is_empty(),
            "the correction leaves nothing dangling"
        );
        let text = serde_json::to_string(&replayed).unwrap();
        assert!(text.contains("restart_notice"));
        let unanswered_mid_history = replayed.iter().any(|turn| {
            turn["role"] == "assistant"
                && turn["content"]
                    .as_array()
                    .is_some_and(|blocks| blocks.iter().any(|b| b["type"] == "tool_use"))
        });
        assert!(
            !unanswered_mid_history,
            "no orphaned tool_use survives the correction"
        );
    }

    /// The notice lands inside the trailing user turn when there is one, so
    /// the conversation still alternates roles.
    #[test]
    fn the_notice_joins_the_trailing_user_turn() {
        let mut history =
            vec![serde_json::json!({"role": "user", "content": [{"type": "text", "text": "go"}]})];
        attach_restart_notice(&mut history, "<restart_notice>x</restart_notice>");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0]["content"].as_array().unwrap().len(), 2);
    }
}
