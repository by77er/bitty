//! The per-process agent loop. Each process is an independent conversation
//! with the model; mailbox messages are injected into that conversation as
//! interrupts in the gaps between tool calls — the same UX as sending
//! messages to a coding agent mid-task. When a turn ends and the mailbox is
//! empty, the process goes idle and blocks until the next message arrives.
//!
//! Context per process is: a system prompt (harness scaffolding + optional
//! role) fixed for its lifetime, plus a conversation that starts either empty
//! (just the briefing) or seeded with a rendered snapshot of the spawner's
//! conversation.

use crate::grants::{Capability, Grant};
use crate::api::Turn;
use crate::actions;
use crate::durable::Event;
use crate::system::{
    GrantSpec, Kind, MAX_GROUP, Mail, Meta, NodeSpec, Priority, Status, System, ToolAlias,
};
use crate::ui;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tokio::sync::mpsc::UnboundedReceiver;

/// Only relevant when compaction is unavailable: warn once as a context
/// approaches a size where the 1M window becomes a real risk.
const UNMANAGED_CONTEXT_WARN: u64 = 500_000;

/// Caps on an inherited-context snapshot, so a fork can't blow out the child's
/// window with the parent's entire history.
const DEFAULT_CALL_TIMEOUT: u64 = 60;
const MAX_CALL_TIMEOUT: u64 = 300;

const MAX_BLOCK_CHARS: usize = 4_000;
const MAX_TRANSCRIPT_CHARS: usize = 60_000;

pub async fn run(
    sys: Arc<System>,
    me: Meta,
    mailbox: UnboundedReceiver<Mail>,
    instructions: String,
    inherited: Option<String>,
) {
    let mut first_turn: Vec<Value> = Vec::new();
    if let Some(transcript) = inherited {
        ui::trace(&me.tag, "⇠ inherited context from spawner");
        first_turn.push(text_block(&format!(
            "<inherited_context from=\"{}\">\nThis is a snapshot of your spawner's conversation up \
             to the moment you were created, for background. You did not do any of this work \
             yourself.\n\n{}\n</inherited_context>",
            me.parent, transcript
        )));
    }
    // An empty briefing means "start and wait" — the process comes up idle
    // rather than burning a turn on nothing.
    if instructions.trim().is_empty() && first_turn.is_empty() {
        drive(sys, me, mailbox, Vec::new()).await;
        return;
    }
    first_turn.push(text_block(&envelope(&Mail::system(&me.parent, instructions))));
    let opening = json!({"role": "user", "content": first_turn});
    sys.journal.record(&me.id, &Event::Input { content: json!(first_turn) });
    drive(sys, me, mailbox, vec![opening]).await;
}

/// Bring a process back with the conversation it already had.
pub async fn resume(
    sys: Arc<System>,
    me: Meta,
    mailbox: UnboundedReceiver<Mail>,
    _instructions: String,
    _inherited: Option<String>,
    history: Vec<Value>,
) {
    ui::trace(&me.tag, &format!("↻ resumed with {} prior turns", history.len()));
    drive(sys, me, mailbox, history).await;
}

async fn drive(
    sys: Arc<System>,
    me: Meta,
    mut mailbox: UnboundedReceiver<Mail>,
    mut history: Vec<Value>,
) {
    let system_prompt = system_blocks(&me);
    let tools = tool_definitions(&me);
    // Low-priority mail that arrived while idle. Held rather than acted on,
    // and flushed into the next turn this process runs for any other reason.
    let mut deferred: Vec<Mail> = Vec::new();
    let mut consecutive_failures: u32 = 0;
    let mut warned_unmanaged = false;

    loop {
        // Nothing to answer yet: wait for the first message instead of
        // sending an empty conversation to the API.
        if history.is_empty() && !wait_for_mail(&sys, &me, &mut mailbox, &mut history, &mut deferred).await {
            return;
        }
        me.set_status(Status::Running);
        sys.note_running();
        let turn = Turn {
            system: &system_prompt,
            messages: &history,
            tools: &tools,
            model: &me.model,
            effort: me.effort.as_deref(),
        };
        let resp = match sys.api.message(turn, &me.tag).await {
            Ok(resp) => {
                consecutive_failures = 0;
                me.context_tokens.store(resp.input_tokens, Ordering::Relaxed);
                if !warned_unmanaged
                    && resp.input_tokens > UNMANAGED_CONTEXT_WARN
                    && !sys.api.compaction_enabled()
                {
                    warned_unmanaged = true;
                    ui::warn(
                        &me.tag,
                        &format!(
                            "context is {}k tokens and compaction is unavailable — this process \
                             will eventually exhaust the window",
                            resp.input_tokens / 1_000
                        ),
                    );
                }
                resp
            }
            Err(e) => {
                consecutive_failures += 1;
                ui::warn(&me.tag, &format!("turn failed: {e:#}"));
                if consecutive_failures >= 3 {
                    ui::warn(&me.tag, "giving up on this turn; going idle until next message");
                    // Not dead, but not working either — tell the neighbors so
                    // whoever delegated to this process can stop waiting.
                    sys.signal_stalled(&me.id, &me.label());
                    if !wait_for_mail(&sys, &me, &mut mailbox, &mut history, &mut deferred).await {
                        return;
                    }
                    consecutive_failures = 0;
                }
                continue;
            }
        };

        let content = sanitize_for_history(resp.content);
        sys.journal.record(&me.id, &Event::Output { content: json!(content) });
        history.push(json!({"role": "assistant", "content": content}));

        match resp.stop_reason.as_str() {
            "tool_use" => {
                let mut blocks: Vec<Value> = Vec::new();
                for block in content.iter().filter(|b| b["type"] == "tool_use") {
                    let name = block["name"].as_str().unwrap_or("");
                    let input = &block["input"];
                    ui::trace(&me.tag, &format!("→ {name} {}", truncate(&input.to_string(), 200)));
                    let (result, is_error) = execute_tool(&sys, &me, name, input, &history).await;
                    if is_error {
                        // The model gets the full text in its tool result; the
                        // console gets one line. A compiler diagnostic dumped
                        // into the terminal buries everything around it.
                        ui::trace(&me.tag, &format!("  ✗ {}", first_line(&result)));
                    }
                    blocks.push(json!({
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": result,
                        "is_error": is_error,
                    }));
                    // Self-stop: exit before the abort lands at the next await.
                    if me.is_stopped() {
                        return;
                    }
                }
                // Interrupts: anything in the mailbox lands in context right
                // here, after the results. The process is already running, so
                // low-priority mail rides along at no extra cost — that is the
                // whole point of holding it.
                let mut consumed = 0;
                for mail in deferred.drain(..) {
                    ui::trace(&me.tag, &format!("⇠ held mail from {}", mail.from));
                    consumed = consumed.max(mail.seq);
                    blocks.push(text_block(&envelope(&mail)));
                }
                while let Ok(mail) = mailbox.try_recv() {
                    ui::trace(&me.tag, &format!("⇠ interrupt: mail from {}", mail.from));
                    consumed = consumed.max(mail.seq);
                    blocks.push(text_block(&envelope(&mail)));
                }
                sys.note_consumed(&me.id, consumed);
                // Recorded after the results, so a crash mid-turn resumes from
                // a point where the completed tool calls are already known and
                // are not run a second time.
                sys.journal.record(&me.id, &Event::Input { content: json!(blocks) });
                sys.journal.flush(&me.id);
                history.push(json!({"role": "user", "content": blocks}));
            }
            // Server-side pause; re-send as-is to let the turn resume.
            "pause_turn" => continue,
            other => {
                if other == "refusal" {
                    ui::warn(&me.tag, "the model declined this request (stop_reason: refusal)");
                } else if other == "max_tokens" {
                    ui::warn(&me.tag, "turn truncated at max_tokens");
                }
                if !wait_for_mail(&sys, &me, &mut mailbox, &mut history, &mut deferred).await {
                    return;
                }
            }
        }
    }
}

/// Idle until mail arrives, then push it (plus anything queued behind it) as
/// the next user message. Returns false if the mailbox closed (shutdown).
async fn wait_for_mail(
    sys: &Arc<System>,
    me: &Meta,
    mailbox: &mut UnboundedReceiver<Mail>,
    history: &mut Vec<Value>,
    deferred: &mut Vec<Mail>,
) -> bool {
    if me.set_status(Status::Idle) {
        ui::trace(&me.tag, "· idle, waiting for messages");
        sys.note_quiesced();
    }
    // Only high-priority mail ends the wait. Low-priority mail is stashed and
    // stays stashed — waking for it is exactly the turn we are trying not to
    // spend. It goes out with whatever wakes this process next.
    let waker = loop {
        let Some(mail) = mailbox.recv().await else {
            return false;
        };
        if mail.priority == Priority::Low {
            ui::trace(&me.tag, &format!("· holding low-priority mail from {}", mail.from));
            deferred.push(mail);
            continue;
        }
        break mail;
    };

    let mut blocks: Vec<Value> = Vec::new();
    let mut consumed = 0;
    // Chronological: anything held while idle predates the message that woke us.
    for mail in deferred.drain(..) {
        ui::trace(&me.tag, &format!("⇠ held mail from {}", mail.from));
        consumed = consumed.max(mail.seq);
        blocks.push(text_block(&envelope(&mail)));
    }
    ui::trace(&me.tag, &format!("⇠ mail from {}", waker.from));
    consumed = consumed.max(waker.seq);
    blocks.push(text_block(&envelope(&waker)));
    while let Ok(mail) = mailbox.try_recv() {
        ui::trace(&me.tag, &format!("⇠ mail from {}", mail.from));
        consumed = consumed.max(mail.seq);
        blocks.push(text_block(&envelope(&mail)));
    }
    sys.note_consumed(&me.id, consumed);
    sys.journal.record(&me.id, &Event::Input { content: json!(blocks) });
    history.push(json!({"role": "user", "content": blocks}));
    true
}

async fn execute_tool(
    sys: &Arc<System>,
    me: &Meta,
    name: &str,
    input: &Value,
    history: &[Value],
) -> (String, bool) {
    match name {
        "spawn_process" => {
            let Some(instructions) = input["instructions"].as_str() else {
                return ("spawn_process requires an 'instructions' string.".into(), true);
            };
            let inherited = match context_mode(input, history) {
                Ok(inherited) => inherited,
                Err(e) => return (e, true),
            };
            let child = NodeSpec {
                instructions: instructions.to_string(),
                name: input["name"].as_str().map(String::from),
                persona: input["role"].as_str().map(String::from),
                inherited,
                wants: grant_spec(input),
                link: input["link"].as_bool().unwrap_or(true),
                model: input["model"].as_str().map(String::from),
                effort: input["effort"].as_str().map(String::from),
                kind: match input["script"].as_str() {
                    Some(source) => Kind::Script(source.to_string()),
                    None => Kind::Agent,
                },
                aliases: match alias_specs(input) {
                    Ok(aliases) => aliases,
                    Err(e) => return (e, true),
                },
            };
            let shown = child.name.clone().map(|n| format!(" ({n})")).unwrap_or_default();
            // Naming the contracts back is what makes the pattern stick: the
            // parent re-reads this line on every later turn, so what it sees is
            // the shape it repeats.
            let contracts: Vec<String> = child.aliases.iter().map(|a| a.name.clone()).collect();
            let mute = child.wants.send.as_ref().is_some_and(|to| to.is_empty());
            let id = match sys.spawn(&me.id, child) {
                Ok(id) => id,
                Err(e) => return (e, true),
            };
            let reach = if mute {
                // Worth stating plainly. A parent that thinks it is owed a
                // reply will otherwise wait for one that cannot arrive.
                " It has no messaging of its own, so it can only answer the tools you gave it — \
                 call one to reach it."
                    .to_string()
            } else {
                format!(" It can reach you at your id, {}.", me.id)
            };
            let tools = if contracts.is_empty() {
                String::new()
            } else {
                format!(" It sees {} as tools.", contracts.join(", "))
            };
            (
                format!("Spawned {id}{shown}. It is now working on your instructions.{tools}{reach}"),
                false,
            )
        }
        "spawn_topology" => spawn_topology(sys, me, input, history),
        "send_message" => {
            let Some(body) = input["message"].as_str() else {
                return ("send_message requires a 'message' string.".into(), true);
            };
            let priority = match input["priority"].as_str().unwrap_or("high") {
                "high" => Priority::High,
                "low" => Priority::Low,
                other => {
                    return (
                        format!("Unknown priority '{other}'. Use \"high\" or \"low\"."),
                        true,
                    );
                }
            };
            // Answering a synchronous call goes back to the blocked caller
            // rather than into a mailbox.
            if let Some(id) = input["in_reply_to"].as_str() {
                if !sys.call_is_pending(id) {
                    return (
                        format!("Nobody is waiting on '{id}' — it already timed out or was answered."),
                        true,
                    );
                }
                sys.resolve_call(id, Ok(body.to_string()));
                return (format!("Answered the call {id}."), false);
            }
            let targets = match resolve_targets(sys, me, input) {
                Ok(targets) => targets,
                Err(e) => return (e, true),
            };
            if targets.len() > 1 {
                ui::trace(&me.tag, &format!("  fan-out to {} recipients", targets.len()));
            }
            return actions::send(sys, me, targets, body, priority);
            #[allow(unreachable_code)]

            let (mut delivered, mut denied, mut failed) = (Vec::new(), Vec::new(), Vec::new());
            for to in targets {
                if to == me.id {
                    denied.push(format!("{to} (yourself)"));
                    continue;
                }
                if !sys.is_visible(me, &to) {
                    // Outside this process's namespace: report it the way a
                    // genuinely absent id would be, rather than confirming
                    // something exists that it should not know about.
                    failed.push(format!("{to} (no such process)"));
                    continue;
                }
                if !me.may(Capability::Send, &to) {
                    denied.push(to);
                    continue;
                }
                match sys.send(
                    &to,
                    Mail {
                        from: me.id.clone(),
                        from_name: me.name.clone(),
                        body: body.to_string(),
                        priority,
                        reply_to: None,
                        seq: 0,
                    },
                ) {
                    Ok(_) => delivered.push(to),
                    Err(e) => failed.push(format!("{to} ({e})")),
                }
            }

            // Report partial success honestly rather than collapsing to one
            // verdict: the sender needs to know exactly who missed out.
            let mut parts = Vec::new();
            if !delivered.is_empty() {
                parts.push(match priority {
                    Priority::High => format!("Delivered to {}.", delivered.join(", ")),
                    Priority::Low => format!(
                        "Queued for {} at low priority — they will not be woken, and will see \
                         this the next time they run.",
                        delivered.join(", ")
                    ),
                });
            }
            if !denied.is_empty() {
                parts.push(format!(
                    "Not permitted: {}. You may only message {}.",
                    denied.join(", "),
                    me.permitted(Capability::Send)
                ));
            }
            if !failed.is_empty() {
                parts.push(format!("Undeliverable: {}.", failed.join(", ")));
            }
            (parts.join(" "), delivered.is_empty())
        }
        "stop_process" => {
            let targets = match resolve_stop_targets(sys, me, input) {
                Ok(targets) => targets,
                Err(e) => return (e, true),
            };
            let (visible, unseen): (Vec<String>, Vec<String>) =
                targets.into_iter().partition(|t| sys.is_visible(me, t));
            let (allowed, denied): (Vec<String>, Vec<String>) =
                visible.into_iter().partition(|t| me.may(Capability::Stop, t));
            if allowed.is_empty() {
                let mut why = Vec::new();
                if !denied.is_empty() {
                    why.push(format!(
                        "Not permitted (you may only stop {}): {}.",
                        me.permitted(Capability::Stop),
                        denied.join(", ")
                    ));
                }
                if !unseen.is_empty() {
                    why.push(format!("Unknown: {}.", unseen.join(", ")));
                }
                return (why.join(" "), true);
            }
            let cascade = input["cascade"].as_bool().unwrap_or(false);
            match sys.stop(&allowed, cascade, Some(&me.id)) {
                Ok(mut ok) => {
                    if !denied.is_empty() {
                        ok.push_str(&format!(
                            " Not permitted (you may only stop {}): {}.",
                            me.permitted(Capability::Stop),
                            denied.join(", ")
                        ));
                    }
                    if !unseen.is_empty() {
                        ok.push_str(&format!(" Unknown: {}.", unseen.join(", ")));
                    }
                    (ok, false)
                }
                Err(err) => (err, true),
            }
        }
        name if me.aliases.iter().any(|a| a.name == name) => {
            let alias = me
                .aliases
                .iter()
                .find(|a| a.name == name)
                .expect("matched above");
            // Validation failures come back as an error result the caller can
            // correct against, rather than being delivered as a bad payload.
            if let Err(e) = validate(&alias.input_schema, input) {
                return (format!("Invalid arguments for '{}': {e}", alias.name), true);
            }
            let body = serde_json::to_string(input).unwrap_or_else(|_| "{}".into());
            call(sys, me, &alias.target, &body, DEFAULT_CALL_TIMEOUT).await
        }
        "call_process" => {
            let (Some(to), Some(message)) = (
                input["process_id"].as_str().or_else(|| input["to"].as_str()),
                input["message"].as_str(),
            ) else {
                return ("call_process requires 'process_id' and 'message'.".into(), true);
            };
            if to == me.id {
                return ("You cannot call yourself.".into(), true);
            }
            if !sys.is_visible(me, to) {
                return (format!("No process with id '{to}'."), true);
            }
            if !me.may(Capability::Send, to) {
                return (
                    format!(
                        "Not permitted: you may only message {}.",
                        me.permitted(Capability::Send)
                    ),
                    true,
                );
            }
            let seconds = input["timeout_seconds"]
                .as_u64()
                .unwrap_or(DEFAULT_CALL_TIMEOUT)
                .clamp(1, MAX_CALL_TIMEOUT);

            call(sys, me, to, message, seconds).await
        }
        "run_script" => {
            let Some(source) = input["script"].as_str() else {
                return ("run_script requires a 'script' string.".into(), true);
            };
            let seconds = input["timeout_seconds"]
                .as_u64()
                .unwrap_or(DEFAULT_CALL_TIMEOUT)
                .clamp(1, MAX_CALL_TIMEOUT);
            ui::trace(me_tag(me), "  … running inline TypeScript");
            crate::script::run_inline(sys.clone(), me.clone(), source.to_string(), seconds).await
        }
        "patch_script" => {
            let (Some(target), Some(source)) =
                (input["process_id"].as_str(), input["script"].as_str())
            else {
                return ("patch_script requires 'process_id' and 'script'.".into(), true);
            };
            if !sys.is_visible(me, target) {
                return (format!("No process with id '{target}'."), true);
            }
            // Replacing what a process runs is at least as powerful as killing
            // it, and creates code, so it takes both authorities.
            if !me.may(Capability::Stop, target) {
                return (
                    format!(
                        "Not permitted: you may only stop {}, and replacing a process's code \
                         requires that authority over it.",
                        me.permitted(Capability::Stop)
                    ),
                    true,
                );
            }
            if !me.grants.spawn.is_permissive() {
                return ("Not permitted: you do not hold the spawn capability.".into(), true);
            }
            match sys.patch_script(target, source.to_string()) {
                Ok(ok) => (ok, false),
                Err(e) => (e, true),
            }
        }
        "list_processes" => (sys.list_for(me), false),
        other => (format!("Unknown tool: {other}"), true),
    }
}

fn spawn_topology(sys: &Arc<System>, me: &Meta, input: &Value, history: &[Value]) -> (String, bool) {
    let Some(entries) = input["processes"].as_array() else {
        return ("spawn_topology requires a 'processes' array.".into(), true);
    };
    if entries.is_empty() {
        return ("spawn_topology needs at least one process.".into(), true);
    }
    if entries.len() > MAX_GROUP {
        return (
            format!("Too many processes: {} requested, limit is {MAX_GROUP}.", entries.len()),
            true,
        );
    }

    let mut nodes = Vec::with_capacity(entries.len());
    let mut seen: Vec<String> = Vec::new();
    for entry in entries {
        let (Some(name), Some(instructions)) =
            (entry["name"].as_str(), entry["instructions"].as_str())
        else {
            return (
                "Every process in a topology needs a 'name' and 'instructions'.".into(),
                true,
            );
        };
        if name == "user" || name == "parent" {
            return (
                format!("'{name}' is a reserved target name and cannot be used as a process name."),
                true,
            );
        }
        if seen.iter().any(|s| s == name) {
            return (format!("Duplicate process name '{name}' in this topology."), true);
        }
        seen.push(name.to_string());

        let inherited = match context_mode(entry, history) {
            Ok(inherited) => inherited,
            Err(e) => return (e, true),
        };
        // Omitted messaging permission defaults to "can report back to me" so
        // results always have somewhere to go; an explicit [] means isolated.
        let mut wants = grant_spec(entry);
        if wants.send.is_none() {
            wants.send = Some(vec!["parent".to_string()]);
        }

        nodes.push(NodeSpec {
            instructions: instructions.to_string(),
            name: Some(name.to_string()),
            persona: entry["role"].as_str().map(String::from),
            inherited,
            wants,
            link: entry["link"].as_bool().unwrap_or(true),
            model: entry["model"].as_str().map(String::from),
            effort: entry["effort"].as_str().map(String::from),
            kind: match entry["script"].as_str() {
                Some(source) => Kind::Script(source.to_string()),
                None => Kind::Agent,
            },
            aliases: match alias_specs(entry) {
                Ok(aliases) => aliases,
                Err(e) => return (e, true),
            },
        });
    }

    match sys.spawn_group(&me.id, nodes) {
        Ok(launched) => {
            let wiring = launched
                .iter()
                .map(|(name, id)| format!("{name} = {id}"))
                .collect::<Vec<_>>()
                .join(", ");
            ui::trace(&me.tag, &format!("⇉ topology: {wiring}"));
            (
                format!(
                    "Spawned topology: {wiring}. Each process knows which peers it may message; \
                     they can reach you at {}.",
                    me.id
                ),
                false,
            )
        }
        Err(e) => (e, true),
    }
}

/// Anywhere a tool names processes it accepts one id, a list of ids, or "*".
/// This reads that shape; what "*" expands to is the caller's decision, since
/// it differs by verb.
enum Ids {
    Star,
    List(Vec<String>),
}

fn read_ids(input: &Value, fields: &[&str], verb: &str) -> Result<Ids, String> {
    let raw = fields
        .iter()
        .map(|field| &input[*field])
        .find(|value| !value.is_null())
        .ok_or_else(|| {
            format!(
                "{verb} requires '{}': a process id, a list of ids, or \"*\".",
                fields[0]
            )
        })?;

    let list = match raw {
        Value::String(one) if one == "*" => return Ok(Ids::Star),
        Value::String(one) => vec![one.clone()],
        Value::Array(items) => {
            let mut out = Vec::new();
            for item in items {
                match item.as_str() {
                    Some("*") => return Ok(Ids::Star),
                    Some(id) => out.push(id.to_string()),
                    None => {
                        return Err(format!(
                            "Every entry in '{}' must be a process id string.",
                            fields[0]
                        ));
                    }
                }
            }
            out
        }
        _ => {
            return Err(format!(
                "{verb} requires '{}': a process id, a list of ids, or \"*\".",
                fields[0]
            ));
        }
    };

    if list.is_empty() {
        return Err(format!("'{}' resolved to an empty list.", fields[0]));
    }
    // Deduplicate while preserving order, so a repeated id can't act twice.
    let mut seen = std::collections::HashSet::new();
    Ok(Ids::List(
        list.into_iter().filter(|id| seen.insert(id.clone())).collect(),
    ))
}

/// Recipients of a `send_message`.
fn resolve_targets(sys: &Arc<System>, me: &Meta, input: &Value) -> Result<Vec<String>, String> {
    // `to` is the documented field; `process_id` stays accepted so the older
    // single-recipient spelling keeps working.
    match read_ids(input, &["to", "process_id"], "send_message")? {
        Ids::List(ids) => Ok(ids),
        Ids::Star => Ok(match me.granted_ids(Capability::Send) {
            // Restricted: exactly the peers this process was granted.
            Some(ids) => ids,
            // Unrestricted: every other live process it can see. Deliberately
            // excludes "user", so a broadcast can't spam the human's console.
            None => sys
                .live_ids(&me.id)
                .into_iter()
                .filter(|id| sys.is_visible(me, id))
                .collect(),
        }),
    }
}

/// Subjects of a `stop_process`. "*" means everything this process is allowed
/// to stop *other than itself* — a coordinator can wipe its workers and then
/// decide separately whether to exit.
fn resolve_stop_targets(
    sys: &Arc<System>,
    me: &Meta,
    input: &Value,
) -> Result<Vec<String>, String> {
    match read_ids(input, &["targets", "process_id"], "stop_process")? {
        Ids::List(ids) => Ok(ids),
        Ids::Star => {
            // Everything this process may stop, other than itself: a
            // coordinator wipes its workers, then decides separately whether
            // to exit.
            let mut ids = match me.granted_ids(Capability::Stop) {
                Some(ids) => ids,
                None => sys
                    .live_ids(&me.id)
                    .into_iter()
                    .filter(|id| sys.is_visible(me, id))
                    .collect(),
            };
            ids.retain(|id| id != &me.id);
            if ids.is_empty() {
                return Err(format!(
                    "\"*\" resolves to nothing you may stop (you may stop {}).",
                    me.permitted(Capability::Stop)
                ));
            }
            Ok(ids)
        }
    }
}

/// Send and block until the answer arrives. Shared by `call_process` and by
/// every tool alias, so a typed alias and a raw call behave identically.
async fn call(
    sys: &Arc<System>,
    me: &Meta,
    to: &str,
    body: &str,
    seconds: u64,
) -> (String, bool) {
    let (id, rx) = sys.register_call(&me.id, to);
    if let Err(e) = sys.send(
        to,
        Mail {
            from: me.id.clone(),
            from_name: me.name.clone(),
            body: body.to_string(),
            priority: Priority::High,
            reply_to: Some(id.clone()),
            seq: 0,
        },
    ) {
        sys.resolve_call(&id, Err(e.clone()));
        return (e, true);
    }
    ui::trace(&me.tag, &format!("  … waiting on {to} (up to {seconds}s)"));
    match tokio::time::timeout(std::time::Duration::from_secs(seconds), rx).await {
        Ok(Ok(result)) => match result {
            Ok(value) => (value, false),
            Err(e) => (e, true),
        },
        Ok(Err(_)) => ("The process ended without replying.".into(), true),
        Err(_) => {
            sys.resolve_call(&id, Err("timed out".into()));
            (
                format!("{to} did not reply within {seconds}s; the answer is lost."),
                true,
            )
        }
    }
}

/// A deliberately small JSON Schema check: enough to catch the mistakes a
/// model actually makes (missing field, wrong type, value outside an enum)
/// without pretending to be a full validator.
fn validate(schema: &Value, input: &Value) -> Result<(), String> {
    if let Some(required) = schema["required"].as_array() {
        for field in required.iter().filter_map(|f| f.as_str()) {
            if input[field].is_null() {
                return Err(format!("missing required field '{field}'"));
            }
        }
    }
    let Some(properties) = schema["properties"].as_object() else {
        return Ok(());
    };
    for (field, spec) in properties {
        let value = &input[field];
        if value.is_null() {
            continue;
        }
        if let Some(expected) = spec["type"].as_str() {
            let ok = match expected {
                "string" => value.is_string(),
                "number" => value.is_number(),
                "integer" => value.is_i64() || value.is_u64(),
                "boolean" => value.is_boolean(),
                "array" => value.is_array(),
                "object" => value.is_object(),
                _ => true,
            };
            if !ok {
                return Err(format!("field '{field}' should be a {expected}"));
            }
        }
        if let Some(allowed) = spec["enum"].as_array() {
            if !allowed.contains(value) {
                return Err(format!(
                    "field '{field}' must be one of {}",
                    Value::Array(allowed.clone())
                ));
            }
        }
    }
    Ok(())
}

/// Read `tools` off a spawn spec: named, schema-typed aliases that are really
/// calls to another process.
fn alias_specs(spec: &Value) -> Result<Vec<ToolAlias>, String> {
    let Some(items) = spec["tools"].as_array() else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for item in items {
        let (Some(name), Some(description), Some(target)) = (
            item["name"].as_str(),
            item["description"].as_str(),
            item["target"].as_str(),
        ) else {
            return Err("Every entry in 'tools' needs 'name', 'description' and 'target'.".into());
        };
        if RESERVED_TOOLS.contains(&name) {
            return Err(format!("'{name}' is a built-in tool and cannot be used as an alias."));
        }
        let schema = match &item["input_schema"] {
            Value::Null => json!({"type": "object", "properties": {}}),
            schema => schema.clone(),
        };
        out.push(ToolAlias {
            name: name.to_string(),
            description: description.to_string(),
            input_schema: schema,
            target: target.to_string(),
        });
    }
    Ok(out)
}

const RESERVED_TOOLS: [&str; 7] = [
    "spawn_process",
    "spawn_topology",
    "send_message",
    "call_process",
    "patch_script",
    "stop_process",
    "list_processes",
];

/// Read the capability fields common to both spawn tools. A missing field
/// means "inherit from me"; an empty list means "nobody". Everything is
/// clamped to the spawner's own grants during resolution, so a request can
/// only ever narrow.
fn grant_spec(spec: &Value) -> GrantSpec {
    let list = |value: &Value| -> Option<Vec<String>> {
        value.as_array().map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(String::from))
                .collect()
        })
    };
    GrantSpec {
        send: list(&spec["can_send_to"]),
        stop: list(&spec["can_stop"]),
        spawn: spec["can_spawn"].as_bool(),
        run: list(&spec["can_run"]),
        net: list(&spec["can_net"]),
        env: list(&spec["can_env"]),
        sys: list(&spec["can_sys"]),
        read: list(&spec["can_read"]),
        write: list(&spec["can_write"]),
    }
}

/// Read a node's `context` field: "empty" (default) or "inherit".
fn context_mode(spec: &Value, history: &[Value]) -> Result<Option<String>, String> {
    match spec["context"].as_str().unwrap_or("empty") {
        "empty" => Ok(None),
        "inherit" => Ok(Some(render_transcript(history))),
        other => Err(format!(
            "Unknown context mode '{other}'. Use \"empty\" (a clean slate) or \"inherit\" (a \
             snapshot of your conversation so far)."
        )),
    }
}

/// Flatten a conversation into a readable transcript. Deliberately lossy: the
/// child reads this as *background*, not as its own turns, so thinking blocks
/// are dropped and tool traffic is summarized. Rendering to text also avoids
/// replaying another process's tool_use/tool_result pairs and thinking
/// signatures, which the API would reject.
fn render_transcript(history: &[Value]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for message in history {
        let role = message["role"].as_str().unwrap_or("");
        let Some(blocks) = message["content"].as_array() else {
            continue;
        };
        for block in blocks {
            match block["type"].as_str().unwrap_or("") {
                "text" => {
                    let text = block["text"].as_str().unwrap_or("").trim();
                    if !text.is_empty() {
                        lines.push(format!("{role}: {}", truncate(text, MAX_BLOCK_CHARS)));
                    }
                }
                "tool_use" => lines.push(format!(
                    "assistant: [called {} with {}]",
                    block["name"].as_str().unwrap_or("?"),
                    truncate(&block["input"].to_string(), 500)
                )),
                "tool_result" => lines.push(format!(
                    "tool result: {}",
                    truncate(block["content"].as_str().unwrap_or(""), 1_000)
                )),
                // thinking / redacted_thinking: the spawner's private reasoning.
                _ => {}
            }
        }
    }

    let transcript = lines.join("\n");
    if transcript.len() <= MAX_TRANSCRIPT_CHARS {
        return transcript;
    }
    // Keep the tail — the most recent context is the most relevant.
    let cut = transcript.len() - MAX_TRANSCRIPT_CHARS;
    let boundary = transcript[cut..]
        .find('\n')
        .map(|i| cut + i + 1)
        .unwrap_or(cut);
    format!(
        "[earlier context truncated]\n{}",
        &transcript[boundary..]
    )
}

fn tool_definitions(me: &Meta) -> Value {
    // Tools a process cannot use are omitted rather than advertised and then
    // refused. A schema in the tool list is far more persuasive than a line of
    // prose saying the capability is missing, so leaving them in gets a
    // process to spend a turn discovering what it should have been told.
    //
    // The always-present tools are emitted first and carry the cache
    // breakpoint, so every process still shares that prefix no matter which
    // optional tools follow. Capability-shaped variation costs one cache entry
    // per shape, not per process.
    const ALWAYS: [&str; 2] = ["send_message", "list_processes"];
    let permitted = |name: &str| match name {
        "send_message" => me.grants.send.is_permissive(),
        // A roster is only useful to a process that can act on a name other
        // than its own — every process may stop itself, so that alone is not a
        // reason to hand one out. A process that can name no one else has
        // nothing to do with a list of ids except learn that a graph exists,
        // which is exactly what a process holding only tools should not learn.
        "list_processes" => {
            me.grants.send.is_permissive()
                || match &me.grants.stop {
                    Grant::All => true,
                    Grant::Nobody => false,
                    Grant::Ids(ids) => ids.iter().any(|id| *id != me.id),
                }
        }
        // Running code inline is the same authority as creating a process to
        // run it, so it rides on the same capability.
        "spawn_process" | "spawn_topology" | "run_script" => me.grants.spawn.is_permissive(),
        "stop_process" => me.grants.stop.is_permissive(),
        // Replacing a script's code needs authority over the process and the
        // right to create code in the first place.
        "patch_script" => me.grants.spawn.is_permissive() && me.grants.stop.is_permissive(),
        "call_process" => me.grants.send.is_permissive(),
        _ => true,
    };

    let (mut always, mut optional) = (Vec::new(), Vec::new());
    for tool in base_tools().as_array().into_iter().flatten() {
        let name = tool["name"].as_str().unwrap_or("");
        if !permitted(name) {
            continue;
        }
        if ALWAYS.contains(&name) {
            always.push(tool.clone());
        } else {
            optional.push(tool.clone());
        }
    }
    // A process holding none of the always-present tools is myopic by
    // construction: it sees its own aliases and nothing else, so there is no
    // shared prefix left to mark.
    let boundary = always.len().checked_sub(1);
    always.extend(optional);
    if let Some(i) = boundary {
        always[i]["cache_control"] = json!({"type": "ephemeral"});
    }
    for alias in &me.aliases {
        // Naming the process that answers is orientation for a holder that can
        // already see it, and a graph leak for one that cannot. A myopic
        // process should learn that it has a tool, not that it has a colleague.
        let answered_by = me
            .grants
            .send
            .permits(&alias.target)
            .then(|| format!("Answered by {}. ", alias.target))
            .unwrap_or_default();
        always.push(json!({
            "name": alias.name,
            "description": format!(
                "{} ({answered_by}Arguments are validated against this schema before \
                 delivery, and the reply comes back inside this turn.)",
                alias.description
            ),
            "input_schema": alias.input_schema,
        }));
    }
    Value::Array(always)
}

fn base_tools() -> Value {
    json!([
        {
            "name": "spawn_process",
            "description": "Start a single new process (actor). It runs concurrently, receives your instructions as its first mailbox message, and can message you back at your process id. Returns the new process id. Spawn processes for genuinely independent or parallel work; handle quick things yourself. For several processes that need to talk to each other, use spawn_topology instead.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "instructions": {
                        "type": "string",
                        "description": "The task briefing for the new process. Be complete and self-contained."
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional short name for the process, e.g. 'researcher'."
                    },
                    "role": {
                        "type": "string",
                        "description": "Optional system prompt for the process: who it is, its expertise, standards, and output style. This shapes its behavior for its whole life, so put durable identity here and the specific task in 'instructions'."
                    },
                    "context": {
                        "type": "string",
                        "enum": ["empty", "inherit"],
                        "description": "\"empty\" (default): the process starts with a clean slate and knows only what you put in 'instructions' and 'role'. \"inherit\": it also receives a read-only transcript of your conversation so far as background. Inherit when shared history matters and would be tedious to restate; stay empty for focused, cheap, independent work."
                    },
                    "link": {
                        "type": "boolean",
                        "description": "Link the new process to you (default true). While linked, you receive an exit signal if it dies or stalls unexpectedly — this is how you find out that a worker you delegated to is gone instead of waiting forever. Pass false for fire-and-forget work whose failure you don't need to hear about."
                    },
                    "can_send_to": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict who it may message: ids of running processes, 'parent' (you), 'user' (the human console). Omit to give it the same reach you have. Pass [] to isolate it entirely — combined with `tools`, and with can_stop: [] and can_spawn: false, that is the strongest form of this whole idea: the process keeps every tool you gave it and loses send_message, list_processes and the spawn tools outright, so it cannot see, name, or route around anything. Prefer it whenever the worker's job is a function of its arguments. Such a worker cannot report back, so call it if you need its answer."
                    },
                    "script": {
                        "type": "string",
                        "description": "TypeScript source. Supplying this makes a *script* process instead of an agent: deterministic code with the same mailbox, links, permissions and namespace, but costing no API tokens at all. Use it for the mechanical parts of a topology — routing, aggregating, validating, counting, reformatting — and keep agents for work that genuinely needs judgment. The script registers a handler and gets the same operations you have:\n\n  bitty.onMail(async (mail, api) => {\n    api.log(`from ${mail.from}: ${mail.body}`);\n    await api.send(api.parent, \"done\", \"low\");\n  });\n\nOn the api object: send(to, message, priority?), stop(targets, cascade?), list(), log(text), fs.read/write/list/mkdir/remove(path), exec(program, args, cwd), fetch(url, opts), env(name), sys(key), plus id, name, parent, instructions. The standard Deno namespace also works — Deno.readTextFile, Deno.writeTextFile, Deno.mkdir, Deno.remove, Deno.readDir, Deno.cwd, Deno.env.get, new Deno.Command(...).output(), and global fetch — all routed through the same capability checks. The script is typechecked when you spawn it, so errors come back to you rather than failing later. Permission errors throw. 'instructions' is passed to the script as api.instructions rather than being interpreted."
                    },
                    "tools": {
                        "type": "array",
                        "description": "Named tools this process should see that are really calls to another process. Each gives it a real contract — a name, a description, a validated argument schema — instead of composing free text and hoping. Arguments are checked before delivery and the reply arrives inside its turn. A tool may only target a process it is already permitted to message, so this cannot hand out reach its permissions do not.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Tool name it will call, e.g. 'lookup_symbol'. Cannot shadow a built-in."},
                                "description": {"type": "string", "description": "What the tool does and when to use it — this is what it reads to decide."},
                                "input_schema": {"type": "object", "description": "JSON Schema for the arguments. Checked for required fields, property types and enums before anything is sent."},
                                "target": {"type": "string", "description": "The process that answers: a name from this spawn group, a running process id, 'parent', or 'self'."}
                            },
                            "required": ["name", "description", "target"]
                        }
                    },
                    "can_read": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Directories this process may read, e.g. [\"/repo/src\"]. Defaults to yours. Pass [] for none. Paths are canonicalized, so '..' and symlinks cannot escape a root, and you can never grant a directory outside your own."
                    },
                    "can_write": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Directories this process may write. Defaults to yours; grant the narrowest set the work needs, and prefer read-only for anything that only inspects."
                    },
                    "model": {
                        "type": "string",
                        "description": "Model for this process — a Claude model id such as claude-opus-5, claude-sonnet-5 or claude-haiku-4-5. Use a smaller one for mechanical work. Defaults to yours, so a cheap worker's own helpers stay cheap. Choosing the right size here saves far more than trimming tokens does."
                    },
                    "effort": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "xhigh", "max"],
                        "description": "Reasoning effort for this process. Defaults to yours. Use low for mechanical or well-specified work and reserve high effort for genuinely hard reasoning."
                    },
                    "can_stop": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Which processes it may stop: names from this spawn group, ids of running processes, 'self', 'parent'. Omit and it defaults to stopping only itself when you restrict its messaging, or to inheriting your own stop permission when you don't. Pass [] to forbid stopping entirely."
                    },
                    "can_spawn": {
                        "type": "boolean",
                        "description": "Whether it may spawn processes of its own (defaults to whatever you hold). Pass false for a leaf worker that must not fan out further."
                    }
                },
                "required": ["instructions"]
            }
        },
        {
            "name": "spawn_topology",
            "description": "Spawn several processes at once as a wired group: each gets its own role, its own starting context, and an explicit allowlist of peers it may message. Processes are created together, so they can reference each other by name — use this to build pipelines (a → b → c), fan-out/fan-in, or reviewer pairs. A process may only send to the targets you grant it; everything else is rejected at runtime. Restricted processes can only stop themselves, not each other.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "processes": {
                        "type": "array",
                        "description": "The nodes of the topology, up to 16.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Short unique name within this topology, e.g. 'writer'. Peers reference it by this name."},
                                "instructions": {"type": "string", "description": "The task briefing for this process."},
                                "role": {"type": "string", "description": "Optional system prompt: who this process is and how it should work."},
                                "context": {"type": "string", "enum": ["empty", "inherit"], "description": "\"empty\" (default) or \"inherit\" a transcript of your conversation so far."},
                                "can_send_to": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Who this process may message: sibling names from this topology, ids of already-running processes, \"parent\" (you), and/or \"user\" (the human console). Omit to default to [\"parent\"]. Pass [] for a process that reports to no one. Grant only what the work requires — this is the wiring of your topology."
                                },
                                "link": {"type": "boolean", "description": "Link this process to you (default true), so you get an exit signal if it dies or stalls. Siblings are never linked to each other: if one dies, only you are told, and relaying that to the others is your job."},
                                "can_stop": {"type": "array", "items": {"type": "string"}, "description": "Which processes it may stop: sibling names, running process ids, 'self', 'parent'. Defaults to itself only. Pass [] to forbid stopping."},
                                "can_spawn": {"type": "boolean", "description": "Whether it may spawn processes of its own. Defaults to whatever you hold; pass false for a leaf worker."},
                                "script": {"type": "string", "description": "TypeScript source; makes this node a deterministic script process costing no API tokens. It registers bitty.onMail((mail, api) => ...) and gets send/stop/list/log under the same permissions as any other process."},
                                "tools": {"type": "array", "description": "Named, schema-typed tools for this node that are really calls to another process (see spawn_process). Give a worker a typed tool instead of telling it to message a peer in prose.", "items": {"type": "object", "properties": {"name": {"type": "string"}, "description": {"type": "string"}, "input_schema": {"type": "object"}, "target": {"type": "string"}}, "required": ["name", "description", "target"]}},
                                "can_read": {"type": "array", "items": {"type": "string"}, "description": "Directories this node may read. Defaults to yours; narrow it to what the role needs."},
                                "can_write": {"type": "array", "items": {"type": "string"}, "description": "Directories this node may write. Prefer [] for reviewers and analyzers."},
                                "model": {"type": "string", "description": "Model for this node — use a smaller one for mechanical work. Defaults to yours."},
                                "effort": {"type": "string", "enum": ["low", "medium", "high", "xhigh", "max"], "description": "Reasoning effort for this node. Defaults to yours; use low for well-specified mechanical work."}
                            },
                            "required": ["name", "instructions"]
                        }
                    }
                },
                "required": ["processes"]
            }
        },
        {
            "name": "send_message",
            "description": "Send a free-form text message to one or more process mailboxes. If a recipient is mid-task it sees the message between its tool calls; if idle, the message wakes it. The special id \"user\" prints on the human's console. To fan out, pass a list of ids or \"*\" (everyone you may message) — the body is written once no matter how many recipients. Fan out deliberately: each delivery wakes an idle recipient and costs it a full turn of thinking, so a broadcast to eight processes is eight times the work of one message.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "to": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}}
                        ],
                        "description": "One process id ('proc-2'), 'user', a list of ids, or \"*\" for every process you are permitted to message. \"*\" never includes the human console."
                    },
                    "message": {"type": "string"},
                    "in_reply_to": {
                        "type": "string",
                        "description": "Answer a call_process that is blocked waiting on you. Take this id from the reply_to attribute on the incoming message. The body goes straight back to the caller instead of into a mailbox."
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "low"],
                        "description": "\"high\" (default) wakes an idle recipient immediately, which costs it a full turn of thinking. \"low\" never wakes anyone: the message is held and delivered the next time that process runs for some other reason, so it costs only its own tokens. Use low for status updates, FYIs, and anything the recipient does not need to act on right now; use high when they must respond or change course."
                    }
                },
                "required": ["to", "message"]
            }
        },
        {
            "name": "stop_process",
            "description": "Permanently stop one or more processes. Their current work is aborted, they can no longer receive mail, and they cannot be restarted (spawn new ones instead). You may stop yourself — your final act, after any last send_message calls. Stopping a process you are linked to sends you an exit signal only if someone else stopped it; stopping yourself is a clean exit and signals nobody.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "targets": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}}
                        ],
                        "description": "One process id ('proc-2'), a list of ids, your own id to stop yourself, or \"*\" for every live process except you — the way to clean up all your workers at once."
                    },
                    "cascade": {"type": "boolean", "description": "Also stop every descendant of each target, transitively. Default false."}
                },
                "required": ["targets"]
            }
        },
        {
            "name": "call_process",
            "description": "Send a message to one process and wait for its answer, returning that answer as this tool's result. Unlike send_message, the reply arrives inside the current turn, so you can use it to decide what to do next — this is what makes a script process usable as a function rather than only as a collaborator. A script answers with whatever its handler returns; an agent answers by calling send_message with in_reply_to. Costs one model request, the same as any tool call.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "process_id": {"type": "string", "description": "The process to call. One target only — this is a request/response, not a broadcast."},
                    "message": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "description": "How long to wait, 1-300, default 60. On timeout the call fails and the answer is discarded even if it arrives later."}
                },
                "required": ["process_id", "message"]
            }
        },
        {
            "name": "run_script",
            "description": "Run TypeScript once, right now, and get its value back as this tool's result. It executes with your own capabilities — the same files, hosts, programs and environment you can reach — so use it for work you want done rather than delegated: computing, parsing, reshaping data, reading a handful of files, checking something on disk. The last expression you return is the answer; throwing returns the error. No process is created, so there is nothing to message or clean up afterwards. Prefer this over spawning a script process for anything that ends immediately; spawn a process when it needs to persist, hold state, or receive messages.\n\nExample: return Deno.readTextFile(path).then(t => t.split(\"\\n\").length);",
            "input_schema": {
                "type": "object",
                "properties": {
                    "script": {"type": "string", "description": "TypeScript. Typechecked before it runs; errors come back to you. Use `return` for the value, and the same api/Deno surface a script process has."},
                    "timeout_seconds": {"type": "integer", "description": "1-300, default 60."}
                },
                "required": ["script"]
            }
        },
        {
            "name": "patch_script",
            "description": "Replace the code running in a script process, keeping its id, mailbox, links and permissions. Prefer this over stopping and respawning whenever anything is wired to the process: permissions are resolved to concrete ids at spawn, so a replacement gets a new id that its peers' allowlists do not name, and their wiring cannot be updated afterward. The running code is replaced with a fresh isolate, so any state the old code held is gone; identity and position in the graph survive. Only script processes have code to replace.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "process_id": {"type": "string"},
                    "script": {"type": "string", "description": "The complete new TypeScript source. Not a diff — it replaces the old source entirely."}
                },
                "required": ["process_id", "script"]
            }
        },
        {
            "name": "list_processes",
            "description": "List every process in the system with its id, name, status (running/idle/stopped), and parent.",
            "input_schema": {"type": "object", "properties": {}}
        }
    ])
}

/// The half of the system prompt that is byte-identical for every process in
/// the run. It is emitted first and carries a cache breakpoint, so the whole
/// tools + scaffolding prefix is written to cache once and read by every other
/// process at a tenth of the price. Nothing process-specific may appear here —
/// a single interpolated id would fork the prefix and destroy the sharing.
const SHARED_PREAMBLE: &str = "\
You are one process in Bitty, an actor-style multi-agent system. Every process has a unique id, \
a private mailbox, and its own conversation with the model. Processes share nothing else: no \
common history, no shared memory, only messages.

How the system works:
- Messages other processes send you are delivered into this conversation inside \
<incoming_message from=\"...\"> tags. They can arrive at any moment, including between your tool \
calls mid-task. Treat them like interruptions from a collaborator: read them when they appear \
and factor them into what you do next.
- When you end a turn without calling a tool, you go idle until the next mailbox message wakes \
you. Going idle is normal — end your turn when you have nothing left to do.
- Message text is free-form. Be concise and information-dense.";

/// The half of the preamble that only means anything to a process that can
/// reach or create other processes. A process holding nothing but tools is not
/// shown it: telling it how a graph works, while giving it no tool that touches
/// one, is the exact confusion the tool list was filtered to avoid. Two shapes
/// means two cache entries, not one per process.
const GRAPH_PREAMBLE: &str = "\
- Processes you spawn are linked to you by default, like spawn_link in an actor framework. If a \
linked process dies unexpectedly or stalls, the harness mails you an <exit_signal> from \
\"system\". That is how you find out a worker is gone instead of waiting for a reply that will \
never come — act on it rather than continuing to wait. Links are only ever between a process and \
the one that spawned it: siblings are not linked to each other, so if one dies, only their \
spawner is told, and relaying that to anything else that depended on it is the spawner's job.
- send_message takes one id, a list of ids, or \"*\" for everyone you may message. Prefer one \
targeted message over a broadcast: waking a recipient costs it a full turn of thinking, so fan \
out only when every recipient genuinely needs the message. Send at priority \"low\" for status \
updates and anything the recipient need not act on immediately — low-priority mail never wakes \
anyone, it just rides along the next time they run, so it is close to free.
- You can give a process you spawn its own tools: named, schema-typed entries that are really \
calls to another process. Prefer that over telling a worker in prose to message a peer — it turns \
an informal convention into a checked contract, and the worker sees it as an ordinary tool.
- send_message is one-way; call_process waits for an answer and hands it back inside the same \
turn, so you can act on the result immediately. Use call_process when you need the answer to \
decide what to do next, and send_message when you do not. If a message arrives carrying a \
reply_to id, someone is blocked waiting on you: answer with send_message and in_reply_to.
- A process can be a TypeScript script instead of an agent: pass `script` to spawn_process or a \
topology node. Script processes have the same mailbox, links, permissions and namespace, but they \
run deterministic code and cost no API tokens. Prefer one for any node whose job is mechanical — \
routing, aggregating, counting, validating, reformatting — and keep agents for judgment.
- Match the model and effort you give a process to the work it will do. A mechanical, \
well-specified task does not need your model or your effort level, and spawning it smaller is by \
far the biggest saving available to you — much larger than writing shorter messages.
- Spawn processes for genuinely parallel or independent workstreams, not for things you can do \
directly. Brief them completely: they start with zero context beyond what you give them. Use \
spawn_process for one helper, spawn_topology when several processes need defined roles and a \
defined communication graph.
- A spawned process starts with an empty context by default. Pass context: \"inherit\" when it \
genuinely needs your history as background; prefer a well-written briefing over inheriting, \
since inherited context costs tokens on every one of its turns.
- Give a worker tools, not instructions, whenever the thing it needs is another process. Passing \
`tools` on spawn turns an instruction like \"send proc-4 a path and it will reply with the \
contents\" into a read_file tool with a checked argument schema, and the worker never learns a \
graph exists — it just has a tool. That is worth doing even for a single edge: a contract survives \
being handed to a cheaper model, a convention does not.
- Finish that move by passing can_send_to: [], can_stop: [] and can_spawn: false alongside the \
tools. Then it is not a preference but the shape of the world: send_message, list_processes and \
the spawn tools are not in that worker's tool list at all, and it is not told a graph exists — so \
there is no id to guess at, no roster to enumerate, and no way to route around the contract you \
wrote. It keeps every tool you gave it, because a tool carries its own authority rather than \
borrowing the holder's. Prefer this whenever a worker's job is a function of its arguments; it is \
the version that stays true after the worker is handed to a cheaper model or its context is \
compacted. Note that such a worker cannot report back — it acts through its tools and then ends \
its turn — so if you need an answer from it, either call it and let it reply, or leave it \
can_send_to: [\"parent\"].
- Start an edge as an agent because prose is the fastest way to specify judgment, then demote it \
to a script once the judgment turns out to be mechanical. The caller does not change, because it \
only ever knew the tool. Routing, scoring, retrying, formatting and aggregating are all better as \
code than as someone's discretion.
- Grant the least privilege that does the job. A process you spawn inherits your capabilities \
unless you narrow them, and inheriting is rarely what the work needs: a reviewer wants read \
access and no write, a summarizer wants neither, a test runner wants one program and not a \
shell. Name the narrowest can_send_to, can_read, can_write, can_run, can_net and can_env that \
lets it finish, and pass [] rather than omitting the field when the answer is none. This is not \
bureaucracy — a narrow grant is what makes a mistake or a bad instruction survivable, and it \
costs you nothing to write.
- Clean up after yourself. Stop workers you spawned once they're no longer needed (stop_process, \
with cascade: true to take down their descendants too). When your own task is fully done and you \
expect no follow-ups, send your final report first, then stop yourself. Stopping is permanent — \
a stopped id cannot be woken or mailed. If you might get follow-up messages, stay idle instead.
- Message text is free-form. Be concise and information-dense with other processes.";

/// The system prompt as content blocks, ordered general → specific so the
/// shared prefix can be cached across every process in the system.
fn system_blocks(me: &Meta) -> Value {
    // Whether this process can touch anything beyond itself. A process that
    // cannot is told nothing about the graph it lives in.
    let connected = me.grants.send.is_permissive()
        || me.grants.spawn.is_permissive()
        || match &me.grants.stop {
            Grant::All => true,
            Grant::Nobody => false,
            Grant::Ids(ids) => ids.iter().any(|id| *id != me.id),
        };
    let mut blocks = vec![json!({"type": "text", "text": SHARED_PREAMBLE})];
    if connected {
        blocks.push(json!({"type": "text", "text": GRAPH_PREAMBLE}));
    }
    // The breakpoint sits on the last shared block, so every process of the
    // same shape shares the whole prefix and nothing per-process precedes it.
    let last = blocks.len() - 1;
    blocks[last]["cache_control"] = json!({"type": "ephemeral"});
    blocks.push(json!({"type": "text", "text": process_identity(me)}));
    Value::Array(blocks)
}

/// Everything that varies per process, kept strictly after the cache
/// breakpoint: who this process is, its place in the tree, its wiring, and
/// its role.
fn process_identity(me: &Meta) -> String {
    let identity = match &me.name {
        Some(name) => format!("{} (named \"{}\")", me.id, name),
        None => me.id.clone(),
    };
    let role = if me.parent == "user" {
        "You are the root process: the human user talks to you directly, and your plain-text \
         replies stream to their console. Messages arriving from \"user\" are the human typing."
            .to_string()
    } else if me.grants.send.permits(&me.parent) {
        format!(
            "You were spawned by {}. When you finish a task it gave you, report the result back \
             with send_message — ending your turn without reporting means your work is lost.",
            me.parent
        )
    } else if me.grants.send.is_permissive() {
        // Spawned by one process, permitted to answer a different one. Saying
        // so is better than naming a parent it cannot reach.
        format!(
            "You were spawned by {}, which you are not permitted to message. When you finish, \
             report the result with send_message to whoever your permissions do allow — ending \
             your turn without reporting means your work is lost.",
            me.parent
        )
    } else {
        // Nothing to report back with, so asking for a report would only get
        // this process to spend a turn discovering it cannot. Its work is done
        // through its tools; ending the turn is the correct ending.
        "You have no way to message anyone. Do your work through the tools you have and then \
         end your turn — that is the whole of your job, and nothing is expected of you afterward."
            .to_string()
    };

    let wiring = if me.grants.is_unrestricted() {
        String::new()
    } else {
        format!(
            "\n\nYour permissions (enforced by the harness — anything else is rejected):\n{}\n\
             Processes you spawn can never exceed these — asking to grant one more than you hold \
             is rejected, so plan within your own reach.",
            me.grants.describe(&|id| me.label_of(id))
        )
    };

    let persona = match &me.persona {
        Some(persona) => format!("\n\nYour role:\n{persona}"),
        None => String::new(),
    };

    format!("You are process {identity}.\n{role}{wiring}{persona}")
}

fn envelope(mail: &Mail) -> String {
    let name_attr = mail
        .from_name
        .as_ref()
        .map(|n| format!(" name=\"{n}\""))
        .unwrap_or_default();
    let reply_attr = mail
        .reply_to
        .as_ref()
        .map(|id| format!(" reply_to=\"{id}\""))
        .unwrap_or_default();
    format!(
        "<incoming_message from=\"{}\"{}{}>\n{}\n</incoming_message>",
        mail.from, name_attr, reply_attr, mail.body
    )
}

fn text_block(text: &str) -> Value {
    json!({"type": "text", "text": text})
}

/// If a server-side fallback happened mid-output, blocks before the last
/// `fallback` marker must drop thinking/tool_use before being echoed back.
fn sanitize_for_history(content: Vec<Value>) -> Vec<Value> {
    let Some(boundary) = content.iter().rposition(|b| b["type"] == "fallback") else {
        return content;
    };
    content
        .into_iter()
        .enumerate()
        .filter(|(i, block)| {
            *i > boundary
                || !matches!(
                    block["type"].as_str().unwrap_or(""),
                    "thinking" | "redacted_thinking" | "tool_use"
                )
        })
        .map(|(_, block)| block)
        .collect()
}

/// A single readable line for the console, however sprawling the original.
fn me_tag(me: &Meta) -> &crate::ui::Tag {
    &me.tag
}

fn first_line(text: &str) -> String {
    let line = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    let rest = text.lines().filter(|l| !l.trim().is_empty()).count().saturating_sub(1);
    let line = truncate(line.trim(), 160);
    if rest > 0 {
        format!("{line} (+{rest} more lines, sent to the process)")
    } else {
        line
    }
}

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        let cut: String = text.chars().take(max).collect();
        format!("{cut}…")
    }
}
