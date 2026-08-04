//! The operations a process can perform on the system, with capability and
//! visibility policy applied.
//!
//! Both process types go through here — an agent reaching these via tool calls
//! and a Deno script reaching them over its stdout protocol get identical
//! enforcement. Keeping one implementation is the point: two copies of this
//! policy would drift, and the drift would be a privilege bug.

use crate::grants::Capability;
use crate::system::{Kind, Mail, Meta, NodeSpec, Priority, System};
use serde_json::Value;
use std::sync::Arc;

/// Deliver a message to each target. Returns (report, is_error); the report
/// names partial outcomes rather than collapsing to a single verdict.
pub fn send(
    sys: &Arc<System>,
    me: &Meta,
    targets: Vec<String>,
    body: &str,
    priority: Priority,
) -> (String, bool) {
    let (mut delivered, mut denied, mut failed) = (Vec::new(), Vec::new(), Vec::new());
    for to in targets {
        if to == me.id {
            denied.push(format!("{to} (yourself)"));
            continue;
        }
        if !sys.is_visible(me, &to) {
            // Outside this process's namespace: report it the way a genuinely
            // absent id would be, rather than confirming something exists that
            // it should not know about.
            failed.push(format!("{to} (no such process)"));
            continue;
        }
        if !me.may(Capability::Send, &to) {
            denied.push(to);
            continue;
        }
        // If this process is what the recipient is blocked calling, the message
        // is the answer — route it there rather than into a mailbox nobody is
        // in a position to read.
        if let Some(call) = sys.call_awaiting(&to, &me.id) {
            sys.resolve_call(&call, Ok(body.to_string()));
            delivered.push(format!("{to} (answering its call)"));
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

    let mut parts = Vec::new();
    if !delivered.is_empty() {
        parts.push(match priority {
            Priority::High => format!("Delivered to {}.", delivered.join(", ")),
            Priority::Low => format!(
                "Queued for {} at low priority — they will not be woken, and will see this the \
                 next time they run.",
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

/// Stop each target it may see and may stop, reporting the rest.
pub fn stop(sys: &Arc<System>, me: &Meta, targets: Vec<String>, cascade: bool) -> (String, bool) {
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

    match sys.stop(&allowed, cascade, Some(&me.id)) {
        Ok(mut report) => {
            if !denied.is_empty() {
                report.push_str(&format!(
                    " Not permitted (you may only stop {}): {}.",
                    me.permitted(Capability::Stop),
                    denied.join(", ")
                ));
            }
            if !unseen.is_empty() {
                report.push_str(&format!(" Unknown: {}.", unseen.join(", ")));
            }
            (report, false)
        }
        Err(e) => (e, true),
    }
}

/// Create processes. Scripts reach this too, so the mechanical half of a
/// topology — spawning a worker per item, retiring it when done — does not have
/// to cost a model turn just because only agents could call it.
///
/// Capability, attenuation and the group cap are all enforced inside
/// `spawn_group`, so a script is refused exactly what an agent would be, in the
/// same words.
pub fn spawn(sys: &Arc<System>, me: &Meta, nodes: &[Value]) -> (String, bool) {
    let mut specs = Vec::with_capacity(nodes.len());
    for node in nodes {
        let Some(instructions) = node["instructions"].as_str() else {
            return ("each process needs an 'instructions' string".into(), true);
        };
        let aliases = match crate::agent::alias_specs(node) {
            Ok(aliases) => aliases,
            Err(e) => return (e, true),
        };
        specs.push(NodeSpec {
            instructions: instructions.to_string(),
            name: node["name"].as_str().map(String::from),
            persona: node["role"].as_str().map(String::from),
            // A script has no conversation, so there is nothing to inherit.
            inherited: None,
            wants: crate::agent::grant_spec(node),
            link: node["link"].as_bool().unwrap_or(true),
            model: node["model"].as_str().map(String::from),
            effort: node["effort"].as_str().map(String::from),
            kind: match node["script"].as_str() {
                Some(source) => Kind::Script(source.to_string()),
                None => Kind::Agent,
            },
            aliases,
        });
    }
    match sys.spawn_group(&me.id, specs) {
        // spawn_group hands back (label, id) pairs; a script wants the ids.
        Ok(spawned) => (
            spawned
                .iter()
                .map(|(_, id)| id.clone())
                .collect::<Vec<_>>()
                .join(","),
            false,
        ),
        Err(e) => (e, true),
    }
}

/// The namespaced process listing.
pub fn list(sys: &Arc<System>, me: &Meta) -> String {
    sys.list_for(me)
}
