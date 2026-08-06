//! Serialized, color-coded console output. Every process prints through here
//! so concurrent actors never interleave mid-line.
//!
//! The same lines feed the dashboard: once `tap()` has been opened, every
//! print also broadcasts a structured event. The feed additionally carries
//! successful inbound deliveries, which are intentionally invisible in plain
//! mode so its established output does not gain duplicate delivery lines.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

static OUT: Mutex<()> = Mutex::new(());
static DASHBOARD_ACTIVE: AtomicBool = AtomicBool::new(false);

/// One console line, structured, for the dashboard feed.
#[derive(Clone)]
pub struct Event {
    /// "say" | "trace" | "mail" | "incoming" | "warn" | "system" |
    /// "external"
    pub kind: &'static str,
    /// The label shown beside the event: normally its speaker or sender.
    pub who: String,
    /// The process whose view owns this event. For normal output this is the
    /// emitter; for an incoming message it is the recipient.
    pub process: Option<String>,
    pub text: String,
}

static TAP: OnceLock<tokio::sync::broadcast::Sender<Event>> = OnceLock::new();

/// Open (or fetch) the dashboard tap. Subscribers receive every console line
/// from then on; a slow subscriber lags and skips rather than blocking the
/// console.
pub fn tap() -> tokio::sync::broadcast::Sender<Event> {
    TAP.get_or_init(|| tokio::sync::broadcast::channel(512).0)
        .clone()
}

fn feed(kind: &'static str, who: &str, process: Option<&str>, text: &str) {
    if let Some(tx) = TAP.get() {
        let _ = tx.send(Event {
            kind,
            who: who.into(),
            process: process.map(String::from),
            text: text.into(),
        });
    }
}

fn process_from_label(label: &str) -> Option<&str> {
    label
        .split_whitespace()
        .next()
        .filter(|part| part.starts_with("proc-") && part[5..].chars().all(|c| c.is_ascii_digit()))
}

/// Suppress the ANSI stdout mirror while the alternate-screen dashboard owns
/// the terminal. Events still flow through the tap; plain mode never calls
/// this, so its output remains byte-for-byte unchanged.
pub fn set_dashboard_active(active: bool) {
    DASHBOARD_ACTIVE.store(active, Ordering::Relaxed);
}

// One color per process, assigned round-robin at spawn time.
const PALETTE: &[u8] = &[36, 32, 35, 34, 33, 31, 96, 92, 95, 94];

#[derive(Clone)]
pub struct Tag {
    pub label: String,
    pub color: u8,
}

impl Tag {
    pub fn new(label: impl Into<String>, index: u64) -> Self {
        Tag {
            label: label.into(),
            color: PALETTE[(index as usize) % PALETTE.len()],
        }
    }
}

fn emit(kind: &'static str, who: &str, text: &str, line: &str) {
    let _guard = OUT.lock().unwrap();
    // Keep tap order identical to console order. Feeding outside this lock
    // would let two actors race such that the dashboard saw B,A while stdout
    // printed A,B.
    feed(kind, who, process_from_label(who), text);
    if DASHBOARD_ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(stdout, "{line}");
    let _ = stdout.flush();
}

/// Record a successful inbound delivery for the recipient's process view.
/// This is intentionally feed-only: plain mode already reports delivery at
/// the sender/tool layer and must not gain a second line of console output.
pub fn arrival(to: &str, from: &str, text: &str) {
    let _guard = OUT.lock().unwrap();
    feed("incoming", from, Some(to), text);
}

/// A line of assistant text from a process.
pub fn say(tag: &Tag, text: &str) {
    emit(
        "say",
        &tag.label,
        text,
        &format!("\x1b[{}m[{}]\x1b[0m {}", tag.color, tag.label, text),
    );
}

/// Tool calls, deliveries, and other machinery — dimmed.
pub fn trace(tag: &Tag, text: &str) {
    emit(
        "trace",
        &tag.label,
        text,
        &format!(
            "\x1b[{}m[{}]\x1b[0m \x1b[2m{}\x1b[0m",
            tag.color, tag.label, text
        ),
    );
}

/// A message a process sent to the human console — highlighted.
pub fn mail_to_user(from_label: &str, text: &str) {
    emit(
        "mail",
        from_label,
        text,
        &format!("\x1b[1;33m[{from_label} → user]\x1b[0m {text}"),
    );
}

pub fn warn(tag: &Tag, text: &str) {
    emit(
        "warn",
        &tag.label,
        text,
        &format!(
            "\x1b[{}m[{}]\x1b[0m \x1b[31m{}\x1b[0m",
            tag.color, tag.label, text
        ),
    );
}

/// Untagged system-level output (banner, /ps listings, errors).
pub fn system(text: &str) {
    emit("system", "", text, &format!("\x1b[2m{text}\x1b[0m"));
}

/// Output written around the UI layer (for example a dependency using
/// `println!`) is captured while the dashboard owns the terminal. Put it in
/// the transcript rather than allowing it to overwrite ratatui's back buffer.
pub fn external(text: &str) {
    emit("external", "", text, text);
}
