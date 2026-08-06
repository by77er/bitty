//! Serialized, color-coded console output. Every process prints through here
//! so concurrent actors never interleave mid-line.
//!
//! The same lines feed the dashboard: once `tap()` has been opened, every
//! print also broadcasts a structured event. The console remains the source
//! of truth; the dashboard is a mirror, so nothing renders in one place and
//! not the other.

use std::io::Write;
use std::sync::{Mutex, OnceLock};

static OUT: Mutex<()> = Mutex::new(());

/// One console line, structured, for the dashboard feed.
#[derive(Clone)]
pub struct Event {
    /// "say" | "trace" | "mail" | "warn" | "system"
    pub kind: &'static str,
    pub who: String,
    pub text: String,
}

static TAP: OnceLock<tokio::sync::broadcast::Sender<Event>> = OnceLock::new();

/// Open (or fetch) the dashboard tap. Subscribers receive every console line
/// from then on; a slow subscriber lags and skips rather than blocking the
/// console.
pub fn tap() -> tokio::sync::broadcast::Sender<Event> {
    TAP.get_or_init(|| tokio::sync::broadcast::channel(512).0).clone()
}

fn feed(kind: &'static str, who: &str, text: &str) {
    if let Some(tx) = TAP.get() {
        let _ = tx.send(Event { kind, who: who.into(), text: text.into() });
    }
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

fn emit(line: &str) {
    let _guard = OUT.lock().unwrap();
    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(stdout, "{line}");
    let _ = stdout.flush();
}

/// A line of assistant text from a process.
pub fn say(tag: &Tag, text: &str) {
    emit(&format!("\x1b[{}m[{}]\x1b[0m {}", tag.color, tag.label, text));
}

/// Tool calls, deliveries, and other machinery — dimmed.
pub fn trace(tag: &Tag, text: &str) {
    emit(&format!(
        "\x1b[{}m[{}]\x1b[0m \x1b[2m{}\x1b[0m",
        tag.color, tag.label, text
    ));
}

/// A message a process sent to the human console — highlighted.
pub fn mail_to_user(from_label: &str, text: &str) {
    emit(&format!("\x1b[1;33m[{from_label} → user]\x1b[0m {text}"));
}

pub fn warn(tag: &Tag, text: &str) {
    emit(&format!(
        "\x1b[{}m[{}]\x1b[0m \x1b[31m{}\x1b[0m",
        tag.color, tag.label, text
    ));
}

/// Untagged system-level output (banner, /ps listings, errors).
pub fn system(text: &str) {
    emit(&format!("\x1b[2m{text}\x1b[0m"));
}
