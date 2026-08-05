//! Bitty — an actor-style agent harness.
//!
//! Usage:
//!   bitty "initial task for the root process"
//!   bitty --once "task"          # exit when every process is idle or stopped
//!   bitty --role "you are ..."   # system prompt for the root process
//!
//! Permissions, Deno's own convention: bare means unrestricted, a value (as
//! `--allow-X=v1,v2` or `--allow-X v1,v2`) scopes it, and omitting a flag
//! entirely denies that capability. Repeatable; scoped values accumulate.
//!   bitty --allow-read --allow-write            # read/write anywhere
//!   bitty --allow-read=DIR --allow-write=DIR    # only these directories
//!   bitty -A                                    # shorthand for all six, unrestricted
//!   bitty --allow-run=cargo,python3             # only these programs
//!   bitty --allow-net=api.example.com           # only this host
//!   bitty --allow-env=HOME,PATH --allow-sys     # only these vars, any system fact
//!
//!   bitty --resume NAME                        # bring back a session by name
//!   bitty --resume [\"message\"]                 # the most recent one, optionally with a nudge
//!   bitty --journal DIR                        # journal somewhere specific instead
//!
//! Every interactive run is a persisted session under .bitty/sessions/<name>,
//! created automatically and named at startup. `--once` is exempt: a one-shot
//! run has nothing to come back to.
//!
//! Console, while running:
//!   plain text        → mail to the root process (interrupts it mid-task)
//!   @proc-3 message   → mail to a specific process
//!   /ps               → list processes
//!   /graph            → supervision tree + who may message whom
//!   /model proc-1 M [E] → switch a process's model (and effort) in flight
//!   /stop proc-3      → stop a process (add --cascade for its descendants)
//!   /quit             → exit

mod agent;
mod durable;
mod grants;
mod actions;
mod api;
mod codex;
mod system;
mod script;
mod ui;

use std::sync::Arc;

/// Where processes are recorded unless told otherwise.
const DEFAULT_JOURNAL: &str = ".bitty/journal";
/// Each run gets its own directory under here, so sessions accumulate side by
/// side and can be resumed by name rather than by remembering a path.
const SESSION_ROOT: &str = ".bitty/sessions";

/// A name a person can retype from memory an hour later. Two words carry the
/// recall and the epoch tail carries the uniqueness — a bare timestamp is
/// unique too, but nobody remembers which one was theirs.
fn new_session_name() -> String {
    const ADJECTIVES: [&str; 16] = [
        "amber", "brisk", "calm", "dusk", "eager", "fresh", "glad", "hollow", "ivory", "jolly",
        "keen", "lucid", "mellow", "noble", "opal", "plain",
    ];
    const NOUNS: [&str; 16] = [
        "otter", "harbor", "cedar", "falcon", "meadow", "quartz", "raven", "summit", "thicket",
        "vault", "willow", "anchor", "beacon", "cove", "dune", "ember",
    ];
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let base = format!(
        "{}-{}-{:x}",
        ADJECTIVES[(secs / 60) as usize % ADJECTIVES.len()],
        NOUNS[secs as usize % NOUNS.len()],
        secs % 0xffff
    );
    // Two runs in the same second would otherwise share a directory and
    // interleave their journals.
    let mut name = base.clone();
    let mut n = 2;
    while std::path::Path::new(SESSION_ROOT).join(&name).exists() {
        name = format!("{base}-{n}");
        n += 1;
    }
    name
}

/// Load `KEY=value` lines into this process's environment, so a token kept in
/// a file reaches the harness without being exported by hand. Returns the names
/// it set — never the values, which have no business on a console.
///
/// Existing variables win: something already exported is a deliberate act, and
/// a stale file should not silently override it.
fn load_env_file(path: &str) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut loaded = Vec::new();
    for line in text.lines() {
        let line = line.trim().strip_prefix("export ").unwrap_or(line.trim());
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        let name = name.trim();
        let value = value.trim().trim_matches(['"', '\'']);
        if name.is_empty() || std::env::var_os(name).is_some() {
            continue;
        }
        // SAFETY: single-threaded startup, before any process is spawned.
        unsafe { std::env::set_var(name, value) };
        loaded.push(name.to_string());
    }
    Ok(loaded)
}

/// The most recently written session, for a bare `--resume`.
fn latest_session() -> Option<String> {
    let mut best: Option<(std::time::SystemTime, String)> = None;
    for entry in std::fs::read_dir(SESSION_ROOT).ok()?.flatten() {
        let Ok(modified) = entry.metadata().and_then(|m| m.modified()) else {
            continue;
        };
        let name = entry.file_name().to_string_lossy().into_owned();
        if best.as_ref().is_none_or(|(t, _)| modified > *t) {
            best = Some((modified, name));
        }
    }
    best.map(|(_, name)| name)
}

/// One `--allow-X` category's state, Deno's own three-way split: never
/// mentioned (denied), mentioned bare (unrestricted), or mentioned with
/// values (scoped to exactly those, accumulating across repeats).
enum Allow {
    Unset,
    All,
    Only(Vec<String>),
}

impl Allow {
    /// `None` from `allow_values` means bare — jump straight to `All`,
    /// discarding any values already accumulated (a superset makes them
    /// moot). `Some` accumulates, so repeating the flag adds rather than
    /// replaces.
    fn apply(&mut self, values: Option<Vec<String>>) {
        match values {
            None => *self = Allow::All,
            Some(more) => match self {
                Allow::Only(existing) => existing.extend(more),
                _ => *self = Allow::Only(more),
            },
        }
    }

    /// The `GrantSpec` field this becomes: `None` inherits the console's
    /// ceiling (unrestricted, for every category console_authority grants
    /// `All`/everywhere), `Some` is an explicit list — empty for "never
    /// mentioned", which is how an omitted flag ends up denied rather than
    /// inherited.
    fn into_spec(self) -> Option<Vec<String>> {
        match self {
            Allow::Unset => Some(Vec::new()),
            Allow::All => None,
            Allow::Only(values) => Some(values),
        }
    }
}

/// One `--allow-X` flag's value, however it was written. `arg=v1,v2` is
/// always the value; otherwise the next token is taken as the value unless
/// it looks like another flag or there is none, in which case the whole
/// flag is bare — Deno's own reading of `--allow-net` alone as "any host".
fn allow_values(
    inline: Option<&str>,
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
) -> Option<Vec<String>> {
    if let Some(v) = inline {
        return Some(v.split(',').map(String::from).collect());
    }
    match args.peek() {
        Some(next) if !next.starts_with("--") => {
            Some(args.next().unwrap().split(',').map(String::from).collect())
        }
        _ => None,
    }
}

use std::time::Duration;
use system::{Mail, NodeSpec, System};
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1).peekable();
    let mut once = false;
    let mut role: Option<String> = None;
    // The filesystem is outside the harness, so nothing reaches it unless the
    // human grants a root here. Everything below the root attenuates from this.
    let mut allow_read = Allow::Unset;
    let mut allow_write = Allow::Unset;
    let mut allow_all = false;
    let mut allow_run = Allow::Unset;
    let mut allow_net = Allow::Unset;
    let mut allow_env = Allow::Unset;
    let mut allow_sys = Allow::Unset;
    let mut journal_dir: Option<String> = None;
    let mut env_file: Option<String> = None;
    let mut session: Option<String> = None;
    let mut resume = false;
    let mut rest: Vec<String> = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--once" => once = true,
            // Deno's spelling, and the same meaning: everything.
            "--allow-all" | "-A" => {
                allow_read = Allow::All;
                allow_write = Allow::All;
                allow_run = Allow::All;
                allow_net = Allow::All;
                allow_env = Allow::All;
                allow_sys = Allow::All;
                allow_all = true;
            }
            "--env-file" => match args.next() {
                Some(path) => env_file = Some(path),
                None => {
                    eprintln!("--env-file needs a path");
                    std::process::exit(2);
                }
            },
            "--journal" => match args.next() {
                Some(dir) => journal_dir = Some(dir),
                None => {
                    eprintln!("--journal needs a directory");
                    std::process::exit(2);
                }
            },
            // `--resume` alone picks up the most recent session. A name may
            // follow either as --resume=NAME or as the next word — the next
            // word is only taken when it actually names a session, so
            // `--resume "keep going"` still reads as a prompt.
            _ if arg == "--resume" || arg.starts_with("--resume=") => {
                resume = true;
                let named = arg.strip_prefix("--resume=").map(String::from).or_else(|| {
                    let candidate = args.peek()?;
                    std::path::Path::new(SESSION_ROOT)
                        .join(candidate)
                        .is_dir()
                        .then(|| args.next())
                        .flatten()
                });
                session = named.or_else(latest_session);
            }
            _ if arg == "--allow-run" || arg.starts_with("--allow-run=") => {
                let inline = arg.strip_prefix("--allow-run=");
                allow_run.apply(allow_values(inline, &mut args));
            }
            _ if arg == "--allow-net" || arg.starts_with("--allow-net=") => {
                let inline = arg.strip_prefix("--allow-net=");
                allow_net.apply(allow_values(inline, &mut args));
            }
            _ if arg == "--allow-env" || arg.starts_with("--allow-env=") => {
                let inline = arg.strip_prefix("--allow-env=");
                allow_env.apply(allow_values(inline, &mut args));
            }
            _ if arg == "--allow-sys" || arg.starts_with("--allow-sys=") => {
                let inline = arg.strip_prefix("--allow-sys=");
                allow_sys.apply(allow_values(inline, &mut args));
            }
            _ if arg == "--allow-read" || arg.starts_with("--allow-read=") => {
                let inline = arg.strip_prefix("--allow-read=");
                allow_read.apply(allow_values(inline, &mut args));
            }
            _ if arg == "--allow-write" || arg.starts_with("--allow-write=") => {
                let inline = arg.strip_prefix("--allow-write=");
                allow_write.apply(allow_values(inline, &mut args));
            }
            "--role" | "--system" => match args.next() {
                Some(text) => role = Some(text),
                None => {
                    eprintln!("--role needs a value");
                    std::process::exit(2);
                }
            },
            _ => rest.push(arg),
        }
    }
    let prompt = rest.join(" ");
    // With no prompt, come up idle and wait for the console. Only --once needs
    // something to do, since it exits as soon as everything settles.
    if prompt.trim().is_empty() && !resume && once {
        eprintln!(
            "usage: bitty [--once] [--role \"prompt\"] [-A | --allow-read[=DIR] \
             --allow-write[=DIR]] \"initial task\""
        );
        std::process::exit(2);
    }

    if allow_all {
        ui::system(
            "--allow-all: every capability is granted — read and write anywhere, any program, \
             any host, any environment variable. Nothing spawned below can exceed it because \
             nothing is bounded. Prefer naming what the task needs: --allow-read=DIR \
             --allow-write=DIR --allow-run=PROGRAM --allow-net=HOST --allow-env=NAME.",
        );
    }

    // Before the client is built, so a key kept in the file works too. A token
    // sitting in .env that was never exported is invisible to the harness, and
    // a process hunting for a variable that is not there is both a waste and an
    // alarming-looking thing for it to be doing.
    let named_file = env_file.is_some();
    match load_env_file(env_file.as_deref().unwrap_or(".env")) {
        Ok(names) if !names.is_empty() => {
            ui::system(&format!("loaded {} from .env — grant with --allow-env={}",
                names.join(", "), names.join(",")));
        }
        Ok(_) => {}
        // Only complain when a file was actually asked for; a missing .env is
        // the normal case, not a problem.
        Err(e) if named_file => {
            eprintln!("cannot read env file: {e}");
            std::process::exit(2);
        }
        Err(_) => {}
    }

    let api = api::Client::from_env()?;
    ui::system(&format!("bitty · model {} · /ps /graph /model /stop /quit · '@proc-N msg' targets a process, plain text goes to root", api.model));

    // Persistence is the default, not something to remember to switch on: an
    // interactive run that dies having built a world of scripts should be
    // resumable. `--once` is exempt — a one-shot batch run has nothing to come
    // back to, and would only litter the session directory.
    if journal_dir.is_none() && !once {
        if resume && session.is_none() {
            ui::system("no session to resume — starting a new one");
        }
        let name = session.clone().unwrap_or_else(new_session_name);
        journal_dir = Some(format!("{SESSION_ROOT}/{name}"));
        ui::system(&format!(
            "session {name} — resume it later with: bitty --resume {name}"
        ));
        session = Some(name);
    }
    // A resume with nothing to restore would otherwise sit silently on an empty
    // journal; say so rather than looking like a successful restore.
    if resume && session.is_none() && journal_dir.is_none() {
        ui::system("nothing to resume");
    }

    let journal: Arc<dyn durable::Journal> = match &journal_dir {
        Some(dir) => match durable::FileJournal::new(dir) {
            Ok(journal) => Arc::new(journal),
            Err(e) => {
                eprintln!("cannot open journal at {dir}: {e}");
                std::process::exit(2);
            }
        },
        None => Arc::new(durable::NoJournal),
    };
    let sys = Arc::new(System::new(api).with_journal(journal.clone()));

    if resume {
        let mut brought_back = 0;
        let mut highest = 0;
        for id in journal.processes() {
            let Some((record, history, stopped, pending)) = durable::restore(journal.replay(&id))
            else {
                continue;
            };
            highest = highest.max(record.ordinal);
            if stopped {
                continue; // a process that ended stays ended
            }
            sys.resume_ids_after(highest);
            ui::system(&format!(
                "↻ {} ({} turns, {} unread)",
                record.id,
                history.len(),
                pending.len()
            ));
            sys.restore(record, history, pending);
            brought_back += 1;
        }
        sys.resume_ids_after(highest);
        if brought_back == 0 {
            ui::system("nothing to resume in the journal");
        }
    }
    let root = if resume {
        // Everything that was running is already back; a prompt on a resume is
        // just a message to the existing root rather than a new tree.
        String::from("proc-1")
    } else {
        sys
        .spawn(
            "user",
            NodeSpec {
                instructions: prompt.clone(),
                name: Some("root".into()),
                persona: role,
                // Root sets the tree's default: anything it spawns inherits
                // this effort unless the spawn names its own.
                effort: Some(api::DEFAULT_EFFORT.to_string()),
                // A bare `--allow-X` (or `-A`) leaves the field `None`, which
                // inherits the console's ceiling — everything, for every one
                // of these — while a never-mentioned flag becomes `Some([])`,
                // denied. Never inherited by falling through unnoticed: an
                // omitted field would otherwise silently hand root everything.
                wants: system::GrantSpec {
                    run: allow_run.into_spec(),
                    net: allow_net.into_spec(),
                    env: allow_env.into_spec(),
                    sys: allow_sys.into_spec(),
                    read: allow_read.into_spec(),
                    write: allow_write.into_spec(),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .map_err(anyhow::Error::msg)?
    };
    if prompt.trim().is_empty() && !resume {
        ui::system("waiting for input — type a message, or /ps /graph /stop /quit");
    }
    if resume && !prompt.trim().is_empty() {
        let _ = sys.send(&root, user_mail(&prompt));
    }

    if once {
        run_until_idle(&sys).await;
        return Ok(());
    }

    // Forward stdin lines from a blocking thread into the async world.
    let (tx, mut lines) = mpsc::unbounded_channel::<String>();
    std::thread::spawn(move || {
        for line in std::io::stdin().lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    while let Some(line) = lines.recv().await {
        let line = line.trim().to_string();
        match line.as_str() {
            "" => continue,
            "/quit" | "/exit" => break,
            "/ps" => ui::system(&sys.list()),
            "/graph" | "/tree" => ui::system(&sys.graph()),
            _ if line.starts_with("/model") => {
                let args: Vec<&str> = line["/model".len()..].split_whitespace().collect();
                match args.as_slice() {
                    [id, model, rest @ ..] => {
                        match sys.set_model(id, model, rest.first().copied()) {
                            Ok(report) => ui::system(&report),
                            Err(why) => ui::system(&why),
                        }
                    }
                    _ => ui::system("usage: /model proc-1 claude-opus-5 [high]"),
                }
            }
            _ if line.starts_with("/stop") => {
                let rest = line["/stop".len()..].trim();
                let cascade = rest.split_whitespace().any(|t| t == "--cascade");
                let mut targets: Vec<String> = rest
                    .split_whitespace()
                    .filter(|t| !t.starts_with("--"))
                    .flat_map(|t| t.split(','))
                    .filter(|t| !t.is_empty())
                    .map(String::from)
                    .collect();
                if targets.iter().any(|t| t == "*") {
                    targets = sys.live_ids("");
                }
                if targets.is_empty() {
                    ui::system("usage: /stop proc-2 [proc-3 ...|*] [--cascade]");
                    continue;
                }
                // No initiator: a human-ordered stop is abnormal from the
                // perspective of anything linked to the target.
                if let Err(e) = sys.stop(&targets, cascade, None) {
                    ui::system(&e);
                }
            }
            _ if line.starts_with('@') => {
                let (spec, body) = line[1..]
                    .split_once(char::is_whitespace)
                    .unwrap_or((&line[1..], ""));
                if body.trim().is_empty() {
                    ui::system("usage: @proc-2[,proc-3,...|*] your message");
                    continue;
                }
                let targets: Vec<String> = if spec == "*" {
                    sys.live_ids("")
                } else {
                    spec.split(',').filter(|t| !t.is_empty()).map(String::from).collect()
                };
                for target in targets {
                    if let Err(e) = sys.send(&target, user_mail(body)) {
                        ui::system(&e);
                    }
                }
            }
            _ => {
                if let Err(e) = sys.send(&root, user_mail(&line)) {
                    ui::system(&e);
                }
            }
        }
    }
    Ok(())
}

fn user_mail(body: &str) -> Mail {
    // Console input is always urgent: the human is waiting.
    Mail::system("user", body.trim().to_string())
}

/// --once: poll until no process is running (idle or stopped) twice in a row.
async fn run_until_idle(sys: &Arc<System>) {
    let mut settled = 0;
    loop {
        tokio::time::sleep(Duration::from_millis(700)).await;
        if sys.all_settled() {
            settled += 1;
            if settled >= 2 {
                ui::system(&format!(
                    "all processes settled — exiting (--once) · peak context {}k tokens",
                    sys.peak_context() / 1_000
                ));
                return;
            }
        } else {
            settled = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Allow, allow_values};

    fn peekable(args: &[&str]) -> std::iter::Peekable<std::vec::IntoIter<String>> {
        args.iter().map(|s| s.to_string()).collect::<Vec<_>>().into_iter().peekable()
    }

    #[test]
    fn bare_flag_with_nothing_following_is_unrestricted() {
        let mut rest = peekable(&[]);
        assert_eq!(allow_values(None, &mut rest), None);
    }

    #[test]
    fn bare_flag_followed_by_another_flag_is_unrestricted() {
        let mut rest = peekable(&["--allow-write"]);
        assert_eq!(allow_values(None, &mut rest), None);
        // The next flag must still be there for the loop to see it next.
        assert_eq!(rest.next().as_deref(), Some("--allow-write"));
    }

    #[test]
    fn a_bare_value_token_is_taken_as_scoped_values() {
        let mut rest = peekable(&["host1,host2"]);
        assert_eq!(
            allow_values(None, &mut rest),
            Some(vec!["host1".to_string(), "host2".to_string()])
        );
        assert_eq!(rest.next(), None); // the value was consumed
    }

    #[test]
    fn an_inline_equals_value_never_touches_the_iterator() {
        let mut rest = peekable(&["--allow-write"]); // would misparse as bare if consulted
        assert_eq!(
            allow_values(Some("a,b"), &mut rest),
            Some(vec!["a".to_string(), "b".to_string()])
        );
        assert_eq!(rest.next().as_deref(), Some("--allow-write"));
    }

    #[test]
    fn repeated_scoped_values_accumulate() {
        let mut allow = Allow::Unset;
        allow.apply(Some(vec!["a".to_string()]));
        allow.apply(Some(vec!["b".to_string()]));
        assert_eq!(allow.into_spec(), Some(vec!["a".to_string(), "b".to_string()]));
    }

    #[test]
    fn a_later_bare_flag_upgrades_scoped_values_to_unrestricted() {
        let mut allow = Allow::Unset;
        allow.apply(Some(vec!["a".to_string()]));
        allow.apply(None);
        assert_eq!(allow.into_spec(), None);
    }

    #[test]
    fn never_mentioned_denies_rather_than_inheriting() {
        assert_eq!(Allow::Unset.into_spec(), Some(Vec::new()));
    }
}
