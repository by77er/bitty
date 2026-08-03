//! Bitty — an actor-style agent harness.
//!
//! Usage:
//!   bitty "initial task for the root process"
//!   bitty --once "task"          # exit when every process is idle or stopped
//!   bitty --role "you are ..."   # system prompt for the root process
//!   bitty --allow-read DIR --allow-write DIR   # filesystem roots (repeatable)
//!   bitty -A                                   # shorthand for read+write everywhere
//!   bitty --allow-run cargo,python3           # programs scripts may execute
//!   bitty --allow-net api.example.com         # hosts they may reach
//!   bitty --allow-env HOME,PATH --allow-sys   # environment and system facts
//!   bitty --journal DIR                        # record processes so they survive a restart
//!   bitty --resume [\"message\"]                 # bring them back, optionally with a nudge
//!
//! Console, while running:
//!   plain text        → mail to the root process (interrupts it mid-task)
//!   @proc-3 message   → mail to a specific process
//!   /ps               → list processes
//!   /graph            → supervision tree + who may message whom
//!   /stop proc-3      → stop a process (add --cascade for its descendants)
//!   /quit             → exit

mod agent;
mod durable;
mod grants;
mod actions;
mod api;
mod system;
mod script;
mod ui;

use std::sync::Arc;

/// Where processes are recorded unless told otherwise.
const DEFAULT_JOURNAL: &str = ".bitty/journal";
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
    let mut allow_read: Vec<String> = Vec::new();
    let mut allow_write: Vec<String> = Vec::new();
    let mut allow_all = false;
    let mut allow_run: Vec<String> = Vec::new();
    let mut allow_net: Vec<String> = Vec::new();
    let mut allow_env: Vec<String> = Vec::new();
    let mut allow_sys: Vec<String> = Vec::new();
    let mut journal_dir: Option<String> = None;
    let mut resume = false;
    let mut rest: Vec<String> = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--once" => once = true,
            // Deno's spelling, and the same meaning: everything.
            "--allow-all" | "-A" => {
                allow_read.push("/".into());
                allow_write.push("/".into());
                allow_all = true;
            }
            "--journal" => match args.next() {
                Some(dir) => journal_dir = Some(dir),
                None => {
                    eprintln!("--journal needs a directory");
                    std::process::exit(2);
                }
            },
            "--resume" => {
                resume = true;
                journal_dir.get_or_insert_with(|| DEFAULT_JOURNAL.to_string());
            }
            "--allow-run" => match args.next() {
                Some(v) => allow_run.extend(v.split(',').map(String::from)),
                None => { eprintln!("--allow-run needs a program name"); std::process::exit(2); }
            },
            "--allow-net" => match args.next() {
                Some(v) => allow_net.extend(v.split(',').map(String::from)),
                None => { eprintln!("--allow-net needs a host"); std::process::exit(2); }
            },
            "--allow-env" => match args.next() {
                Some(v) => allow_env.extend(v.split(',').map(String::from)),
                None => { eprintln!("--allow-env needs a variable name"); std::process::exit(2); }
            },
            "--allow-sys" => {
                allow_sys.extend(["hostname", "osRelease", "arch", "cwd"].map(String::from));
            }
            "--allow-read" => match args.next() {
                Some(path) => allow_read.push(path),
                None => {
                    eprintln!("--allow-read needs a directory");
                    std::process::exit(2);
                }
            },
            "--allow-write" => match args.next() {
                Some(path) => allow_write.push(path),
                None => {
                    eprintln!("--allow-write needs a directory");
                    std::process::exit(2);
                }
            },
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
            "usage: bitty [--once] [--role \"prompt\"] [-A | --allow-read DIR --allow-write DIR] \
             \"initial task\""
        );
        std::process::exit(2);
    }

    if allow_all {
        ui::system(
            "--allow-all: every capability is granted — read and write anywhere, any program, \
             any host, any environment variable. Nothing spawned below can exceed it because \
             nothing is bounded. Prefer naming what the task needs: --allow-read DIR \
             --allow-write DIR --allow-run PROGRAM --allow-net HOST --allow-env NAME.",
        );
    }

    let api = api::Client::from_env()?;
    ui::system(&format!("bitty · model {} · /ps /graph /stop /quit · '@proc-N msg' targets a process, plain text goes to root", api.model));

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
                // Always explicit, never inherited: the console's ceiling is
                // the whole filesystem, so falling through to "inherit" would
                // silently hand root everything. Empty means no access.
                // Omitting a field inherits the console's authority, which is
                // everything — so --allow-all simply leaves them unset, and
                // otherwise each is pinned to exactly what was granted.
                wants: system::GrantSpec {
                    run: (!allow_all).then_some(allow_run),
                    net: (!allow_all).then_some(allow_net),
                    env: (!allow_all).then_some(allow_env),
                    sys: (!allow_all).then_some(allow_sys),
                    read: (!allow_all).then_some(allow_read),
                    write: (!allow_all).then_some(allow_write),
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
