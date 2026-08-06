# Bitty

[![CI](https://github.com/by77er/Bitty/actions/workflows/ci.yml/badge.svg)](https://github.com/by77er/Bitty/actions/workflows/ci.yml)

An agent meta-harness built on the **actor model**. Every agent is an actor: a **process** with an address, private state (its own conversation with the model), and a **mailbox**. Processes share nothing — the only ways to interact are to **spawn** a new process or **send** a message to an address you know, and both are exposed to the model as tools. Incoming mail is injected mid-task, between tool calls — the same UX as interrupting a coding agent while it works.

![The actor model, applied to agents](actors.svg)

Processes come in two flavors: **agents** (a model conversation) and **scripts** (TypeScript actors on an embedded Deno runtime — same mailbox, same permissions, zero API tokens). Use agents for judgment and scripts for the mechanical parts: routing, aggregation, validation.

## Why

Single-agent harnesses hit a wall: one context window, one train of thought, one thing at a time. The actor model is the classic answer to exactly this shape of problem, and the properties that made it work for Erlang/OTP transfer directly to agents: **concurrency without shared state** (each process has its own context, so nothing steps on anything), **failure isolation** (a dead process signals its links with a message instead of taking anyone down with it), and **supervision** (spawners learn about their children's deaths and can re-plan or respawn). Bitty hands these primitives to the model itself — spawn, send, link, stop — and the agents decide how to organize.

That turns out to enable systems that run indefinitely, not just tasks that finish:

- **long-running services** — a script actor can serve HTTP from inside the system (`Deno.serve`), so a swarm can host something rather than just produce an artifact and exit;
- **self-maintaining projects** — give one agent ownership of a codebase or document and let others file requests through its mailbox;
- **pipelines and fan-out** — writer → editor chains, parallel research workers reporting back to a coordinator;
- **safe delegation** — each process holds only the files, peers, programs, hosts and variables it was granted, and can never grant a child more than it has, so untrusted subtasks stay contained.

## How it works

- Each process runs its own agentic loop as a tokio task, with an mpsc channel as its mailbox. Mail is injected between tool calls; an idle process blocks until woken.
- **Tools:** `spawn_process`, `spawn_topology`, `send_message`, `call_process` (send and block for the reply, in-turn), `mailbox` (page long mail), `stop_process`, `list_processes`, `run_script` (TypeScript in a persistent per-process session), and `patch_script` (replace a script process's code in place). A process is only shown the tools its capabilities allow.
- **Capabilities:** grants for `Send` / `Stop` / `Spawn` / `Run` / `Net` / `Env` / `Sys` plus read and write path roots, all clamped so a child can never hold authority its spawner lacks. Visibility follows authority — an isolated worker can't even enumerate its siblings.
- **Links:** a dying process signals its spawner (as `<exit_signal>` mail, never a kill), OTP-style, so coordinators can re-plan or respawn.
- **Topologies:** `spawn_topology` wires a whole group at once, with per-node roles, models, scripts, and `can_send_to` allowlists.
- **Tool aliases:** spawns can define typed tools that route to another actor; arguments are schema-validated before delivery, and a spawner may only point an alias at a process it could message itself (or a sibling in the same topology). The holder calls it as a plain async function inside `run_script` and is never told a graph exists.
- **Scripts:** an embedded `deno_core` runtime with `bitty.onMail`/`onStop`/`send`/`spawn`/`sleep`, `bitty.connect` (an awaitable WebSocket), `fetch`, `Deno.serve`, `Deno.Command` and the file APIs — every one checked against the process's grants. TypeScript is transpiled and syntax-checked before it runs.
- **Cost controls:** per-process model tier and reasoning effort — including mixing providers, so a Claude coordinator can run ChatGPT workers or vice versa — plus low-priority mail that never wakes anyone, artifact-backed long mail that is paged instead of injected wholesale, a shared prompt-cache prefix across the whole system, server-side compaction where the model supports it and harness-side summarization where it does not, and `--max-tokens` to make the system wind down on a budget.

See the source for the full details — `src/agent.rs` (the loop and tool surface), `src/system.rs` (process table and supervision), `src/script.rs` (the Deno runtime), `src/grants.rs` (capabilities), `src/actions.rs` (the shared policy layer), `src/durable.rs` (journaling).

## Install & use

Building needs a **recent nightly-ish rustup toolchain**: the crate is edition 2024, and `deno_core` 0.409 will not compile on an older nightly — a Rust 1.89 nightly fails outright. `rust-toolchain.toml` pins the version rustup should fetch, so a plain `cargo build` picks it up.

```bash
git clone https://github.com/by77er/Bitty && cd Bitty
cargo install --path .          # or: cargo build --release

export ANTHROPIC_API_KEY=sk-ant-...   # or put it in .env, which is loaded at startup

bitty "Research X with two parallel workers and summarize."
bitty --role "You coordinate a writing pipeline." "Draft a page on actor systems."
bitty --tui --allow-read . --allow-write . "Refactor the parser and keep the tests green."
bitty --once --gate "cargo test" "Fix the failing parser tests."
bitty --resume                        # pick up the most recent session
```

Options (`bitty --help` prints the same list):

| Flag | Effect |
| --- | --- |
| `--tui` | open the live alternate-screen dashboard (needs an interactive terminal) |
| `--once` | exit once every process has settled |
| `--role TEXT`, `--system TEXT` | add role text to the root process |
| `--max-tokens N` | ask the system to wind down after N billable tokens |
| `--gate COMMAND` | with `--once`, a command that must pass at quiescence; a failure goes back to the root as work, not an exit |
| `--gate-attempts N` | maximum gate failures before giving up (default 3) |
| `--resume[=NAME]` | resume NAME, or the most recent session; any remaining words are sent as a message |
| `--journal DIR` | store the process journal in DIR instead of a session directory |
| `--env-file FILE` | load `KEY=value` lines before the client is built (`.env` is read by default) |
| `-A`, `--allow-all` | grant every capability, unscoped |
| `--allow-read[=PATHS]`, `--allow-write[=PATHS]` | filesystem roots, comma-separated |
| `--allow-run[=PROGRAMS]` | programs processes may execute |
| `--allow-net[=HOSTS]` | hosts they may reach |
| `--allow-env[=NAMES]`, `--allow-sys[=KEYS]` | environment variables and system facts they may read |
| `-h`, `--help` | print help and exit |

Every permission is denied unless granted, a bare flag means unrestricted, and the root process can only narrow what it hands to a child. Whatever is left on the command line is the initial prompt.

The console is wired into the actor system while it runs:

| Input | Effect |
| --- | --- |
| plain text | mailed to the root process (interrupts it mid-task) |
| `@proc-3 message` | mail specific processes (comma-separated; `@*` to fan out) |
| `/ps`, `/graph` | process list / supervision and messaging graph |
| `/model proc-2 small [low]` | retune a process's model tier and effort for its next turn |
| `/stop proc-2 [--cascade]` | stop processes (`*` for all) |
| `/quit` | exit |

Pass `--tui` for a Codex-style interactive view: a chat-first transcript and
rounded composer, a slim selectable process tree listing each process's live
context and cost, and a status line carrying the selected process's model, its
own current context, its cost, the run's total cost, the session name and any
active filter. A cost that had to be estimated from the built-in price table is
marked `~$0.0412`, and a model with no known price shows `$?` rather than a
zero. Run totals — cache-hit share, billable tokens, peak context — sit in the
header beside the cat.
Up/Down selects a process and filters its activity, Esc clears the filter, the
mouse wheel (or Page Up/Down) scrolls the transcript, Home/End jump to the
oldest and latest lines, `Ctrl-T` toggles low-level trace lines, `Ctrl-O`
releases the mouse, and Ctrl-C exits. While the TUI is capturing the mouse the
terminal cannot do its own click-and-drag selection, so `Ctrl-O` hands the
mouse back for copying text: the status line then reads `mouse off · ctrl-o`,
Page Up/Down, Home and End still scroll, and `Ctrl-O` again takes the wheel
back. Most terminals will also bypass mouse reporting for a single drag if you
hold a modifier — Option in iTerm2, Fn in macOS Terminal.app, Shift in most
Linux terminals — which needs no toggling at all. `Ctrl-T` is the only trace toggle — every other printable key, `t`
included, types into the composer. With traces hidden the view is summarized
rather than truncated: each run of consecutive trace lines from one speaker
collapses into one line, and a long message body is clamped to its first eight
rendered rows plus a dim `… +N lines · ctrl-t` marker so one big report cannot
bury the transcript (never your own input, never a warning). The status line
reads `traces summarized`; `Ctrl-T` shows everything in full. A block of
collapsed traces is followed by a blank row so it reads as a unit. Every process
gets its own colour, derived from a stable hash of its id, and wears it in the
transcript, the process list and the status-line filter. Selecting a row bands
it and marks it with a cyan `›` rather than repainting it, so the process keeps
its colour while it is the selected one. Each process's model size rides in the
column directly under its status glyph — `S`, `M` or `L`, blank for a script —
coloured cool-to-warm with the tier, so it stays legible at the narrowest rail.
The rail's outer edge is a heavy rule, the tree's own pipes are thin and dim
beneath it, and the fields of a row are parted by a middle dot. A process that
is mid-turn spins a braille spinner beside its name in the rail and in the
status line, with the elapsed time for that turn. The cat in the header reflects
working, idle, and recent-warning state. Incoming worker and
user messages appear in the recipient's process view, and adjacent streamed
lines are grouped into one speaker turn. Plain mode remains the default (also
for automation that consumes stdout); in TUI mode, embedded `console.log` calls
and stray process stdout/stderr are routed into the transcript instead of
writing over the screen.

Agent-bound messages over roughly 8,000 characters are stored outside the
recipient's model context. The agent sees a short preview plus a private
`artifact_id` and can use its `mailbox` tool to list, page, or discard the full
body. Script processes continue to receive complete bodies directly, since
their mail does not consume model context.

## Persistence

Interactive runs are journaled by default under `.bitty/sessions/<name>/` — one
append-only `proc-N.jsonl` per process, plus `mail/<proc-id>/<artifact-id>.json`
for artifact-backed message bodies. The journal records spawns with their grants,
aliases and model, every model turn in both directions, mail at send time,
mailbox cursors, compactions, retunes, script patches and stops, with periodic
checkpoints that fold the history so far. `bitty --resume` rebuilds the process
table from it: personas, grants, conversations, undelivered mail, patched script
source and per-process model overrides all come back. Every model turn is
flushed to disk before its tool calls run, and a mailbox cursor is only advanced
once the message it covers is already durable — so a resume is consistent at
turn boundaries and mail is at-least-once. A turn interrupted mid-tools is
dropped rather than replayed, and the process is told which of its calls are in
doubt. That guarantee covers agent processes only: a script journals no turn and
its cursor advances before its handler runs, so mail delivered to a script
during the crashed turn is lost rather than redelivered. `--journal DIR` puts
the journal elsewhere; `--once` skips journaling entirely, since a one-shot
batch run has nothing to come back to.

## Configuration

Anthropic is the default backend and reads `ANTHROPIC_API_KEY` (or
`ANTHROPIC_AUTH_TOKEN` for an OAuth bearer), with `ANTHROPIC_BASE_URL` to point
elsewhere. `BITTY_PROVIDER=codex` switches to the ChatGPT/Codex Responses
endpoint, which reuses the Codex CLI's `~/.codex/auth.json` credentials and
refreshes them in place; `BITTY_CODEX_URL` overrides that endpoint.

Models are named as tiers — `small`, `medium`, `large` — which each provider maps
to its own model, so a topology or a journaled session survives a provider
switch; concrete ids are still accepted. `BITTY_MODEL` sets the root's tier
(default `large` at effort `high`; spawned processes inherit the model and drop to
effort `low` unless told otherwise). `BITTY_COMPACTION=off` disables server-side
compaction, and `BITTY_CONTEXT_WINDOW`, `BITTY_COMPACT_ABOVE` and
`BITTY_COMPACT_FLOOR` tune when the harness compacts a conversation itself.

To run without credentials, point `ANTHROPIC_BASE_URL` at any mock server
speaking the Messages API SSE format. `test/` holds exactly that: mocks that
script a multi-process scenario and assert server-side, covering stops and
links, topology ACLs, capability attenuation, compaction round-tripping, script
processes and mailbox paging — see `test/README.md`.
