# Bitty

[![CI](https://github.com/by77er/Bitty/actions/workflows/ci.yml/badge.svg)](https://github.com/by77er/Bitty/actions/workflows/ci.yml)

An agent meta-harness built on the **actor model**. Every agent is a **process** with an address, private state (its own conversation with the model), and a **mailbox**. Processes share nothing: the only ways to interact are to **spawn** a process or **send** mail to an address you know, and both are tools the model calls. Mail lands mid-task, between tool calls, the way you'd interrupt a coding agent while it works.

![The actor model, applied to agents](actors.svg)

Processes come in two flavors: **agents** (a model conversation) and **scripts** (TypeScript actors on an embedded Deno runtime: same mailbox, same permissions, zero tokens). Agents for judgment, scripts for the mechanical parts.

## Why

Single-agent harnesses hit a wall: one context window, one train of thought, one thing at a time. The actor model is the classic answer to that shape of problem, and the properties that made it work for Erlang/OTP transfer directly: **concurrency without shared state**, **failure isolation** (a dead process signals its links with mail instead of taking anyone down), and **supervision** (spawners learn about their children's deaths and can re-plan or respawn). Bitty hands those primitives to the model and lets the agents decide how to organize.

That enables systems that run indefinitely, not just tasks that finish:

- **Long-running services.** A script actor can `Deno.serve` from inside the system, so a swarm can host something rather than emit an artifact and exit.
- **Self-maintaining projects.** Give one agent ownership of a codebase and let others file requests through its mailbox.
- **Pipelines and fan-out.** Writer → editor chains; parallel researchers reporting to a coordinator.
- **Safe delegation.** A process holds only the files, peers, programs, hosts and variables it was granted, and can never grant a child more than it has.

## How it works

- Each process is a tokio task running its own agentic loop, with an mpsc channel as its mailbox. Mail is injected between tool calls; an idle process blocks until woken.
- **Tools:** `spawn_process`, `spawn_topology`, `send_message`, `call_process` (send and block for the reply), `mailbox` (page long mail), `stop_process`, `list_processes`, `run_script`, `patch_script`. A process only sees the tools its capabilities allow.
- **Capabilities:** `Send` / `Stop` / `Spawn` / `Run` / `Net` / `Env` / `Sys` plus read and write roots, clamped so a child never holds authority its spawner lacks. Visibility follows authority: an isolated worker can't even list its siblings.
- **Links:** a dying process signals its spawner as `<exit_signal>` mail, never a kill, OTP-style.
- **Topologies:** `spawn_topology` wires a whole group at once, with per-node roles, models, scripts and `can_send_to` allowlists.
- **Tool aliases:** a spawner can define typed tools that route to another actor. Arguments are schema-validated before delivery, and an alias may only target a process the spawner could message itself. The holder calls it as a plain async function and is never told a graph exists.
- **Scripts:** an embedded `deno_core` runtime with `bitty.onMail` / `send` / `spawn`, `fetch`, `Deno.serve`, `Deno.Command` and the file APIs, every call checked against the process's grants. TypeScript is transpiled and syntax-checked before it runs.
- **Cost controls:** per-process model tier and effort, with providers mixed freely (a Claude coordinator can run ChatGPT workers). Low-priority mail never wakes anyone. Long mail is stored as an artifact and paged, not injected wholesale. One prompt-cache prefix is shared across the system, and `--max-tokens` winds it down on a budget.

Source map: `src/agent.rs` (the loop and tool surface), `src/system.rs` (process table and supervision), `src/script.rs` (Deno runtime), `src/grants.rs` (capabilities), `src/actions.rs` (policy layer), `src/durable.rs` (journaling).

## Install & use

Needs a recent nightly toolchain (`deno_core` 0.409 wants const `TypeId`). `rust-toolchain.toml` pins one, so rustup fetches it on the first `cargo build`.

```bash
git clone https://github.com/by77er/Bitty && cd Bitty
cargo install --path .
export ANTHROPIC_API_KEY=sk-ant-...   # or put it in .env

bitty "Research X with two parallel workers and summarize."
bitty --role "You coordinate a writing pipeline." "Draft a page on actor systems."
bitty --tui --allow-read . --allow-write . "Refactor the parser and keep the tests green."
bitty --once --gate "cargo test" "Fix the failing parser tests."
bitty --resume                        # pick up the most recent session
```

**Permissions** follow Deno's convention: omitted means denied, a bare flag means unrestricted, a value scopes it. `--allow-read[=PATHS]`, `--allow-write[=PATHS]`, `--allow-run[=PROGRAMS]`, `--allow-net[=HOSTS]`, `--allow-env[=NAMES]`, `--allow-sys[=KEYS]`, or `-A` for everything. The root process can only narrow what it hands to a child.

**Run modes:** `--tui` for the live dashboard, `--once` to exit when everything settles, `--gate CMD` (with `--once`) to require a command to pass first (a failure goes back to the root as work, up to `--gate-attempts`), `--max-tokens N` to wind down on a budget, `--role TEXT` for the root's system prompt. `bitty --help` lists the rest.

The console is wired into the actor system while it runs:

| Input | Effect |
| --- | --- |
| plain text | mail the root process (interrupts it mid-task) |
| `@proc-3 message` | mail specific processes (`@*` to fan out) |
| `/ps`, `/graph` | process list / supervision and messaging graph |
| `/model proc-2 small [low]` | retune a process's model tier and effort |
| `/stop proc-2 [--cascade]` | stop processes (`*` for all) |
| `/quit` | exit |

`--tui` opens an alternate-screen dashboard: a chat transcript, a selectable process tree with live context and cost per process, and a status line with the run's totals. Up/Down filters by process, `Ctrl-T` toggles trace lines, `Ctrl-O` releases the mouse for copying text.

## Persistence

Interactive runs are journaled under `.bitty/sessions/<name>/`: one append-only `proc-N.jsonl` per process plus artifact-backed mail bodies. `bitty --resume` rebuilds the whole process table from it: personas, grants, conversations, undelivered mail, patched scripts and model overrides. Every model turn is flushed before its tool calls run and a mailbox cursor only advances once the mail is durable, so a resume is consistent at turn boundaries and mail is at-least-once. `--once` skips journaling.

## Configuration

Anthropic is the default backend (`ANTHROPIC_API_KEY`, or `ANTHROPIC_AUTH_TOKEN` for OAuth; `ANTHROPIC_BASE_URL` to point elsewhere). `BITTY_PROVIDER=codex` switches to the ChatGPT/Codex Responses endpoint, reusing the Codex CLI's `~/.codex/auth.json`.

Models are named as tiers, `small` / `medium` / `large`, that each provider maps to its own model, so a topology or journaled session survives a provider switch. `BITTY_MODEL` sets the root's tier (default `large` at effort `high`; children inherit the model at effort `low`). `BITTY_COMPACTION=off` disables server-side compaction; `BITTY_CONTEXT_WINDOW`, `BITTY_COMPACT_ABOVE` and `BITTY_COMPACT_FLOOR` tune the harness's own.

To run without credentials, point `ANTHROPIC_BASE_URL` at any mock server speaking the Messages API SSE format. `test/` is exactly that: mocks that script multi-process scenarios and assert server-side. See `test/README.md`.
