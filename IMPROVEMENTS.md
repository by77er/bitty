# Improvement plan: REPL-first Bitty + robustness

Status: A1–A5 and A7 implemented (A3 was already true — inline scripts carry
the full `bitty` api); A6 deferred as noted. B8–B13 implemented. C14–C16 not
started. Verified by unit tests plus mock scenarios in `test/` (notably
`mock_inline.py` for session persistence and results-by-reference,
`mock_alias.py`/`mock_myopic.py` for session toolsets, `mock_summarize.py`
for merge compaction + the session notice, `mock_overflow.py` for overflow
recovery, `mock_gate.py` for gates) and a live Codex end-to-end run.

Derived from an architectural comparison against Prime Intellect's prime-agent
(https://www.primeintellect.ai/blog/prime-agent, source at
github.com/PrimeIntellect-ai/prime-agent). The load-bearing observation: prime-agent's
single-REPL design cost them their permission model entirely — they have none.
It will not cost Bitty, because every environment op already routes through
`actions.rs` and the grants layer regardless of what JS calls it. The op
boundary *is* the capability boundary, so REPL-ification changes the model's
interface, not the enforcement.

## A. REPL-first unification

1. **Persistent per-agent session.** Each agent process lazily owns a long-lived
   V8 isolate (same machinery as script processes). `run_script` evaluates in it;
   state persists across calls and turns via `g.*` (`globalThis`). Prime-agent's
   kernel-per-session, minus the Jupyter apparatus.
2. **Results by reference.** Past a size threshold, a `run_script` value stays in
   the session as `g.results.rN` and the tool result is a preview + handle, not
   the payload. This is the mechanism behind prime-agent's long-context and
   token-efficiency wins ("programmatic function calls on data versus spending
   tokens reading data through tools"), and the enforceable version of the
   "redirect output through a script" prompt guidance.
3. **Actor verbs in the session.** The session exposes the same `bitty.*` api
   script processes get (send/stop/list/spawn/fs/exec/fetch/...), so agents and
   scripts see one environment. Structural verbs stay as schema tools too —
   they are cheap, cached, and load-bearing for the interrupt UX.
4. **Toolsets through the REPL.** Parent-created tools (aliases) become typed
   async functions in the child's session namespace instead of schema tools.
   Validation stays host-side at the op boundary; the spawn-time
   target-reachability check and "an alias cannot launder authority" survive
   untouched. Documented to the child as TS signatures in its identity block.
5. **Unify agent and script actors.** After 1–4 a process = mailbox + grants +
   session; a script's behavior is fixed code, an agent's is a model deciding
   what to run next. Shared onMail/onStop/patch machinery.
6. **Session restart doctrine.** No heap snapshots (prime-agent could pickle a
   Python namespace; V8 heap does not serialize, and their "never replay
   uncertain work" doctrine is right anyway). Session state is ephemeral across
   restarts, like script heap; durable state goes in files. *Deferred: a
   journaled per-process init source re-run on restore (`Event::Patched` is
   already the right primitive).*
7. **Post-compaction session notice.** After client-side compaction, list the
   surviving session globals in the compacted block so the model reuses instead
   of redefining (prime-agent's `<ipython_state>` probe).

## B. Robustness

8. **Typed error taxonomy + overflow recovery.** Classify API failures
   (rate-limit / overloaded / server / auth / invalid / context-overflow /
   refusal) instead of string matching; route context-overflow to
   compact-and-retry with a one-shot guard instead of the 3-strikes stall.
9. **Per-model compaction gating.** `compacts_for_us` consults the resolved
   model's caps, not just provider + global flag — otherwise small-tier
   processes get neither server-side nor client-side compaction and grow
   silently until the API refuses them.
10. **Token-based compaction trigger + merge-style recompaction.** Trigger on
    real prompt tokens vs per-model window minus reserve (chars/4 only as
    estimate for growth since the last real count). Re-compaction merges into
    the prior summary ("preserve everything; move In Progress to Done") rather
    than re-summarizing the summary as ordinary history.
11. **Crash-uncertainty notice.** When restore drops a trailing unanswered
    assistant turn, tell the resumed process which tool calls may already have
    fired ("verify before repeating them"). Sends are journaled at the
    recipient at send time, so silent re-execution means duplicate messages and
    workers. Prime-agent's `<worker_interrupted>` doctrine.
12. **Mail hardening.** Body cap with truncation notice; per-(sender, recipient)
    flood limit (prime-agent: token bucket + queue caps). Agent-bound bodies
    above the inline threshold are durable recipient-scoped artifacts: the
    conversation gets a preview and handle, and the always-available `mailbox`
    tool lists, pages, or discards the original. Call replies use the same
    mechanism, and the prompt explicitly warns that `in_reply_to` and ordinary
    mail are independent channels so an answer should only use one.
13. **Gates and budgets for `--once`.** `--gate CMD` runs at quiesce; failure
    mails root bounded output and the run continues, up to a max-attempt cap,
    with skip-if-unchanged (write-root snapshot taken after the failing run).
    `--max-tokens` budget counted as input + output + cache-write, excluding
    cache reads (or long verifier loops exhaust budgets re-reading their own
    context).

## TUI (implemented)

`bitty --tui` takes over the terminal with a live view of the system — a
cat mascot for state at a glance, curated panels so you see what's going on
without being overwhelmed. Built on **ratatui + crossterm** (the standard
stack — the hand-rolling in this repo is reserved for places where an
abstraction would sit on a load-bearing invariant, like the provider
layer's raw-Value message shape; terminal rendering is commodity and gets a
commodity crate).

- **Opt-in flag, plain console stays the default.** The mock suite and
  `--once` scripts grep stdout lines; the TUI must not replace that. `--tui`
  enters the alternate screen; everything else is unchanged.
- **The structured event stream is the source of truth.** `ui.rs`'s broadcast
  tap (`ui::Event { kind, who, process, text }` + `ui::tap()`) is wired into
  `say/trace/mail_to_user/warn/system` and feeds the TUI the same lines
  the plain console prints, structured and un-ANSI'd. Successful deliveries
  additionally emit a feed-only `incoming` event owned by the recipient, so
  user and worker mail appears in that process's view without duplicating
  plain-mode output. Adjacent streamed `say` lines from one process coalesce
  into one visual turn instead of repeating its label. A lagging subscriber
  skips rather than blocking printing. In TUI mode, `emit` to stdout is
  suppressed (the tap is the display); direct stdout/stderr is quarantined
  into the transcript, and terminal control bytes are rendered inert.
- **`System::snapshot()`** provides per-process
  `{id, name, parent, status, runs, tokens}` plus `billable` and a settled
  flag, computed under one procs lock (don't call `all_settled()` inside —
  double lock).
- **Layout.**
  - Compact header: animated ASCII cat + `bitty` + session name + one-word
    system state.
  - Slim left rail: process tree from the snapshot (status glyph ●/○/■, id,
    name, model/effort, ctx tokens), arrow-key selectable.
  - Dominant main transcript: activity from the tap, colored by kind and
    presented as chat rather than a monitoring panel. `trace` lines are hidden
    by default behind the `Ctrl-T` toggle — that's the not-overwhelming part.
    Selecting a process filters the feed to it; Esc clears. Mouse-wheel and
    Page Up/Down input scroll the transcript buffer rather than moving the
    process selection; End returns to the latest activity.
  - Bottom: a rounded composer and compact model/context/billable status line,
    wired to the same console dispatch as plain mode (plain text → root,
    `@proc-N msg`, `/ps` `/graph` `/model` `/stop` `/quit`). The shared
    dispatcher keeps the two consoles from drifting.
- **The cat** (2–3 ASCII frames, mood from snapshot + recent events):
  working → `(^･ω･^)` with a tail-wag frame alternation; all idle →
  `(-ω-)ᶻᶻ`; a warn in the last 5s → `(⊙ω⊙)!`. A status indicator that
  happens to be a cat, not a toy.
- **Redraw** on a 250ms tick + on every tap event (debounced); snapshot
  refetched on the tick.
- **Testing:** pure layout/filter/wrapping behavior uses a ratatui test backend,
  alternate-screen entry/restore has a pseudo-terminal smoke test, and the
  plain-mode mock suite runs unchanged. `ui::tap` stays inert unless subscribed.

## C. Later (not in this pass)

14. **Skills store** — mechanism behind "promote a recurring script": a
    project-scoped `.bitty/harness/` of named script sources and
    role/topology templates, indexed one line each in the identity block.
15. **Subtree spend attribution** in `/graph`.
16. **Detach** (`bitty serve` + console-as-client), rising in priority as gates
    make long unattended runs common.

## Explicitly not adopted from prime-agent

- The single-tool paradigm as such (dissolves the capability model; their
  Factorio reward-hacking case study is the cautionary tale).
- Their steering model (finish turn, fresh run) — Bitty's mid-turn mail
  injection is the defining feature and strictly lower-latency.
- Session branching/trees — branch semantics across concurrent actor journals
  are incoherent; per-process linear journals + named sessions stay.
- Kernel-style heap snapshots (see A6).
