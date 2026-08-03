# Bitty

An actor-style agent harness. Every agent is a **process** with an id and a
private **mailbox**, and can **spawn** other processes and **send** free-form
messages by id — all exposed to the model as tools. Mailbox messages are
injected into a process's context as interrupts in the gaps between tool
calls, the same UX as messaging a coding agent mid-task.

## How it works

```
┌─ proc-1 root ────────────┐        ┌─ proc-2 worker ──────────┐
│ own conversation w/model │ spawn  │ own conversation w/model │
│ mailbox ◄────────────────┼───┐    │ mailbox ◄─┐              │
│ tools: spawn/send/list   │   └────┼───────────┼── send_message
└──────────────────────────┘        └───────────┘
```

- Each process runs its own agentic loop (`src/agent.rs`) as a tokio task,
  with a `tokio::sync::mpsc` channel as its mailbox.
- **Interrupts:** after each batch of tool calls, the mailbox is drained and
  any messages are appended to the same user turn (after the tool results)
  inside `<incoming_message from="proc-N">` tags — so mail lands mid-task,
  between tool calls, without breaking the tool-use protocol.
- **Idle:** when a process ends its turn without calling a tool, it blocks on
  its mailbox; the next message wakes it and becomes the next user turn.
- The Claude API is called over raw HTTP (`src/api.rs`, reqwest + SSE
  streaming — there's no official Rust SDK). Turns stream, so long agentic
  turns can't hit HTTP timeouts, and text prints live per-process.

### Process types

A process is either an **agent** (a model conversation) or a **script** (a
TypeScript actor on an embedded Deno runtime — `deno_core` + V8 in-process, not
a subprocess). Both are full actors: same mailbox, links, capabilities,
namespace, and lifecycle. They differ only in what decides their behavior, and
a script costs **no API tokens at all**.

Pass `script` to `spawn_process` or a topology node to get one:

```ts
bitty.onMail(async (mail, api) => {
  const n: number = mail.body.length;
  api.log(`counted ${n} from ${mail.from}`);
  await api.send(api.parent, `len=${n}`);
});
```

The `api` object mirrors an agent's tools one-for-one — `send(to, message,
priority?)`, `stop(targets, cascade?)`, `list()`, `log(text)`, plus `id`, `name`,
`parent`, `instructions`. Every call routes through `src/actions.rs`, the same
policy layer the agent's tools use, so grants and visibility are enforced
identically; a denied call throws in JS. TypeScript is stripped with `deno_ast`
before V8 sees it, so the syntax accepted matches `deno run`.

Use scripts for the mechanical nodes of a topology — routing, aggregation,
validation, counting, format conversion — and keep agents for judgment. It is
the cheapest available substitution: not a smaller model, but no model.

V8 isolates aren't `Send`, so each script owns a dedicated OS thread with its
own current-thread runtime. `System` therefore holds a `tokio::runtime::Handle`
rather than relying on an ambient one, so a script can spawn processes onto the
main runtime from its own thread.

### Tools available to every process

| Tool | Effect |
|---|---|
| `spawn_process` | Start one process: briefing, optional `role`, `context` mode, and `link`; returns its id |
| `spawn_topology` | Start a wired group at once: per-node role, context, link, and `can_send_to` allowlist |
| `send_message` | Deliver free-form text to one id, a list of ids, or `"*"` (everyone you may message) |
| `stop_process` | Permanently stop one id, a list, or `"*"` (every live process but you); optional `cascade` to descendants |
| `list_processes` | ids, names, status (running/idle/stopped), context size, parents |

Anywhere a tool names processes it takes one id, a list, or `"*"`, and reports
partial success rather than collapsing to one verdict — delivered to these, not
permitted for those, unknown for the rest.

Stopped processes stay in the registry as tombstones — that is what makes
"proc-7 has been stopped" possible instead of "no such process", keeps
re-stopping idempotent, and lets `/graph` show what happened. They are bounded:
the mailbox channel and task handle are released the moment a process stops, and
only the 64 most recent tombstones are retained, reaped at spawn time (the one
moment the registry grows, and safely after any exit signal has been delivered).
Reaping is safe because the id counter is monotonic, so a stale reference can
never alias a live process.

Stop semantics: the target's tokio task is aborted at its next await (mid-API-call
included), the entry stays listed as `stopped`, mail to it returns an error, and
re-stopping is a non-error no-op. Stopped ids can't be restarted — spawn a new
process instead. The system prompt tells workers to report results first, then
stop themselves when no follow-ups are expected, so the registry doesn't
accumulate idle actors.

## Fan-out

`send_message` takes one id, a list, or `"*"` — for a restricted process that
resolves to exactly its allowlist, for an unrestricted one to every live process,
and never to the human console. The body is written once regardless of recipient
count, and each recipient is permission-checked individually, so results report
partial success honestly: delivered to these, not permitted for those,
undeliverable for the rest. Fan-out is deliberately *not* free-feeling in the
prompt, because each delivery wakes an idle recipient and costs it a full turn.

## Links and exit signals

Links work like `spawn_link`: spawning a process links it to its spawner, in both
directions, and `link: false` opts out. When a linked process dies abnormally —
killed by someone else, cascade-stopped — or stalls after repeated API failures,
the harness delivers an `<exit_signal>` to the other end naming the process and
the reason. Stopping *yourself* is a clean exit and signals nobody.

Unlike OTP's default, an exit signal never kills the recipient: it arrives as
ordinary mail, so every process here effectively traps exits and decides for
itself whether to re-plan, take the work over, or spawn a replacement.

This closes a real deadlock. Process entries are never removed, so a mailbox
sender is never dropped, so a process waiting on one that died would wait
forever. The link is what tells it to stop.

**Links are not the wiring.** `can_send_to` says who may *talk* to whom; links
say who is *told* about a death. Siblings in a topology are wired to each other
but not linked, so when one dies only their spawner is signalled — and relaying
that to the others is the spawner's job. That's supervision-tree shaped
deliberately: notification follows the tree, not the message graph, so a death
wakes one process rather than every peer that happened to be adjacent.

The consequence to design around: a sibling depending on a sibling learns nothing
directly. Route the dependency through the coordinator, or don't make it. And
deadlock is rarer, not impossible — two processes each waiting for the other to
speak first still hang, which is what `/ps` and the `— system idle` line make
visible.

## Context

Each process has two independent pieces of context, both scoped to that
process alone:

**System prompt** — two content blocks ordered general → specific, which is a
caching decision, not a stylistic one. Block 0 is the harness scaffolding: how
mailboxes work, what the tools mean, cleanup discipline. It is **byte-identical
for every process in the run** and carries the cache breakpoint. Block 1 is
everything that varies — this process's id, its parent, its wiring, its `role`.

Because caching is a prefix match, this is what makes the ~1.8K-token prefix
(tool definitions plus scaffolding) shared across the whole system: the first
process writes it, every other process reads it at a tenth of the price. An
earlier version opened with "You are process proc-2…", which forked the prefix
at the first token and meant every process paid a full cache write for the same
scaffolding. Nothing process-specific may be added to block 0 — a single
interpolated id silently destroys the sharing for the entire topology.

The root process's role comes from `--role`.

**Conversation** — starts one of two ways, chosen per spawn:

- `context: "empty"` (default) — the process knows only its briefing and role.
  Cheap, focused, no coupling to the spawner.
- `context: "inherit"` — it additionally receives a rendered snapshot of the
  spawner's conversation at spawn time, inside `<inherited_context>` tags.

Inheritance is a *transcript*, not a fork of the message array: text and tool
traffic are flattened into readable lines, the spawner's thinking blocks are
dropped, and long blocks are truncated (4K per block, 60K total, tail-biased).
That's deliberate — replaying another process's `tool_use`/`tool_result` pairs
and thinking signatures as if they were the child's own would be both
semantically wrong and rejected by the API. It's a snapshot, so later parent
turns don't propagate; use `send_message` for ongoing updates.

**Filesystem access** is the one authority that starts empty and comes from the
console, since it is the only thing that reaches outside the harness:

```sh
bitty --allow-read ./repo --allow-write ./repo "fix the failing test"
bitty --allow-read ./repo --allow-read ./docs "summarize"   # repeatable
bitty -A "..."                                              # everything, with a warning
```

Root then narrows per child with `can_read` / `can_write` on spawn; omitting them
inherits the spawner's roots. `/graph` shows each process's `reads→` and
`writes→` so a grant can be audited at a glance. Network and subprocess
execution are **not** implemented — a script can compute and touch granted
files, nothing else.

**Compaction** — every request carries
`context_management: {"edits": [{"type": "compact_20260112"}]}`, so the server
watches each process's prompt size and, at a safe margin below the window,
summarizes the earlier conversation into a `compaction` block that replaces it
going forward. This is automatic and per-process; nothing in the harness has to
decide when to fire. The block is echoed back verbatim as part of the assistant
turn — that round-trip *is* the compaction state, so the harness preserves whole
content arrays rather than extracting text. A `⟳ server compacted earlier
context` trace line marks each occurrence.

If the server rejects compaction (model or account without the beta), the client
latches it off, warns once, and retries the same request without it — a missing
beta degrades rather than killing the run. Unrelated betas are unaffected. Set
`BITTY_COMPACTION=off` to disable it deliberately; `/ps` then flags that contexts
are unmanaged, and any process past 500K tokens warns once.

`/ps` shows each process's current context size (the prompt total: uncached +
cache-write + cache-read), which is the number compaction watches.

## Spend

Two knobs matter more than token trimming:

**Per-process model and effort.** `spawn_process` and each topology node take
`model` and `effort`, inherited from the spawner when omitted — so a cheap
worker's own helpers stay cheap. Running mechanical work on a smaller model at
lower effort is a multiplier on price-per-token, not a percentage off the count,
which makes it the largest lever available. `/graph` shows each process's model
and effort so you can see where the money is going.

**Message priority.** `send_message` takes `priority: "high" | "low"`. High
(default) wakes an idle recipient, which costs it a full turn — a whole prompt
re-read. Low never wakes anyone: the message is held and delivered the next time
that process runs for some other reason, so it costs only its own tokens. Status
updates and FYIs should be low; anything requiring a response or a change of
course should be high. Held mail is flushed chronologically ahead of whatever
eventually wakes the process.

The tradeoff to know: low-priority mail sent to a process that never wakes again
is never read. That is the intended behavior — it is what makes it free — but it
means low is wrong for anything load-bearing.

**Tools are deliberately identical for every process**, even ones that cannot use
half of them. Trimming per-capability would save ~700 tokens of schema but fork
the cache prefix at position zero, since tools render before `system` — turning a
0.1× cached read of the whole 1.8K prefix into a full write. The tokens saved
cost more than they save.

## Capabilities

Permissions are one model covering every verb, in `src/grants.rs`. A process
holds a `Grant` per `Capability` — `Send`, `Stop`, `Spawn` — and each grant is
`All`, an explicit set of ids, or `Nobody`. There is one check
(`Meta::may(cap, target)`) and one renderer, so adding a verb is a new enum
variant rather than another field with its own idea of "unrestricted".

**Attenuation is the load-bearing rule: a process can never hold a grant its
spawner lacks.** Without it, isolation was one spawn deep — a restricted node
could call `spawn_process`, get an unrestricted child, and use it as a proxy to
reach anything. Now the child's grants are clamped to the parent's, transitively.

An **explicit** request for more is rejected outright, naming what was refused,
rather than silently trimmed — a coordinator wiring a worker to a peer it can't
reach has a broken plan and should learn immediately. An **omitted** field isn't
a request; it inherits, and clamping there is just the default working.

Two rules survive attenuation because they're structural, not privileges: a
process may always stop *itself* (otherwise it can never exit cleanly), and stop
authority is never inherited from a restricted spawner — a grandchild defaults to
stopping only itself rather than quietly acquiring the power to kill its parent.
An explicit `can_send_to` means exactly that list; messaging the spawner is a
policy choice, which is why topology nodes default to `["parent"]` when the field
is omitted.

Targets resolve to sibling names in the same spawn call, `parent`, `self`,
`user`, or **the id of any already-running process** — so a later group can be
wired to an earlier one.

## Tool aliases

A process can give a process it spawns **named, schema-typed tools that are
really calls to another process**. The holder sees an ordinary tool with a
description and an argument schema; invoking it validates the arguments, sends
them to the target, and returns the reply inside the same turn.

```json
{"name": "worker", "instructions": "Use your add tool on 2 and 40.",
 "can_send_to": ["parent", "calc"],
 "tools": [{"name": "add", "description": "Add two numbers.", "target": "calc",
            "input_schema": {"type": "object",
                             "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                             "required": ["a", "b"]}}]}
```

This turns an informal convention ("message the calc process with a sum") into a
checked contract. Bad arguments come back as an error result the caller can
correct against, and never reach the target. Validation is a deliberate subset
of JSON Schema — required fields, property types, enums — which is what models
actually get wrong.

**An alias cannot launder authority.** It may only target a process the holder
is already permitted to message; otherwise the spawn is rejected. So a tool is
never a way around the capability model.

**Caching.** Aliases are appended after the built-in tools, and the last
built-in carries a cache breakpoint. So the base tool list stays a shared cached
prefix for every process in the system regardless of what aliases follow, and a
process with its own tools pays only for its own tail. That is why per-process
tools are acceptable here when trimming tools per capability was not: the cost
is bounded and buys a typed interface, rather than being pure loss.

## Visibility

Capabilities control what a process may *do*; visibility controls what it may
*see*. Without the second, an isolated worker could still call `list_processes`
and enumerate the whole system — names, statuses, parents, context sizes — which
undercuts the isolation.

**Visibility tracks authority.** A process holding `All` over messaging or
stopping may act on anything, so it sees everything; hiding the system from it
would be incoherent. A process confined to an allowlist is in a namespace: it
sees itself, everything it spawned (transitively), its spawner, and any process
named in its grants. Nothing else exists as far as it is concerned.

Descendants expand from *self* only. Seeding the walk with the spawner would
sweep in every sibling through the shared parent edge, which is the bug this
design exists to prevent — a topology node cannot see its peers unless it was
wired to them.

An id outside the view reports as **unknown**, not forbidden. "Not permitted"
would confirm the existence of something the process shouldn't know about, so
absence and denial are deliberately different answers. `list_processes`, `"*"`
expansion, `send_message`, and `stop_process` all respect this. The console does
not — `/ps` and `/graph` are global, because the human is outside the namespace.

## Topologies

`spawn_topology` creates several processes at once and wires them together.
Because ids are allocated for the whole group before any of it starts, nodes can
reference each other by symbolic name:

```json
{"processes": [
  {"name": "writer", "role": "You are a terse technical writer.",
   "instructions": "Draft the section, send it to the editor.",
   "context": "inherit", "can_send_to": ["editor"]},
  {"name": "editor", "role": "You are a copy editor.",
   "instructions": "Polish the draft, deliver it, then report back.",
   "can_send_to": ["parent", "user"]}
]}
```

`can_send_to` accepts sibling names plus `parent` and `user`. It is **enforced**:
a send to anyone else returns an error tool result naming the permitted targets.
Omitting it defaults to `["parent"]` so results always have somewhere to go; `[]`
means the process reports to no one. A process with an allowlist may stop only
itself, so a worker can't kill its coordinator. Unrestricted processes
(`spawn_process`, or the root) can message and stop anyone. Max 16 nodes per
topology; unknown target names are rejected before anything starts running.

## Run

```sh
export ANTHROPIC_API_KEY=sk-ant-...     # or: ant auth login && eval "$(ant auth print-credentials --env)"
cargo run -- "Research X with two parallel workers and summarize."
cargo run -- --role "You coordinate a writing pipeline." "Draft a page on actor systems."
```

While it runs, the console is wired to the actor system:

- plain text → mailed to the root process (interrupts it mid-task)
- `@proc-3 message` → mail a process; `@proc-2,proc-3 msg` or `@* msg` to fan out
- `/ps` → flat process list; `/graph` → supervision tree plus who may message whom
- `/stop proc-2 proc-3 [--cascade]` or `/stop *` → stop; `/quit` → exit
- `--once` flag: exit automatically when every process is idle or stopped (for scripts)

Config: `BITTY_MODEL` overrides the model (default `claude-opus-5`);
`ANTHROPIC_BASE_URL` overrides the endpoint.

Requests enable server-side **refusal fallbacks** (`fallbacks: "default"`) by
default — if safety classifiers decline a request, it's re-served by the
recommended fallback model in the same call. Remove the `fallbacks` field and
beta header in `src/api.rs` if you don't want that.

## Extending

Processes currently have only the actor tools — they're pure reasoners that
coordinate. To give them real capabilities, add a tool definition in
`tool_definitions()` and a match arm in `execute_tool()` (`src/agent.rs`).
Content blocks are kept as raw `serde_json::Value` throughout so new API block
types (and thinking-block signatures) round-trip without code changes.

## Testing without an API key

`scratchpad/mock_server.py` (from the build session) shows the pattern: point
`ANTHROPIC_BASE_URL` at any server that speaks the Messages API SSE format and
script deterministic multi-actor scenarios.
