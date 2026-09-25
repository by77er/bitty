# Offline test suite

The harness is exercised against mock Messages API servers rather than the live
API: each mock scripts a deterministic multi-process scenario and **asserts
server-side**, returning an `ASSERTION` 400 that surfaces in the harness output
whenever the harness sends something wrong. This is how behavior like
permission enforcement and compaction round-tripping is verified without
credentials.

Build first, then run the suite from anywhere; it needs `python3`:

```bash
cargo build
test/run_suite.sh
```

It runs every scenario below against `target/debug/bitty` and reports pass or
fail for each. A scenario fails on an assertion, on a run that never settles,
or on a mock that never came up.

| Mock | Covers |
|---|---|
| `mock_server.py` | self-stop, idempotent re-stop, mail to a stopped process, thinking-signature round-trip |
| `mock_topology.py` | topology wiring, ACL enforcement on send and stop, inherited vs empty context, role text in the system prompt |
| `mock_compaction.py` | compact beta header + `context_management` shape, compaction block round-trip through an *unknown* delta type, graceful degradation when the server rejects the beta |
| `mock_notices.py` | multicast partial success, `"*"` resolution, exit signals following links (and NOT reaching a merely-wired sibling), array stop targets |
| `mock_caps.py` | capability attenuation (a restricted process's child is not unrestricted), rejection of over-requests, `can_spawn: false` enforcement |
| `mock_script.py` | embedded-Deno script process: TS transpilation, mail delivery, computed reply, and that a script never calls the model API |
| `mock_call.py` | `call_process` against a script, and `patch_script` keeping the id and wiring |
| `mock_fs.py` | filesystem capabilities: a read inside the grant succeeds, a `..` escape is refused, and a grant beyond the spawner's own is rejected at spawn |
| `mock_alias.py` | tool aliases: a typed tool that routes to another process |
| `mock_exec.py` | `api.exec` returning text vs `Deno.Command` returning exact bytes |
| `mock_sh.py` | the `sh()` and `read()` helpers in the `run_script` session |
| `mock_flush.py` | a text-only turn is flushed to the journal before the process goes idle |
| `mock_check.py` | the syntax check at spawn (parse-only by design, so a syntax error is refused before a process id is claimed), and the Deno API shim |
| `mock_inline.py` | inline TypeScript in the `run_script` session: persistent state and results by reference |
| `mock_tools.py` | the tool list follows a process's capabilities, and always-tools lead it so the cached prefix stays shared |
| `mock_reply.py` | an agent answering another agent's `call_process` |
| `mock_myopic.py` | a worker holding tools but no view of the process graph |
| `mock_serve.py` | `Deno.serve` inside a script process, within its network grant |
| `mock_reactive.py` | reactive scripts: sleeping, and waiting on a socket instead of polling |
| `mock_summarize.py` | client-side compaction: the summary turn is asked for with no tools, replaces the conversation, keeps the opening briefing, and the process keeps working |
| `mock_overflow.py` | a context-overflow error compacts and retries instead of stalling |
| `mock_mailbox.py` | oversized agent mail preview/handle metadata, recipient paging, and discard |
| `mock_gate.py` | a `--gate` that fails, is fixed, and passes |
| `mock_codex_eof.py` | a Codex stream that dies mid-body is retried (runs only where `~/.codex/auth.json` exists) |
| `mock_restart.py` | a script process survives a harness restart through the journal and `--resume` |

Lessons worth keeping in mind when extending these:

- **Never match against `json.dumps(...)` output.** An early version of
  `mock_notices.py` searched for `from="system"` in a dumped JSON string, where
  the quotes are backslash-escaped. The assertion could never fire, so it looked
  like it passed. Match on extracted block text instead.
- **Assert where the thing lands.** An exit signal arriving while a process is
  mid-turn is appended to the *same* user turn as its tool results, not a
  separate one. An assertion in the next turn's branch never fires.
- **Avoid cross-process assertions that race.** Processes run concurrently, so
  "process A must have observed X by the time B does Y" is timing-dependent.
  Assert inside the branch that actually receives the thing.
