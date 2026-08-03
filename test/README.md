# Offline test suite

The harness is exercised against mock Messages API servers rather than the live
API: each mock scripts a deterministic multi-process scenario and **asserts
server-side**, returning an `ASSERTION` 400 that surfaces in the harness output
whenever the harness sends something wrong. This is how behavior like
permission enforcement and compaction round-tripping is verified without
credentials.

`run_suite.sh` runs all four and reports pass/fail:

| Mock | Covers |
|---|---|
| `mock_server.py` | self-stop, idempotent re-stop, mail to a stopped process, thinking-signature round-trip |
| `mock_topology.py` | topology wiring, ACL enforcement on send and stop, inherited vs empty context, role text in the system prompt |
| `mock_compaction.py` | compact beta header + `context_management` shape, compaction block round-trip through an *unknown* delta type, graceful degradation when the server rejects the beta |
| `mock_script.py` | embedded-Deno script process: TS transpilation, mail delivery, computed reply, and that a script never calls the model API |
| `mock_caps.py` | capability attenuation (a restricted process's child is not unrestricted), rejection of over-requests, `can_spawn: false` enforcement |
| `mock_notices.py` | multicast partial success, `"*"` resolution, exit signals following links (and NOT reaching a merely-wired sibling), array stop targets |

Two lessons worth keeping in mind when extending these:

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
