"""A model turn that ends without tool calls must be on disk before the
harness does anything else.

The bug this pins: FileJournal batches writes and auto-flushes only Spawned and
Stopped, so an Output record for a text-only turn sat in the BufWriter. On
--resume the replayed history ended on a user Input, restore concluded the turn
had never finished, and the harness re-drove a turn that had already run — at
full price and with its side effects.

Asserted while bitty is still running, deliberately: an orderly exit flushes the
BufWriter on drop and would hide the bug, and killing the harness is not a
portable test. A spawned script polls root's journal file from inside the
harness and reports what it found; the mock then fails the run if the text-only
turn's Output was not on disk. The script does the looking rather than the mock
because mail delivery itself flushes the recipient's journal, so anything that
inspects the disk only after being woken by mail would pass either way.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

JOURNAL = os.environ["BITTY_TEST_JOURNAL"]
MARKER = "flush-marker-9c41"

# Polls the journal from inside the harness and reports what it saw. The check
# has to happen before it mails root, because sending mail flushes the
# recipient's journal (system.rs) — a mock that waited for that mail and then
# looked at the disk itself would be testing the mail path, not the turn.
SCRIPT = """
const path = %s + "/proc-1.jsonl";
// Rebuilt from halves on purpose: spelled out, the marker would also appear in
// this script's own source, which the journal records verbatim as part of the
// spawn — and the probe would match that and pass on the broken code.
const marker = %s + %s;
const NL = String.fromCharCode(10);
bitty.onMail((): string => "ok");
(async () => {
  let seen = "no journal file";
  for (let i = 0; i < 50; i++) {
    await bitty.sleep(100);
    let text = "";
    try { text = bitty.fs.read(path); } catch (_e) { continue; }
    const lines = text.split(NL).filter((l) => l.includes('"event":"Output"'));
    seen = lines.length + " Output record(s) on disk";
    // A text block, not just the marker: the turn under test is the only one
    // that ended without tool calls.
    const done = lines.filter((l) =>
      l.includes(marker) && l.includes('"type":"text"')
    );
    if (done.length > 0) {
      bitty.send("proc-1", "flushed: " + seen);
      return;
    }
  }
  bitty.send("proc-1", "MISSING: " + seen);
})();
""" % (json.dumps(JOURNAL), json.dumps(MARKER[:6]), json.dumps(MARKER[6:]))


def sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def turn(blocks, stop_reason):
    ev = [{"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant",
           "content": [], "model": "claude-opus-5", "usage": {"input_tokens": 10}}}]
    for i, b in enumerate(blocks):
        if b[0] == "text":
            ev += [{"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}},
                   {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b[1]}}]
        else:
            ev += [{"type": "content_block_start", "index": i,
                    "content_block": {"type": "tool_use", "id": b[1], "name": b[2], "input": {}}},
                   {"type": "content_block_delta", "index": i,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(b[3])}}]
        ev.append({"type": "content_block_stop", "index": i})
    ev += [{"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 5}},
           {"type": "message_stop"}]
    return ev


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def fail(self, m):
        b = json.dumps({"type": "error", "error": {"type": "invalid_request_error",
                                                   "message": "ASSERTION: " + m}}).encode()
        self.send_response(400)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        messages = body["messages"]
        n = len([m for m in messages if m["role"] == "assistant"])
        if n == 0:
            ev = turn([("tool_use", "t1", "spawn_process",
                        {"name": "waker", "instructions": "x", "script": SCRIPT})], "tool_use")
        elif n == 1:
            r = [b for b in messages[-1].get("content", []) if isinstance(b, dict)
                 and b.get("type") == "tool_result"]
            if not r or r[0]["is_error"]:
                return self.fail(f"the waker should have spawned: {r}")
            # The turn under test: text only, no tool_use, so nothing after it
            # will flush the journal on the pre-fix code.
            ev = turn([("text", MARKER + "\n")], "end_turn")
        elif n == 2:
            woken = json.dumps(messages[-1].get("content"))
            if "MISSING" in woken or "flushed:" not in woken:
                return self.fail(
                    "a turn that ended without tool calls never reached disk — the waker "
                    f"polled proc-1.jsonl for 5s and saw: {woken[:400]}")
            ev = turn([("text", "Done.\n")], "end_turn")
        else:
            ev = turn([("text", "Idle.\n")], "end_turn")
        p = sse(ev)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(p)))
        self.end_headers()
        self.wfile.write(p)


ThreadingHTTPServer(("127.0.0.1", 8745), H).serve_forever()
