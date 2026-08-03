"""Filesystem capabilities: grant, attenuate, and refuse escapes.

root is started with --allow-read/--allow-write on a repo directory only.
It spawns a script that reads a file inside the repo (allowed), tries to read
outside it via '..' (must be refused by canonicalization), and root then tries
to grant a child access outside its own root (must be rejected at spawn).
"""
import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.environ["BITTY_TEST_REPO"]
SECRET = os.environ["BITTY_TEST_SECRET"]

READER = """
bitty.onMail((mail, api): string => {
  if (mail.body === "read") {
    const text: string = api.fs.read(api.repo + "/src/main.rs");
    return `ok:${text.trim().length}`;
  }
  try {
    api.fs.read(api.repo + "/../secret/keys.txt");
    return "ESCAPED";
  } catch (e) {
    return `refused:${e instanceof Error ? e.message : String(e)}`;
  }
});
"""


def sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def turn(blocks, stop_reason):
    ev = [{"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant",
           "content": [], "model": "claude-opus-5", "usage": {"input_tokens": 10}}}]
    for i, b in enumerate(blocks):
        if b[0] == "text":
            ev.append({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}})
            ev.append({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b[1]}})
        else:
            ev.append({"type": "content_block_start", "index": i, "content_block": {"type": "tool_use", "id": b[1], "name": b[2], "input": {}}})
            ev.append({"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": json.dumps(b[3])}})
        ev.append({"type": "content_block_stop", "index": i})
    ev.append({"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 5}})
    ev.append({"type": "message_stop"})
    return ev


def results(msg):
    return [b for b in msg.get("content", []) if isinstance(b, dict) and b.get("type") == "tool_result"]


def flatten_system(system):
    return system if isinstance(system, str) else " ".join(b.get("text", "") for b in system)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def fail(self, m):
        b = json.dumps({"type": "error", "error": {"type": "invalid_request_error",
                                                   "message": "ASSERTION: " + m}}).encode()
        self.send_response(400); self.send_header("content-type", "application/json")
        self.end_headers(); self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        system, messages = flatten_system(body["system"]), body["messages"]
        n = len([m for m in messages if m["role"] == "assistant"])

        if n == 0:
            if "Reading files under" not in system or REPO not in system:
                return self.fail(f"root's prompt should name its granted roots: {system[-400:]!r}")
            if SECRET in system:
                return self.fail("root must not hold the secret directory")
            src = READER.replace("api.repo", json.dumps(REPO))
            ev = turn([("tool_use", "t1", "spawn_process",
                        {"name": "reader", "instructions": "read files",
                         "script": src, "can_write": []})], "tool_use")
        elif n == 1:
            if results(messages[-1])[0]["is_error"]:
                return self.fail(f"spawn failed: {results(messages[-1])[0]['content']}")
            ev = turn([("tool_use", "t2", "call_process",
                        {"process_id": "proc-2", "message": "read"})], "tool_use")
        elif n == 2:
            r = results(messages[-1])[0]
            if r["is_error"] or not r["content"].startswith("ok:"):
                return self.fail(f"granted read should succeed: {r}")
            ev = turn([("tool_use", "t3", "call_process",
                        {"process_id": "proc-2", "message": "escape"})], "tool_use")
        elif n == 3:
            r = results(messages[-1])[0]
            if "ESCAPED" in r["content"]:
                return self.fail(f"'..' escaped the granted root: {r['content']!r}")
            if not r["content"].startswith("refused:"):
                return self.fail(f"expected a refusal, got {r['content']!r}")
            # Root cannot confer access it does not hold.
            ev = turn([("tool_use", "t4", "spawn_process",
                        {"name": "thief", "instructions": "peek",
                         "script": "bitty.onMail(() => 'x');", "can_read": [SECRET]})], "tool_use")
        elif n == 4:
            r = results(messages[-1])[0]
            if not r["is_error"]:
                return self.fail("granting a root outside your own must be rejected")
            if "never be granted more authority" not in r["content"]:
                return self.fail(f"rejection should cite attenuation: {r['content']!r}")
            ev = turn([("text", "Done.\n")], "end_turn")
        else:
            ev = turn([("text", "Idle.\n")], "end_turn")

        p = sse(ev)
        self.send_response(200); self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(p))); self.end_headers(); self.wfile.write(p)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8741), H).serve_forever()
