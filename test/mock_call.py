"""Mock verifying synchronous calls into a script, and code replacement.

root spawns an adder script, calls it (answer arrives inside the same turn),
then patches the script to multiply instead — same process id — and calls again.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ADD = """
bitty.onMail((mail, _api): string => {
  const parts: number[] = mail.body.split("+").map((x: string) => Number(x.trim()));
  return String(parts.reduce((a, b) => a + b, 0));
});
"""
MUL = """
bitty.onMail((mail, _api): string => {
  const parts: number[] = mail.body.split("+").map((x: string) => Number(x.trim()));
  return String(parts.reduce((a, b) => a * b, 1));
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
        if "process proc-2" in system:
            return self.fail("the script must never call the model API")

        if n == 0:
            ev = turn([("tool_use", "t1", "spawn_process",
                        {"name": "calc", "instructions": "arithmetic", "script": ADD})], "tool_use")
        elif n == 1:
            if results(messages[-1])[0]["is_error"]:
                return self.fail("script spawn failed")
            ev = turn([("tool_use", "t2", "call_process",
                        {"process_id": "proc-2", "message": "2 + 3 + 4"})], "tool_use")
        elif n == 2:
            r = results(messages[-1])[0]
            if r["is_error"]:
                return self.fail(f"call failed: {r['content']}")
            if r["content"].strip() != "9":
                return self.fail(f"expected the sum 9 inline, got {r['content']!r}")
            ev = turn([("tool_use", "t3", "patch_script",
                        {"process_id": "proc-2", "script": MUL})], "tool_use")
        elif n == 3:
            r = results(messages[-1])[0]
            if r["is_error"]:
                return self.fail(f"patch failed: {r['content']}")
            if "id, mailbox, links and permissions are unchanged" not in r["content"]:
                return self.fail(f"patch should confirm identity is kept: {r['content']!r}")
            ev = turn([("tool_use", "t4", "call_process",
                        {"process_id": "proc-2", "message": "2 + 3 + 4"})], "tool_use")
        elif n == 4:
            r = results(messages[-1])[0]
            if r["is_error"]:
                return self.fail(f"call after patch failed: {r['content']}")
            if r["content"].strip() != "24":
                return self.fail(
                    f"patched code should multiply to 24 at the SAME id, got {r['content']!r}")
            ev = turn([("tool_use", "t5", "call_process",
                        {"process_id": "proc-9", "message": "hi"})], "tool_use")
        elif n == 5:
            r = results(messages[-1])[0]
            if not r["is_error"] or "No process" not in r["content"]:
                return self.fail(f"calling an absent process should fail cleanly: {r}")
            ev = turn([("text", "Done.\n")], "end_turn")
        else:
            ev = turn([("text", "Idle.\n")], "end_turn")

        p = sse(ev)
        self.send_response(200); self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(p))); self.end_headers(); self.wfile.write(p)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8740), H).serve_forever()
