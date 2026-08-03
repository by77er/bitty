"""Deno.serve inside a script process.

root spawns a script that serves on 127.0.0.1:8899, then runs an inline script
that fetches it. The request travels: inline isolate -> op_bitty_fetch -> the
listening socket on the main runtime -> mail to the script process -> its
handler -> back out as an HTTP response. Also checks that serving a port the
net grant does not cover is refused.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SERVER = """
Deno.serve({ port: 8899, hostname: "127.0.0.1" }, async (req: Request): Promise<Response> => {
  const url = new URL(req.url);
  if (url.pathname === "/echo") {
    return Response.json({ echoed: await req.text(), method: req.method });
  }
  return new Response("hello from " + bitty.id, { headers: { "x-served-by": "bitty" } });
});
"""

# Bounded retry: the script process boots on its own thread, so the socket may
# not be listening the instant the spawn call returns.
CLIENT = """
const deadline = Date.now() + 10000;
let last = "";
while (Date.now() < deadline) {
  try {
    const r = bitty.fetch("http://127.0.0.1:8899/");
    const e = bitty.fetch("http://127.0.0.1:8899/echo", { method: "POST", body: "ping" });
    return r.status + " " + r.body + " | " + e.body;
  } catch (err) { last = String(err); }
}
return "never came up: " + last;
"""

DENIED = """
Deno.serve({ port: 9999, hostname: "127.0.0.1" }, () => new Response("nope"));
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

        if "process proc-1" not in system:
            return self.fail("only the root should be taking turns here")

        if n == 0:
            ev = turn([("tool_use", "t1", "spawn_process",
                        {"name": "web", "instructions": "serve", "script": SERVER})], "tool_use")
        elif n == 1:
            r = results(messages[-1])[0]
            if r["is_error"]:
                return self.fail(f"spawning the server failed: {r['content']}")
            ev = turn([("tool_use", "t2", "run_script", {"script": CLIENT})], "tool_use")
        elif n == 2:
            r = results(messages[-1])[0]
            if r["is_error"]:
                return self.fail(f"fetching the served port failed: {r['content']}")
            got = r["content"]
            if "200" not in got or "hello from proc-2" not in got:
                return self.fail(f"the handler's response should come back: {got!r}")
            if '"echoed":"ping"' not in got.replace(" ", ""):
                return self.fail(f"a POST body should reach the handler: {got!r}")
            # A port outside the net grant must be refused at bind time.
            ev = turn([("tool_use", "t3", "run_script", {"script": DENIED})], "tool_use")
        elif n == 3:
            r = results(messages[-1])[0]
            if not r["is_error"]:
                return self.fail("serving a port outside the net grant must be refused")
            if "not permitted to serve" not in r["content"]:
                return self.fail(f"refusal should explain: {r['content']!r}")
            ev = turn([("text", "Served.\n")], "end_turn")
        else:
            ev = turn([("text", "Idle.\n")], "end_turn")

        p = sse(ev)
        self.send_response(200); self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(p))); self.end_headers(); self.wfile.write(p)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8757), H).serve_forever()
