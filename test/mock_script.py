"""Mock verifying script processes: an embedded-Deno TypeScript actor.

root (agent, mocked) spawns a script process, messages it, and must receive the
script's reply. The script source carries a TypeScript type annotation, so this
also proves transpilation runs before V8 sees the code.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPT = """
// TypeScript annotations must survive transpilation.
bitty.onMail(async (mail: {from: string; body: string}, api): Promise<void> => {
  const n: number = mail.body.length;
  // Console output is a structured process log, never a raw terminal write.
  console.log(`counted ${n} characters from ${mail.from}`);
  await api.send(api.parent, `len=${n}`);
});
"""

STATE = {"replied": False}


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


def texts(msg):
    c = msg.get("content", [])
    if isinstance(c, str):
        return c
    return " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")


def flatten_system(system):
    if isinstance(system, str):
        return system
    return " ".join(b.get("text", "") for b in system)


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
        system, messages = flatten_system(body["system"]), body["messages"]
        n = len([m for m in messages if m["role"] == "assistant"])

        if "process proc-2" in system:
            return self.fail("a script process must never call the model API")

        if n == 0:
            ev = turn([("tool_use", "t1", "spawn_process",
                        {"name": "counter", "instructions": "count characters",
                         "script": SCRIPT})], "tool_use")
        elif n == 1:
            r = results(messages[-1])[0]
            if r["is_error"]:
                return self.fail(f"script spawn failed: {r['content']}")
            ev = turn([("tool_use", "t2", "send_message",
                        {"to": "proc-2", "message": "hello world"})], "tool_use")
        elif n == 2:
            r = results(messages[-1])[0]
            if r["is_error"]:
                return self.fail(f"could not message the script: {r['content']}")
            ev = turn([("text", "Waiting for the script.\n")], "end_turn")
        elif not STATE["replied"]:
            incoming = texts(messages[-1])
            if "len=11" not in incoming:
                return self.fail(f"expected the script's computed reply len=11: {incoming!r}")
            if 'from="proc-2"' not in incoming:
                return self.fail(f"reply should come from the script process: {incoming!r}")
            STATE["replied"] = True
            ev = turn([("tool_use", "t3", "stop_process", {"targets": "proc-2"})], "tool_use")
        else:
            # Later turns carry the exit signal for the stopped script; the
            # reply assertion above has already run and must not re-run.
            ev = turn([("text", "Done.\n")], "end_turn")

        p = sse(ev)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(p)))
        self.end_headers()
        self.wfile.write(p)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8739), H).serve_forever()
