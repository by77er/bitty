"""Tool aliases: a typed tool that is really a call to another process.

root spawns a script that answers arithmetic, plus a worker given a typed
`add` tool pointing at it. The worker calls `add` like any tool. The test also
checks that bad arguments are refused before delivery, and that an alias
pointing somewhere the worker may not message is rejected at spawn.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CALC = """
bitty.onMail((mail): string => {
  const args = JSON.parse(mail.body);
  return String(Number(args.a) + Number(args.b));
});
"""
SCHEMA = {"type": "object",
          "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
          "required": ["a", "b"]}


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
        names = [t["name"] for t in body["tools"]]

        if "process proc-3" in system:      # the worker holding the alias
            if "add" not in names:
                return self.fail(f"the worker should see its alias as a tool: {names}")
            if not any(t.get("cache_control") for t in body["tools"]):
                return self.fail("base tools should carry a cache breakpoint before aliases")
            if names.index("add") != len(names) - 1:
                return self.fail(f"aliases must come after the shared base tools: {names}")
            if n == 0:
                ev = turn([("tool_use", "w1", "add", {"a": 2, "b": 40})], "tool_use")
            elif n == 1:
                r = results(messages[-1])[0]
                if r["is_error"] or r["content"].strip() != "42":
                    return self.fail(f"alias should return the computed answer: {r}")
                ev = turn([("tool_use", "w2", "add", {"a": "two"})], "tool_use")
            elif n == 2:
                r = results(messages[-1])[0]
                if not r["is_error"]:
                    return self.fail("bad arguments must not be delivered")
                if "missing required field 'b'" not in r["content"]:
                    return self.fail(f"validation should name the problem: {r['content']!r}")
                ev = turn([("text", "Alias works.\n")], "end_turn")
            else:
                ev = turn([("text", "Idle.\n")], "end_turn")

        elif "add" in names:
            return self.fail("only the worker should see the alias")

        elif n == 0:
            ev = turn([("tool_use", "t1", "spawn_topology", {"processes": [
                {"name": "calc", "instructions": "arithmetic", "script": CALC,
                 "can_send_to": ["parent", "worker"]},
                {"name": "worker", "instructions": "Use your add tool on 2 and 40.",
                 "can_send_to": ["parent", "calc"],
                 "tools": [{"name": "add", "description": "Add two numbers.",
                            "input_schema": SCHEMA, "target": "calc"}]},
            ]})], "tool_use")
        elif n == 1:
            if results(messages[-1])[0]["is_error"]:
                return self.fail(f"topology failed: {results(messages[-1])[0]['content']}")
            # An alias pointing where the holder may not message must be refused.
            ev = turn([("tool_use", "t2", "spawn_process",
                        {"name": "sneak", "instructions": "x", "can_send_to": [],
                         "tools": [{"name": "reach", "description": "d", "target": "parent"}]})],
                      "tool_use")
        elif n == 2:
            r = results(messages[-1])[0]
            if not r["is_error"]:
                return self.fail("an alias must not grant reach the permissions deny")
            if "not permitted to message" not in r["content"]:
                return self.fail(f"rejection should explain: {r['content']!r}")
            ev = turn([("text", "Done.\n")], "end_turn")
        else:
            ev = turn([("text", "Idle.\n")], "end_turn")

        p = sse(ev)
        self.send_response(200); self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(p))); self.end_headers(); self.wfile.write(p)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8742), H).serve_forever()
