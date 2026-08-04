"""Myopic processes: a worker that holds tools and knows of no graph at all.

root spawns a store (script), a reader given a typed `lookup` tool and
can_send_to: [], and a boss whose own reach is narrow. The reader must see
exactly its alias — no send_message, no list_processes, and no mention of which
process answers it — and the alias must still work, because an alias carries its
own authority. The boss must not be able to hand out an alias reaching further
than the boss itself can.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STORE = """
bitty.onMail((mail): string => {
  if (mail.body === "spawn") {
    const ids = bitty.spawn({ name: "hatchling", instructions: "wait", can_send_to: [] });
    return "spawned:" + ids.join(",");
  }
  const args = JSON.parse(mail.body);
  return args.key === "answer" ? "42" : "unknown";
});
"""
SCHEMA = {"type": "object",
          "properties": {"key": {"type": "string"}},
          "required": ["key"]}


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
        tools = body["tools"]
        names = [t["name"] for t in tools]

        if "process proc-3" in system:       # the reader: tools, and no graph
            # Self-stop survives every narrowing on purpose, and names nobody
            # but itself, so it is not a graph leak. Everything that could name
            # another process must be gone.
            leaked = [t for t in names if t not in ("lookup", "stop_process")]
            if leaked:
                return self.fail(f"a myopic process should see only its own tools: {leaked}")
            if "lookup" not in names:
                return self.fail(f"the alias must survive having no messaging: {names}")
            desc = tools[0]["description"]
            if "proc-" in desc:
                return self.fail(f"the alias must not name the process behind it: {desc!r}")
            # The permissions section should not advertise a roster either.
            if "list_processes" in system or "send_message" in system:
                return self.fail("a myopic process should not be told about messaging tools")
            if n == 0:
                ev = turn([("tool_use", "r1", "lookup", {"key": "answer"})], "tool_use")
            elif n == 1:
                r = results(messages[-1])[0]
                if r["is_error"] or r["content"].strip() != "42":
                    return self.fail(f"an alias carries its own authority and must work: {r}")
                ev = turn([("text", "Found it.\n")], "end_turn")
            else:
                ev = turn([("text", "Idle.\n")], "end_turn")

        elif "process proc-4" in system:      # the boss: narrow reach of its own
            if n == 0:
                # Pointing an alias past the spawner's own reach is laundering.
                ev = turn([("tool_use", "b1", "spawn_process",
                            {"name": "proxy", "instructions": "x", "can_send_to": [],
                             "tools": [{"name": "tell_user", "description": "d",
                                        "target": "user"}]})], "tool_use")
            elif n == 1:
                r = results(messages[-1])[0]
                if not r["is_error"]:
                    return self.fail("an alias must not reach past the spawner's own authority")
                if "not permitted to message" not in r["content"]:
                    return self.fail(f"rejection should explain: {r['content']!r}")
                ev = turn([("text", "Refused, as it should be.\n")], "end_turn")
            else:
                ev = turn([("text", "Idle.\n")], "end_turn")

        elif "ask" in names:        # the mute worker spawned below: nothing to do
            ev = turn([("text", "Idle.\n")], "end_turn")

        elif "lookup" in names:
            return self.fail("only the reader should see the alias")

        elif "process proc-1" not in system:   # anything else spawned: idle
            ev = turn([("text", "Idle.\n")], "end_turn")

        elif n == 0:
            ev = turn([("tool_use", "t1", "spawn_topology", {"processes": [
                {"name": "store", "instructions": "key-value store", "script": STORE,
                 "can_send_to": ["parent", "reader"]},
                {"name": "reader", "instructions": "Look up 'answer' and report what you find.",
                 "can_send_to": [], "can_stop": [], "can_spawn": False,
                 "tools": [{"name": "lookup", "description": "Look up a key.",
                            "input_schema": SCHEMA, "target": "store"}]},
                {"name": "boss", "instructions": "Try to give a helper a tool that reaches the user.",
                 "can_send_to": ["parent"]},
            ]})], "tool_use")
        elif n == 1:
            r = results(messages[-1])[0]
            if r["is_error"]:
                return self.fail(f"topology failed: {r['content']}")
            # A myopic worker is a normal thing to spawn, not an error.
            # A script must be able to create processes itself.
            ev = turn([("tool_use", "t8", "call_process",
                        {"process_id": "proc-2", "message": "spawn", "timeout_seconds": 20})],
                      "tool_use")
        elif n == 2:
            r = results(messages[-1])[0]
            if r["is_error"] or "spawned:proc-" not in r["content"]:
                return self.fail(f"a script should be able to spawn: {r}")
            ev = turn([("tool_use", "t2", "spawn_process",
                        {"name": "solo", "instructions": "y", "can_send_to": [],
                         "tools": [{"name": "ask", "description": "d", "target": "parent"}]})],
                      "tool_use")
        elif n == 3:
            r = results(messages[-1])[0]
            if r["is_error"]:
                return self.fail(f"a mute worker holding tools must be allowed: {r['content']}")
            if "ask" not in r["content"]:
                return self.fail(f"the spawn result should name the tools it created: {r['content']!r}")
            if "no messaging of its own" not in r["content"]:
                return self.fail(f"the spawn result should say it cannot reply: {r['content']!r}")
            ev = turn([("text", "Done.\n")], "end_turn")
        else:
            ev = turn([("text", "Idle.\n")], "end_turn")

        p = sse(ev)
        self.send_response(200); self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(p))); self.end_headers(); self.wfile.write(p)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8756), H).serve_forever()
