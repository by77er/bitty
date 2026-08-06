"""Mock Messages API scripting a topology scenario for the bitty harness.

Scenario:
  root spawns a 2-node topology:
    writer  (role, context=inherit, can_send_to=["editor"])
    editor  (role, context=empty,   can_send_to=["parent", "user"])
  writer drafts -> messages editor -> tries to message root (DENIED) -> stops itself
  editor polishes -> messages user -> messages root -> stops itself
  root stops itself.

Server-side assertions (surface as HTTP 400 in the harness):
  - writer's system prompt must contain its role text and its wiring allowlist
  - writer's first turn must contain the inherited transcript of root's convo
  - editor's first turn must NOT contain inherited context
  - writer's send to root must come back is_error with "Not permitted"
  - writer must not be able to stop the editor
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = open("/tmp/claude-1000/-home-bit-Code-Bitty/29b2838b-63ef-4bac-9206-4b943883dc56/scratchpad/topology_requests.jsonl", "w")


def sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def turn(blocks, stop_reason):
    events = [{"type": "message_start", "message": {"id": "msg_mock", "type": "message", "role": "assistant", "content": [], "model": "claude-opus-5"}}]
    for i, b in enumerate(blocks):
        kind = b[0]
        if kind == "text":
            events.append({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}})
            events.append({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b[1]}})
        elif kind == "tool_use":
            events.append({"type": "content_block_start", "index": i, "content_block": {"type": "tool_use", "id": b[1], "name": b[2], "input": {}}})
            events.append({"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": json.dumps(b[3])}})
        events.append({"type": "content_block_stop", "index": i})
    events.append({"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 10}})
    events.append({"type": "message_stop"})
    return events


def results_of(msg):
    return [b for b in msg.get("content", []) if isinstance(b, dict) and b.get("type") == "tool_result"]



def flatten_system(system):
    """system is now a list of content blocks (general preamble first, then the
    process-specific block) so the shared prefix can be cached."""
    if isinstance(system, str):
        return system
    return " ".join(b.get("text", "") for b in system)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def fail(self, msg):
        body = json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": "ASSERTION: " + msg}}).encode()
        self.send_response(400)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        LOG.write(json.dumps(body) + "\n")
        LOG.flush()
        system = flatten_system(body["system"])
        messages = body["messages"]
        n = len([m for m in messages if m["role"] == "assistant"])
        first_turn = json.dumps(messages[0])

        if "process proc-1" in system:
            if body.get("output_config", {}).get("effort") != "high":
                return self.fail(f"root keeps its own explicit effort: {body.get('output_config')}")
            if n == 0:
                blocks = [("tool_use", "t1", "spawn_topology", {"processes": [
                    {"name": "writer", "role": "You are a terse technical writer.",
                     "instructions": "Draft two sentences on actor systems, send to editor.",
                     "context": "inherit", "can_send_to": ["editor"]},
                    {"name": "editor", "role": "You are a copy editor. You cut adverbs.",
                     "instructions": "Polish the writer's draft, deliver to the user, then report to parent.",
                     "can_send_to": ["parent", "user"]},
                ]})]
                events = turn(blocks, "tool_use")
            elif n == 1:
                r = results_of(messages[-1])
                if not r or r[0]["is_error"]:
                    return self.fail(f"topology spawn failed: {r}")
                if "writer = proc-2" not in r[0]["content"] or "editor = proc-3" not in r[0]["content"]:
                    return self.fail(f"expected name=id wiring in result, got {r[0]['content']}")
                events = turn([("text", "Topology is up; waiting on the editor.\n")], "end_turn")
            else:
                events = turn([("tool_use", "t2", "stop_process", {"process_id": "proc-1", "cascade": True})], "tool_use")

        elif "process proc-2" in system:
            if n == 0:
                if "terse technical writer" not in system:
                    return self.fail("writer's role text missing from its system prompt")
                # Spawned effort is low unless explicitly raised — a worker
                # must not quietly inherit its coordinator's effort level.
                if body.get("output_config", {}).get("effort") != "low":
                    return self.fail(f"a spawned node should default to low effort: {body.get('output_config')}")
                if "proc-3 (editor)" not in system:
                    return self.fail("writer's wiring allowlist missing from its system prompt")
                if "inherited_context" not in first_turn or "spawn_topology" not in first_turn:
                    return self.fail("writer should have inherited root's transcript")
                blocks = [("tool_use", "w1", "send_message", {"process_id": "proc-3", "message": "Draft: actors are isolated. Messages are the only coupling."})]
                events = turn(blocks, "tool_use")
            elif n == 1:
                # Not on the allowlist — must be denied.
                events = turn([("tool_use", "w2", "send_message", {"process_id": "proc-1", "message": "sneaking a status update to root"})], "tool_use")
            elif n == 2:
                r = results_of(messages[-1])
                if not r or not r[0]["is_error"] or "Not permitted" not in r[0]["content"]:
                    return self.fail(f"send outside allowlist should have been denied, got {r}")
                events = turn([("tool_use", "w3", "stop_process", {"process_id": "proc-3"})], "tool_use")
            elif n == 3:
                r = results_of(messages[-1])
                if not r or not r[0]["is_error"] or "may only stop" not in r[0]["content"] \
                        or "yourself" not in r[0]["content"]:
                    return self.fail(f"restricted process should not stop a peer, got {r}")
                events = turn([("tool_use", "w4", "stop_process", {"process_id": "proc-2"})], "tool_use")
            else:
                return self.fail("writer stopped itself; it must not call the API again")

        elif "process proc-3" in system:
            if n == 0:
                if "copy editor" not in system:
                    return self.fail("editor's role text missing from its system prompt")
                if "inherited_context" in first_turn:
                    return self.fail("editor used context=empty but received inherited context")
                events = turn([("text", "Waiting for a draft.\n")], "end_turn")
            elif n == 1:
                if "proc-2" not in json.dumps(messages[-1]):
                    return self.fail("editor should have received the writer's mail")
                blocks = [
                    ("tool_use", "e1", "send_message", {"process_id": "user", "message": "Actors are isolated. Messages are the only coupling."}),
                    ("tool_use", "e2", "send_message", {"process_id": "proc-1", "message": "Delivered to the user."}),
                    ("tool_use", "e3", "stop_process", {"process_id": "proc-3"}),
                ]
                events = turn(blocks, "tool_use")
            else:
                return self.fail("editor stopped itself; it must not call the API again")
        else:
            return self.fail("unknown process in system prompt")

        payload = sse(events)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8735), Handler).serve_forever()
