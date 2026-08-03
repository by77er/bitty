"""Mock verifying the capability model: attenuation, rejection, spawn grant.

root spawns a topology:
  worker    can_send_to ["parent"]                 -> may message only root
  leaf      can_send_to ["parent"], can_spawn false -> may not spawn at all

worker then:
  1. spawns a child with NO can_send_to  -> child must INHERIT worker's narrow
     grants, not become unrestricted. This is the escalation hole: before the
     capability model, that child could message and stop anything.
  2. asks for a child that may message "user" -> REJECTED, since worker itself
     cannot message the user. Over-requesting is an error, not a silent trim.
leaf then:
  3. tries to spawn at all -> REJECTED, it does not hold the spawn capability.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"checks": set()}


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

        if "process proc-1" in system:          # root, unrestricted
            if n == 0:
                # Root holds every in-harness capability but no run/fs grant,
                # and must be *told* so rather than having to discover it by
                # trying and failing.
                if "Messaging: any process" not in system:
                    return self.fail("root should be unrestricted in messaging")
                if "Running: not permitted" not in system:
                    return self.fail(
                        f"root was given no --allow-run and must be told: {system[-300:]!r}")
                ev = turn([("tool_use", "t1", "spawn_topology", {"processes": [
                    {"name": "worker", "instructions": "Delegate a subtask.",
                     "can_send_to": ["parent"]},
                    {"name": "leaf", "instructions": "Do a small thing.",
                     "can_send_to": ["parent"], "can_spawn": False},
                ]})], "tool_use")
            else:
                ev = turn([("text", "Waiting.\n")], "end_turn")

        elif "process proc-2" in system:        # worker: send limited to root
            if n == 0:
                for needle in ["Your permissions", "Messaging: only proc-1"]:
                    if needle not in system:
                        return self.fail(f"worker prompt missing {needle!r}: {system[-400:]!r}")
                if "Stopping: only proc-2" not in system:
                    return self.fail("worker should default to stopping only itself")
                ev = turn([("tool_use", "w1", "spawn_process",
                            {"instructions": "Sub-task.", "name": "grandchild"})], "tool_use")
            elif n == 1:
                r = results(messages[-1])[0]
                if r["is_error"]:
                    return self.fail(f"plain spawn should succeed: {r['content']}")
                STATE["checks"].add("inherit_spawned")
                # Over-request: worker cannot message the user, so it cannot
                # confer that on a child.
                ev = turn([("tool_use", "w2", "spawn_process",
                            {"instructions": "Talk to the human.", "name": "proxy",
                             "can_send_to": ["user"]})], "tool_use")
            elif n == 2:
                r = results(messages[-1])[0]
                c = r["content"]
                if not r["is_error"]:
                    return self.fail(f"over-request must be rejected, got success: {c}")
                if "never be granted more authority" not in c:
                    return self.fail(f"rejection should explain attenuation: {c}")
                if "user" not in c:
                    return self.fail(f"rejection should name the refused target: {c}")
                STATE["checks"].add("over_request_rejected")
                ev = turn([("tool_use", "w3", "list_processes", {})], "tool_use")
            elif n == 3:
                c = results(messages[-1])[0]["content"]
                # Namespace: self, own child, spawner — but NOT the sibling leaf.
                for expected in ["proc-2", "proc-4", "proc-1"]:
                    if expected not in c:
                        return self.fail(f"worker should see {expected}: {c!r}")
                if "proc-3" in c:
                    return self.fail(
                        f"LEAK: worker can see its sibling proc-3 (leaf), which is outside "
                        f"its namespace: {c!r}")
                STATE["checks"].add("namespaced_list")
                # An id outside the view must read as absent, not as forbidden.
                ev = turn([("tool_use", "w4", "send_message",
                            {"to": "proc-3", "message": "hello sibling"})], "tool_use")
            elif n == 4:
                c = results(messages[-1])[0]["content"]
                if "no such process" not in c:
                    return self.fail(
                        f"out-of-view id should read as absent, not forbidden: {c!r}")
                if "Not permitted" in c:
                    return self.fail(f"absence must not be reported as a permission error: {c!r}")
                STATE["checks"].add("out_of_view_is_absent")
                ev = turn([("text", "Understood.\n")], "end_turn")
            else:
                ev = turn([("text", "Idle.\n")], "end_turn")

        elif "process proc-3" in system:        # leaf: spawn denied
            if n == 0:
                if "Spawning: not permitted" not in system:
                    return self.fail(f"leaf prompt should show spawning denied: {system[-300:]!r}")
                ev = turn([("tool_use", "l1", "spawn_process",
                            {"instructions": "sneak a helper", "name": "helper"})], "tool_use")
            elif n == 1:
                r = results(messages[-1])[0]
                if not r["is_error"]:
                    return self.fail(f"leaf must not be able to spawn: {r['content']}")
                if "spawn capability" not in r["content"]:
                    return self.fail(f"denial should cite the capability: {r['content']}")
                STATE["checks"].add("spawn_denied")
                ev = turn([("text", "Cannot spawn.\n")], "end_turn")
            else:
                ev = turn([("text", "Idle.\n")], "end_turn")

        elif "process proc-4" in system:        # grandchild of worker
            # The escalation check: it must NOT be unrestricted.
            if "Your permissions" not in system:
                return self.fail(
                    "ESCALATION: grandchild of a restricted worker is unrestricted — "
                    f"prompt has no permissions block: {system[-400:]!r}")
            if "Messaging: any process" in system:
                return self.fail("ESCALATION: grandchild may message any process")
            if "proc-1" not in system or "proc-2" not in system:
                return self.fail(f"grandchild should inherit worker's reach: {system[-400:]!r}")
            STATE["checks"].add("no_escalation")
            ev = turn([("text", "Working within my grants.\n")], "end_turn")
        else:
            idx=system.find("You are process")
            return self.fail(f"unexpected process: {system[idx:idx+90]!r} n={n} tools={[t['name'] for t in b['tools']][:3]}")

        p = sse(ev)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(p)))
        self.end_headers()
        self.wfile.write(p)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8738), H).serve_forever()
