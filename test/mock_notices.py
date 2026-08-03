"""Mock verifying multicast fan-out, array stop targets, and link semantics.

Links are spawn_link-style: parent<->child only. A sibling is NOT told when
another sibling dies — the spawner holds the link, is told, and relays.

Topology: root spawns alpha, beta, gamma (linked by default; wired to each other
for messaging, which must NOT create links).
  root  : multicasts to ["proc-2","proc-3","proc-9"] -> partial success
          stops ["proc-4","proc-9"] in one call     -> partial success
          must receive the exit signal for gamma (it is linked)
  alpha : broadcasts "*" -> exactly its wiring (beta+gamma+root), never "user"
  beta  : waits on gamma. Must NOT be signalled when gamma dies — it is only
          wired to gamma, not linked to it.
  gamma : killed by root.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"killed": False, "root_signalled": False}


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
    """Raw text of a message's blocks. Must NOT go through json.dumps: the
    envelope contains quotes, and matching against escaped JSON silently never
    fires."""
    content = msg.get("content", [])
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text")



def flatten_system(system):
    """system is now a list of content blocks (general preamble first, then the
    process-specific block) so the shared prefix can be cached."""
    if isinstance(system, str):
        return system
    return " ".join(b.get("text", "") for b in system)

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def fail(self, m):
        b = json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": "ASSERTION: " + m}}).encode()
        self.send_response(400)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        system, messages = flatten_system(body["system"]), body["messages"]
        n = len([m for m in messages if m["role"] == "assistant"])
        last = json.dumps(messages[-1])

        if "process proc-1" in system:
            if n == 0:
                ev = turn([("tool_use", "t1", "spawn_topology", {"processes": [
                    {"name": "alpha", "instructions": "Broadcast a note to everyone you can reach.",
                     "can_send_to": ["parent", "beta", "gamma"]},
                    {"name": "beta", "instructions": "Wait for gamma's result, then report to parent.",
                     "can_send_to": ["parent", "gamma"]},
                    {"name": "gamma", "instructions": "Wait for instructions.",
                     "can_send_to": ["parent", "beta"]},
                ]})], "tool_use")
            elif n == 1:
                # Multicast with one bogus recipient -> partial success.
                ev = turn([("tool_use", "t2", "send_message",
                            {"to": ["proc-2", "proc-3", "proc-9"], "message": "kickoff"})], "tool_use")
            elif n == 2:
                r = results(messages[-1])[0]
                c = r["content"]
                if r["is_error"]:
                    return self.fail(f"partial multicast should not be a hard error: {c}")
                if "Delivered to proc-2, proc-3." not in c:
                    return self.fail(f"expected both live recipients delivered: {c}")
                if "Undeliverable" not in c or "proc-9" not in c:
                    return self.fail(f"expected proc-9 reported undeliverable: {c}")
                # Array stop, with one bogus id, while beta waits on gamma.
                ev = turn([("tool_use", "t3", "stop_process",
                            {"targets": ["proc-4", "proc-9"]})], "tool_use")
            elif n == 3:
                r = results(messages[-1])[0]
                c = r["content"]
                if r["is_error"]:
                    return self.fail(f"partial stop should not be a hard error: {c}")
                if "Stopped: proc-4." not in c:
                    return self.fail(f"expected proc-4 stopped: {c}")
                if "Unknown" not in c or "proc-9" not in c:
                    return self.fail(f"expected proc-9 reported unknown: {c}")
                # Confirmed stopped — only now may a call from gamma be a bug.
                STATE["killed"] = True
                # The exit signal rides in on the SAME user turn as the tool
                # results, as an interrupt — that is where it must be asserted.
                incoming = texts(messages[-1])
                if "exit_signal" not in incoming:
                    return self.fail(
                        f"root holds gamma's link and must be signalled: {incoming!r}")
                for needle in ["proc-4", "gamma", "stopped by proc-1", "linked"]:
                    if needle not in incoming:
                        return self.fail(f"exit signal missing {needle!r}: {incoming!r}")
                STATE["root_signalled"] = True
                ev = turn([("text", "Gamma stopped.\n")], "end_turn")
            else:
                # Root holds the link to gamma, so root — not beta — is signalled.
                incoming = texts(messages[-1])
                if "exit_signal" in incoming:
                    STATE["root_signalled"] = True
                    if "proc-4" not in incoming or "gamma" not in incoming:
                        return self.fail(f"exit signal must name the process: {incoming!r}")
                    if "stopped by proc-1" not in incoming:
                        return self.fail(f"exit signal must give a reason: {incoming!r}")
                    if "linked" not in incoming:
                        return self.fail(f"exit signal should explain the link: {incoming!r}")
                ev = turn([("text", "Done.\n")], "end_turn")

        elif "process proc-2" in system:  # alpha
            if n == 0:
                ev = turn([("tool_use", "a1", "send_message", {"to": "*", "message": "hello all"})], "tool_use")
            elif n == 1:
                c = results(messages[-1])[0]["content"]
                # "*" must resolve to exactly alpha's wiring: parent + beta + gamma.
                for expected in ["proc-1", "proc-3", "proc-4"]:
                    if expected not in c:
                        return self.fail(f"'*' missed {expected}: {c}")
                if "user" in c:
                    return self.fail(f"'*' must not include the human console: {c}")
                ev = turn([("text", "Broadcast sent.\n")], "end_turn")
            else:
                ev = turn([("text", "idle.\n")], "end_turn")

        elif "process proc-3" in system:  # beta — wired to gamma, not linked to it
            incoming = texts(messages[-1])
            if "exit_signal" in incoming:
                return self.fail(
                    "beta is only WIRED to gamma, not LINKED to it — it must not receive an "
                    f"exit signal; only the spawner holds the link: {incoming!r}")
            ev = turn([("text", "Waiting on gamma.\n")], "end_turn")

        elif "process proc-4" in system:  # gamma — gets killed
            if STATE["killed"]:
                return self.fail("gamma was stopped; it must not call the API again")
            ev = turn([("text", "Standing by.\n")], "end_turn")
        else:
            return self.fail("unknown process")

        p = sse(ev)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(p)))
        self.end_headers()
        self.wfile.write(p)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8737), H).serve_forever()
