"""Script processes must come back alive after a harness restart.

A script keeps its behaviour in the isolate, so a restart has to re-run its
source or the process returns as a shell that answers nothing. This runs in two
phases against one journal: phase one spawns a script that records each boot to
a file, phase two resumes and checks the script booted a second time, knows it
was resumed, and still answers mail.

The failure this pins: the spawn record was buffered and only flushed at turn
boundaries, and a script has no turns — so its journal file stayed empty and the
process was never restored at all.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MARK = os.environ.get("BITTY_RESTART_MARK", "/tmp/bitty-restart-mark")
PHASE = os.environ.get("BITTY_RESTART_PHASE", "1")
# Phase two resumes a restored history, so the assistant-turn count already
# includes phase one. Count this server's own requests instead.
SEEN = {"n": 0}

SCRIPT = """
const path = %s;
let prior = "";
try { prior = bitty.fs.read(path); } catch (_e) { prior = ""; }
bitty.fs.write(path, prior + (bitty.resumed ? "resumed" : "fresh") + "\\n");
let seen = 0;
bitty.onMail((mail): string => { seen++; return "alive seen=" + seen; });
""" % json.dumps(MARK)


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
            return self.respond(turn([("text", "idle\n")], "end_turn"))

        if PHASE == "1":
            if n == 0:
                ev = turn([("tool_use", "t1", "spawn_process",
                            {"name": "keeper", "instructions": "hold", "script": SCRIPT})], "tool_use")
            elif n == 1:
                r = results(messages[-1])[0]
                if r["is_error"]:
                    return self.fail(f"spawning the script failed: {r['content']}")
                # Traffic, so the log has something to compact.
                ev = turn([("tool_use", "t9", "send_message",
                            {"to": ["proc-2"], "message": "x" * 400})], "tool_use")
            elif n == 2:
                ev = turn([("text", "phase one done\n")], "end_turn")
            else:
                ev = turn([("text", "idle\n")], "end_turn")
        else:
            # Phase two: the harness has restarted from the journal.
            step = SEEN["n"]; SEEN["n"] += 1
            if step == 0:
                # The script boots on its own thread, so wait for it rather
                # than racing it.
                import time
                deadline = time.time() + 20
                boots = []
                while time.time() < deadline:
                    boots = open(MARK).read().split() if os.path.exists(MARK) else []
                    if len(boots) >= 2:
                        break
                    time.sleep(0.2)
                if len(boots) < 2:
                    return self.fail(f"the script did not boot again after restart: {boots}")
                if boots[0] != "fresh" or boots[1] != "resumed":
                    return self.fail(f"a restarted script should know it was resumed: {boots}")
                # And it must still be a working actor, not just a booted one.
                ev = turn([("tool_use", "t2", "call_process",
                            {"process_id": "proc-2", "message": "ping", "timeout_seconds": 15})],
                          "tool_use")
            elif step == 1:
                r = [x for x in results(messages[-1]) if x.get("tool_use_id") == "t2"]
                r = r[0] if r else {"is_error": True, "content": "no result for the call"}
                if r["is_error"] or "alive" not in r["content"]:
                    return self.fail(f"a restored script must still answer mail: {r}")
                ev = turn([("text", "phase two done\n")], "end_turn")
            else:
                ev = turn([("text", "idle\n")], "end_turn")

        self.respond(ev)

    def respond(self, ev):
        p = sse(ev)
        self.send_response(200); self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(p))); self.end_headers(); self.wfile.write(p)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("BITTY_RESTART_PORT", "8759"))), H).serve_forever()
