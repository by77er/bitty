"""Verification gates: a failing gate is work for root, not an exit.

The suite runs with --gate "test -f $DIR/marker": the first quiesce fails the
gate, the bounded output lands in root's mailbox, root fixes the workspace
with run_script, and the rerun passes. Server-side assertions check the gate
mail's shape and that no further gate mail arrives once the marker exists.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GATE_DIR = os.environ.get("BITTY_TEST_GATE_DIR", "/tmp")


def sse(e):
    return "".join(f"event: {x['type']}\ndata: {json.dumps(x)}\n\n" for x in e).encode()


def turn(blocks, sr):
    ev = [{"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant",
           "content": [], "model": "claude-opus-5", "usage": {"input_tokens": 10}}}]
    for i, b in enumerate(blocks):
        if b[0] == "text":
            ev += [{"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}},
                   {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b[1]}}]
        else:
            ev += [{"type": "content_block_start", "index": i, "content_block": {"type": "tool_use", "id": b[1], "name": b[2], "input": {}}},
                   {"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": json.dumps(b[3])}}]
        ev.append({"type": "content_block_stop", "index": i})
    ev += [{"type": "message_delta", "delta": {"stop_reason": sr}, "usage": {"output_tokens": 5}},
           {"type": "message_stop"}]
    return ev


SEEN = {"gate_mail": 0}


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
        text = json.dumps(body["messages"])
        n = len([m for m in body["messages"] if m["role"] == "assistant"])

        if "Quality gate failed" in text and SEEN["gate_mail"] == 0:
            SEEN["gate_mail"] += 1
            if "exited with 1" not in text:
                return self.fail("the gate mail must carry the exit code")
            if "Output:" not in text:
                return self.fail("the gate mail must carry the bounded output")
            if os.path.exists(os.path.join(GATE_DIR, "marker")):
                return self.fail("the marker must not exist before root creates it")
            # Root fixes the workspace, exercising run_script + the write grant.
            script = f"await Deno.writeTextFile({json.dumps(os.path.join(GATE_DIR, 'marker'))}, 'ok'); return 'made'"
            return self.respond(turn(
                [("tool_use", "t1", "run_script", {"script": script})], "tool_use"))

        if "Quality gate failed" in text and "made" in text:
            # The tool result came back; finish the turn and go idle so the
            # gate reruns against the now-fixed workspace.
            return self.respond(turn([("text", "marker created; done\n")], "end_turn"))

        if SEEN["gate_mail"] > 1 or ("was not rerun" in text):
            return self.fail("the gate must pass once the marker exists")

        # First quiesce: claim the work is done without doing it.
        if n == 0:
            return self.respond(turn([("text", "did the work (not really)\n")], "end_turn"))
        return self.respond(turn([("text", "idle\n")], "end_turn"))

    def respond(self, ev):
        p = sse(ev)
        self.send_response(200); self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(p))); self.end_headers(); self.wfile.write(p)


ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("PORT", "8762"))), H).serve_forever()
