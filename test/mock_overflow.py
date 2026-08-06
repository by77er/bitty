"""Overflow recovery: a prompt refused for length compacts and retries.

A 'prompt is too long' 400 is not a transient failure — retrying it verbatim
can never succeed. The harness must respond by summarising the conversation
and retrying once, and must not treat a second overflow as another invitation
to compact (one recovery per incident).

Sequence: a couple of growing turns, then an overflow 400, then the harness's
compaction request, then the retried turn — which must carry the summary and
be answered normally.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BALLAST = "b" * 20000


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


SEEN = {"overflows": 0, "compactions": 0, "post": 0}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def fail(self, m):
        b = json.dumps({"type": "error", "error": {"type": "invalid_request_error",
                                                   "message": "ASSERTION: " + m}}).encode()
        self.send_response(400); self.send_header("content-type", "application/json")
        self.end_headers(); self.wfile.write(b)

    def overflow(self):
        # A real-looking refusal, deliberately not an ASSERTION.
        b = json.dumps({"type": "error", "error": {"type": "invalid_request_error",
                                                   "message": "prompt is too long: 1053000 tokens > 1000000 maximum"}}).encode()
        self.send_response(400); self.send_header("content-type", "application/json")
        self.end_headers(); self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        messages = body["messages"]
        text = json.dumps(messages)
        n = len([m for m in messages if m["role"] == "assistant"])

        if "context checkpoint compaction" in text:
            if SEEN["overflows"] != 1:
                return self.fail("compaction should follow the first overflow")
            SEEN["compactions"] += 1
            return self.respond(turn([("text", "HANDOFF: ballast work done; verify next.\n")], "end_turn"))

        if SEEN["compactions"] > 0:
            SEEN["post"] += 1
            if SEEN["post"] == 1 and "compacted_context" not in text:
                return self.fail("the retried turn must carry the compaction summary")
            return self.respond(turn([("text", "recovered after overflow\n")], "end_turn"))

        # Two growing tool turns, then the overflow.
        if n < 2:
            return self.respond(turn(
                [("text", BALLAST + "\n"), ("tool_use", f"t{n}", "list_processes", {})], "tool_use"))
        if SEEN["overflows"] == 0:
            SEEN["overflows"] += 1
            return self.overflow()
        # A second verbatim retry of the oversized prompt means the recovery
        # path did not compact.
        return self.fail("the overflowed prompt was retried without compacting")

    def respond(self, ev):
        p = sse(ev)
        self.send_response(200); self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(p))); self.end_headers(); self.wfile.write(p)


ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("PORT", "8761"))), H).serve_forever()
