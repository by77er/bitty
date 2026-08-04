"""Local compaction: a conversation replaced by a summary of itself.

Anthropic compacts server-side. Codex refuses stored responses and has no
equivalent, so past a threshold the harness has to summarise its own
conversation or the turn is simply refused for length.

This drives root past the threshold, then checks that the compaction turn is
asked for with no tools, that what comes back replaces the conversation, that
the opening briefing survives, and that the process keeps working afterward.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Big enough that two turns of it cross the threshold the suite sets.
BALLAST = "x" * 60000


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


SEEN = {"compactions": 0, "post": 0}


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
        messages = body["messages"]
        text = json.dumps(messages)
        n = len([m for m in messages if m["role"] == "assistant"])

        # The compaction turn is the one asking for a handoff summary.
        if "context checkpoint compaction" in text:
            if body.get("tools"):
                return self.fail("a compaction turn must not offer tools")
            SEEN["compactions"] += 1
            return self.respond(turn([("text", "HANDOFF: built the thing; next step is to verify it.\n")], "end_turn"))

        # After compacting, the conversation must be short again, must still
        # carry the briefing, and must carry the summary.
        if SEEN["compactions"] > 0:
            SEEN["post"] += 1
            if SEEN["post"] == 1:
                if len(text) > 200000:
                    return self.fail(f"the conversation should be small after compaction: {len(text)} chars")
                if "compacted_context" not in text or "HANDOFF" not in text:
                    return self.fail("the summary should replace the turns it summarised")
                if "the original briefing" not in text:
                    return self.fail("the opening briefing must survive compaction")
                # The turns just before compaction have to survive too: a
                # summary alone is lossy about what was just finished, which is
                # how a process ends up redoing completed work.
                if "MOST-RECENT-TURN" not in text:
                    return self.fail("recent turns must survive compaction verbatim")
                # And the process must still be able to work.
                return self.respond(turn([("tool_use", "t9", "list_processes", {})], "tool_use"))
            return self.respond(turn([("text", "compacted and still working\n")], "end_turn"))

        # Before the threshold: emit ballast to grow the conversation.
        if n < 3:
            return self.respond(turn([("text", BALLAST + "\n")], "end_turn")) if n else \
                   self.respond(turn([("tool_use", f"t{n}", "list_processes", {})], "tool_use"))
        marker = " MOST-RECENT-TURN" if n >= 3 else ""
        return self.respond(turn([("text", BALLAST + marker + "\n")], "end_turn"))

    def respond(self, ev):
        p = sse(ev)
        self.send_response(200); self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(p))); self.end_headers(); self.wfile.write(p)


ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("PORT", "8760"))), H).serve_forever()
