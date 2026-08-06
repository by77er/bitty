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

# Big enough that a few turns of it cross the token threshold the suite sets
# (the harness estimates chars/4 when the mock reports tiny real usage), and
# bigger than the verbatim tail budget, so ballast turns cannot ride through
# compaction untouched.
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
                # The compacted block has to say what the session is holding,
                # or the model redefines (or re-fetches) it.
                if "session survived" not in text or "keep" not in text:
                    return self.fail("the compacted block must name surviving session state")
                # And the process must still be able to work.
                return self.respond(turn([("tool_use", "t9", "list_processes", {})], "tool_use"))
            return self.respond(turn([("text", "compacted and still working\n")], "end_turn"))

        # Park state in the session first, so the post-compaction block can
        # prove it names what survived.
        if n == 0:
            return self.respond(turn(
                [("tool_use", "s0", "run_script",
                  {"script": "g.keep = 'IMPORTANT-STATE'; return 'parked';"})], "tool_use"))

        # If the conversation has grown this far without a compaction, the
        # trigger is broken — say so rather than passing vacuously. (An
        # earlier version of this mock ended every ballast turn with
        # end_turn, so the process idled before ever crossing the threshold
        # and none of the assertions below could fire.)
        if n > 8:
            return self.fail("compaction never triggered while the conversation grew")

        # Before the threshold: keep the process in a tool loop while ballast
        # grows the conversation — compaction is checked between tool turns.
        # Later turns are smaller than the verbatim tail budget, so the marker
        # can prove the tail survives: three 60K ballast turns put the chars/4
        # estimate just under the 50K-token threshold the suite sets, and the
        # 8K marker turns walk it across a couple of turns later.
        body_text = BALLAST if n < 4 else "y" * 8000
        marker = " MOST-RECENT-TURN" if n >= 4 else ""
        return self.respond(turn(
            [("text", body_text + marker + "\n"),
             ("tool_use", f"t{n}", "list_processes", {})], "tool_use"))

    def respond(self, ev):
        p = sse(ev)
        self.send_response(200); self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(p))); self.end_headers(); self.wfile.write(p)


ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("PORT", "8760"))), H).serve_forever()
