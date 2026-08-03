"""Mock Messages API verifying bitty's server-side compaction wiring.

Turn 1: assert the request carries the compact beta header + context_management,
        then emit a `compaction` block whose summary arrives via deltas of an
        event type the harness has never seen (compaction_delta).
Turn 2: assert the compaction block was echoed back verbatim, summary intact.
        Then reject the request the way a server without the beta would.
Turn 3: assert the harness dropped compaction and retried, and that usage
        accounting still works. Finish.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SUMMARY = "Earlier: the user asked for a haiku; assistant explored three drafts."
STATE = {"rejected": False}


def sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def turn(blocks, stop_reason, input_tokens):
    events = [{"type": "message_start", "message": {
        "id": "msg_mock", "type": "message", "role": "assistant", "content": [], "model": "claude-opus-5",
        "usage": {"input_tokens": input_tokens, "cache_creation_input_tokens": 1000,
                  "cache_read_input_tokens": 2000, "output_tokens": 5}}}]
    for i, b in enumerate(blocks):
        kind = b[0]
        if kind == "text":
            events.append({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}})
            events.append({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b[1]}})
        elif kind == "compaction":
            # Streams as an unfamiliar delta type — the harness must accumulate
            # it structurally rather than by a hardcoded name.
            events.append({"type": "content_block_start", "index": i, "content_block": {"type": "compaction", "content": ""}})
            mid = len(b[1]) // 2
            events.append({"type": "content_block_delta", "index": i, "delta": {"type": "compaction_delta", "content": b[1][:mid]}})
            events.append({"type": "content_block_delta", "index": i, "delta": {"type": "compaction_delta", "content": b[1][mid:]}})
        events.append({"type": "content_block_stop", "index": i})
    events.append({"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 5}})
    events.append({"type": "message_stop"})
    return events



def flatten_system(system):
    """system is now a list of content blocks (general preamble first, then the
    process-specific block) so the shared prefix can be cached."""
    if isinstance(system, str):
        return system
    return " ".join(b.get("text", "") for b in system)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def fail(self, msg, code=400):
        body = json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": msg}}).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        betas = self.headers.get("anthropic-beta", "")
        messages = body["messages"]
        n = len([m for m in messages if m["role"] == "assistant"])

        if n == 0:
            if "compact-2026-01-12" not in betas:
                return self.fail("ASSERTION: compact beta header missing, got " + betas)
            if body.get("context_management") != {"edits": [{"type": "compact_20260112"}]}:
                return self.fail(f"ASSERTION: bad context_management: {body.get('context_management')}")
            events = turn([("text", "Working.\n"), ("compaction", SUMMARY)], "end_turn", 150_000)

        elif not STATE["rejected"]:
            blocks = [b for m in messages if m["role"] == "assistant" for b in m["content"]]
            comp = [b for b in blocks if b.get("type") == "compaction"]
            if not comp:
                return self.fail(f"ASSERTION: compaction block was not echoed back: {blocks}")
            if comp[0].get("content") != SUMMARY:
                return self.fail(f"ASSERTION: compaction summary lost/garbled: {comp[0]!r}")
            # Now simulate a server that doesn't support the beta at all.
            STATE["rejected"] = True
            return self.fail("context_management is not supported for this model")

        else:
            if "compact-2026-01-12" in betas:
                return self.fail("ASSERTION: harness kept sending compaction after rejection")
            if "context_management" in body:
                return self.fail("ASSERTION: context_management still present after rejection")
            if "server-side-fallback-2026-07-01" not in betas:
                return self.fail("ASSERTION: unrelated beta was dropped along with compaction")
            events = turn([("text", "Recovered without compaction.\n")], "end_turn", 300_000)

        payload = sse(events)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8736), Handler).serve_forever()
