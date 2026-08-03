"""Mock Anthropic Messages API for testing the bitty harness offline.

Scripts a deterministic scenario:
  proc-1 (root): spawn worker -> idle -> (woken by worker's mail) send to user -> done
  proc-2 (worker): send result to proc-1 -> done

Also asserts that thinking blocks (with signature) are echoed back unchanged;
returns HTTP 400 if not, which surfaces as a visible error in the harness.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = open("/tmp/claude-1000/-home-bit-Code-Bitty/29b2838b-63ef-4bac-9206-4b943883dc56/scratchpad/mock_requests.jsonl", "w")


def sse(events):
    out = []
    for ev in events:
        out.append(f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n")
    return "".join(out).encode()


def turn(blocks, stop_reason):
    """Build the SSE event list for one response turn."""
    events = [{"type": "message_start", "message": {"id": "msg_mock", "type": "message", "role": "assistant", "content": [], "model": "claude-opus-5"}}]
    for i, b in enumerate(blocks):
        kind = b[0]
        if kind == "thinking":
            events.append({"type": "content_block_start", "index": i, "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
            events.append({"type": "content_block_delta", "index": i, "delta": {"type": "thinking_delta", "thinking": b[1]}})
            events.append({"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta", "signature": b[2]}})
        elif kind == "text":
            events.append({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}})
            # split into two deltas to test accumulation
            mid = len(b[1]) // 2
            events.append({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b[1][:mid]}})
            events.append({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b[1][mid:]}})
        elif kind == "tool_use":
            events.append({"type": "content_block_start", "index": i, "content_block": {"type": "tool_use", "id": b[1], "name": b[2], "input": {}}})
            payload = json.dumps(b[3])
            mid = len(payload) // 2
            events.append({"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": payload[:mid]}})
            events.append({"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": payload[mid:]}})
        events.append({"type": "content_block_stop", "index": i})
    events.append({"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 10}})
    events.append({"type": "message_stop"})
    return events



def flatten_system(system):
    """system is now a list of content blocks (general preamble first, then the
    process-specific block) so the shared prefix can be cached."""
    if isinstance(system, str):
        return system
    return " ".join(b.get("text", "") for b in system)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def fail(self, msg):
        body = json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": msg}}).encode()
        self.send_response(400)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        LOG.write(json.dumps(body) + "\n")
        LOG.flush()

        system = flatten_system(body.get("system", ""))
        messages = body["messages"]
        assistants = [m for m in messages if m["role"] == "assistant"]
        n = len(assistants)

        # Verify thinking blocks round-trip unchanged
        for m in assistants:
            for block in m["content"]:
                if block.get("type") == "thinking" and block.get("signature") != "sig-mock-1":
                    return self.fail("thinking signature was not echoed back unchanged")

        if "process proc-1" in system:
            if n == 0:
                blocks = [
                    ("thinking", "planning the delegation", "sig-mock-1"),
                    ("text", "Spawning a worker for the arithmetic.\n"),
                    ("tool_use", "toolu_1", "spawn_process", {"instructions": "Compute 2+2 and send the result to proc-1.", "name": "worker"}),
                ]
                events = turn(blocks, "tool_use")
            elif n == 1:
                events = turn([("text", "Standing by for the worker.\n")], "end_turn")
            elif n == 2:
                last_user = json.dumps(messages[-1])
                if "incoming_message" not in last_user or "proc-2" not in last_user:
                    return self.fail("expected worker mail in an <incoming_message> envelope")
                # Worker already stopped itself: this must be a non-error no-op.
                blocks = [("tool_use", "toolu_2", "stop_process", {"process_id": "proc-2"})]
                events = turn(blocks, "tool_use")
            elif n == 3:
                results = [b for b in messages[-1]["content"] if b.get("type") == "tool_result"]
                if not results or results[0]["is_error"] or "Already stopped" not in results[0]["content"]:
                    return self.fail(f"expected idempotent 'Already stopped' result, got {results}")
                # Mailing a stopped process must error.
                blocks = [("tool_use", "toolu_3", "send_message", {"process_id": "proc-2", "message": "ping after stop"})]
                events = turn(blocks, "tool_use")
            elif n == 4:
                results = [b for b in messages[-1]["content"] if b.get("type") == "tool_result"]
                if not results or not results[0]["is_error"] or "stopped" not in results[0]["content"]:
                    return self.fail(f"expected is_error result for mail to stopped process, got {results}")
                blocks = [("tool_use", "toolu_4", "send_message", {"process_id": "user", "message": "2 + 2 = 4 (computed by proc-2, now stopped)"})]
                events = turn(blocks, "tool_use")
            else:
                events = turn([("text", "All done.\n")], "end_turn")
        elif "process proc-2" in system:
            if n == 0:
                # Parallel tool batch ending in self-stop: send result, then stop self.
                blocks = [
                    ("text", "Computing 2+2.\n"),
                    ("tool_use", "toolu_w1", "send_message", {"process_id": "proc-1", "message": "The answer is 4."}),
                    ("tool_use", "toolu_w2", "stop_process", {"process_id": "proc-2"}),
                ]
                events = turn(blocks, "tool_use")
            else:
                return self.fail("worker stopped itself; it must never call the API again")
        else:
            return self.fail("unknown process in system prompt")

        payload = sse(events)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8734), Handler).serve_forever()
