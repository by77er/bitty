"""Long agent mail is previewed, paged through mailbox, and discardable."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LONG_BODY = "HEAD:" + ("x" * 9_000) + ":TAIL_SENTINEL"
ARTIFACT_ID = "mail-proc-2-1"


def sse(events):
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()


def turn(blocks, stop):
    events = [{
        "type": "message_start",
        "message": {
            "id": "m",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-opus-5",
            "usage": {"input_tokens": 1},
        },
    }]
    for index, block in enumerate(blocks):
        if block[0] == "text":
            events += [
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": block[1]},
                },
            ]
        else:
            events += [
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": block[1],
                        "name": block[2],
                        "input": {},
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block[3]),
                    },
                },
            ]
        events.append({"type": "content_block_stop", "index": index})
    events += [
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]
    return events


def text(message):
    content = message.get("content", [])
    if isinstance(content, str):
        return content
    return " ".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def results(message):
    return [
        block
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]


def flatten_system(system):
    if isinstance(system, str):
        return system
    return " ".join(block.get("text", "") for block in system)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def fail(self, message):
        payload = json.dumps({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "ASSERTION: " + message,
            },
        }).encode()
        self.send_response(400)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        system = flatten_system(body["system"])
        messages = body["messages"]
        assistant_turns = len([m for m in messages if m["role"] == "assistant"])
        all_text = " ".join(text(message) for message in messages)

        if "process proc-2" in system:
            if "mailbox" not in [tool["name"] for tool in body["tools"]]:
                return self.fail("every agent must be offered the mailbox tool")
            if assistant_turns == 0:
                events = turn([("text", "Waiting for data.\n")], "end_turn")
            elif assistant_turns == 1:
                if f'artifact_id="{ARTIFACT_ID}"' not in all_text:
                    return self.fail("long mail did not carry its artifact handle")
                if 'artifact_chars="9019"' not in all_text:
                    return self.fail("long mail did not carry its original character count")
                if "HEAD:" not in all_text or "TAIL_SENTINEL" in all_text:
                    return self.fail("the worker should receive a head preview, not the full body")
                events = turn(
                    [(
                        "tool_use",
                        "w1",
                        "mailbox",
                        {
                            "action": "read",
                            "id": ARTIFACT_ID,
                            "offset": len(LONG_BODY) - 64,
                            "limit": 128,
                        },
                    )],
                    "tool_use",
                )
            elif assistant_turns == 2:
                result = results(messages[-1])[0]
                if result["is_error"] or "TAIL_SENTINEL" not in result["content"]:
                    return self.fail(f"mailbox read did not return the requested tail: {result}")
                events = turn(
                    [(
                        "tool_use",
                        "w2",
                        "mailbox",
                        {"action": "discard", "id": ARTIFACT_ID},
                    )],
                    "tool_use",
                )
            elif assistant_turns == 3:
                result = results(messages[-1])[0]
                if result["is_error"] or "Discarded" not in result["content"]:
                    return self.fail(f"mailbox discard failed: {result}")
                events = turn(
                    [(
                        "tool_use",
                        "w3",
                        "send_message",
                        {"to": "proc-1", "message": "MAILBOX_WORKER_OK"},
                    )],
                    "tool_use",
                )
            else:
                events = turn([("text", "Done.\n")], "end_turn")
        else:
            if "MAILBOX_WORKER_OK" in all_text:
                events = turn([("text", "MAILBOX_OK\n")], "end_turn")
            elif assistant_turns == 0:
                events = turn(
                    [(
                        "tool_use",
                        "r1",
                        "spawn_process",
                        {"name": "reader", "instructions": "Wait for long data."},
                    )],
                    "tool_use",
                )
            elif assistant_turns == 1:
                events = turn(
                    [(
                        "tool_use",
                        "r2",
                        "send_message",
                        {"to": "proc-2", "message": LONG_BODY},
                    )],
                    "tool_use",
                )
            else:
                events = turn([("text", "Waiting for the reader.\n")], "end_turn")

        payload = sse(events)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


ThreadingHTTPServer(("127.0.0.1", 8793), Handler).serve_forever()
