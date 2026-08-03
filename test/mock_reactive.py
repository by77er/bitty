"""Reactive scripts: awaiting does not deafen a process.

A script process is mail-driven, so the danger is that anything it awaits
either blocks the mailbox or never gets polled. root spawns a sleeper that
awaits a long sleep, then mails it mid-sleep; the reply must come back, proving
the event loop and the mailbox are raced rather than one starving the other.

Then a socket: the script connects to a WebSocket, awaits a frame that only
arrives later, and answers with it — no polling anywhere.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Sleeps far longer than the test takes. If awaiting blocked the mailbox, the
# call below would time out instead of being answered.
SLEEPER = """
let woken = 0;
bitty.onMail(async (mail): Promise<string> => {
  if (mail.body === "nap") { await bitty.sleep(8000); woken++; return "slept"; }
  return "awake while napping, woken=" + woken;
});
"""

LISTENER = """
let socket: any = null;
bitty.onMail(async (mail): Promise<string> => {
  if (mail.body.startsWith("connect ")) {
    socket = await bitty.connect(mail.body.slice(8));
    return "connected";
  }
  const frame = await socket.recv();
  return "frame:" + frame;
});
"""


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
            return self.fail("only the root should be taking turns here")

        if n == 0:
            ev = turn([("tool_use", "t1", "spawn_process",
                        {"name": "sleeper", "instructions": "nap", "script": SLEEPER})], "tool_use")
        elif n == 1:
            r = results(messages[-1])[0]
            if r["is_error"]:
                return self.fail(f"spawning the sleeper failed: {r['content']}")
            # Start the long sleep without waiting for it.
            ev = turn([("tool_use", "t2", "send_message",
                        {"to": ["proc-2"], "message": "nap"})], "tool_use")
        elif n == 2:
            # Mid-sleep, call it. A blocked mailbox would time out here.
            ev = turn([("tool_use", "t3", "call_process",
                        {"process_id": "proc-2", "message": "status", "timeout_seconds": 15})],
                      "tool_use")
        elif n == 3:
            r = results(messages[-1])[0]
            if r["is_error"]:
                return self.fail(f"a sleeping script must still answer mail: {r['content']}")
            if "awake while napping" not in r["content"]:
                return self.fail(f"unexpected reply from the sleeper: {r['content']!r}")
            ev = turn([("tool_use", "t4", "spawn_process",
                        {"name": "listener", "instructions": "listen", "script": LISTENER})], "tool_use")
        elif n == 4:
            r = results(messages[-1])[0]
            if r["is_error"]:
                return self.fail(f"spawning the listener failed: {r['content']}")
            ev = turn([("tool_use", "t5", "call_process",
                        {"process_id": "proc-3", "message": "connect ws://127.0.0.1:8901/",
                         "timeout_seconds": 15})], "tool_use")
        elif n == 5:
            r = results(messages[-1])[0]
            if r["is_error"] or "connected" not in r["content"]:
                return self.fail(f"the socket should connect: {r}")
            # The server sends its frame a second from now; the script awaits it.
            ev = turn([("tool_use", "t6", "call_process",
                        {"process_id": "proc-3", "message": "await", "timeout_seconds": 20})],
                      "tool_use")
        elif n == 6:
            r = results(messages[-1])[0]
            if r["is_error"]:
                return self.fail(f"awaiting a frame failed: {r['content']}")
            if "frame:hello-socket" not in r["content"]:
                return self.fail(f"the awaited frame should come back: {r['content']!r}")
            ev = turn([("text", "Reactive.\n")], "end_turn")
        else:
            ev = turn([("text", "Idle.\n")], "end_turn")

        p = sse(ev)
        self.send_response(200); self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(p))); self.end_headers(); self.wfile.write(p)


def websocket_server():
    """A WebSocket server small enough to not need a dependency: handshake,
    wait a second, then push one text frame."""
    import base64
    import hashlib
    import socket
    import time

    GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 8901))
    srv.listen(5)
    while True:
        try:
            conn, _ = srv.accept()
            request = conn.recv(4096).decode("latin-1")
            key = ""
            for line in request.split("\r\n"):
                if line.lower().startswith("sec-websocket-key:"):
                    key = line.split(":", 1)[1].strip()
            accept = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
            conn.send((
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
            ).encode())
            # The delay is the point: the script must be awaiting, not polling.
            time.sleep(1.0)
            payload = b"hello-socket"
            conn.send(bytes([0x81, len(payload)]) + payload)
        except Exception:
            pass


if __name__ == "__main__":
    threading.Thread(target=websocket_server, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", 8758), H).serve_forever()
