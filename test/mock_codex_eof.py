"""Codex stream retry: a response that dies mid-chunk must be retried, not
forwarded to the process as a failed turn.

Raw sockets on purpose: the failure being simulated is a truncated chunked
body, which an http.server handler cannot produce. First request gets headers
plus half a chunk and an abrupt close; the retry gets a complete response.
"""
import os
import socket
import threading

PORT = int(os.environ.get("PORT", "8770"))
SEEN = {"requests": 0}

GOOD_EVENTS = (
    'data: {"type":"response.output_text.delta","delta":"CODEX_RETRY_OK\\n"}\n\n'
    'data: {"type":"response.completed","response":{"id":"resp_1",'
    '"usage":{"input_tokens":10,"output_tokens":5,'
    '"input_tokens_details":{"cached_tokens":0}}}}\n\n'
    "data: [DONE]\n\n"
)


def drain_request(conn):
    conn.settimeout(5)
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = conn.recv(65536)
        if not chunk:
            return
        data += chunk
    head = data.split(b"\r\n\r\n", 1)[0].decode(errors="replace").lower()
    length = 0
    for line in head.split("\r\n"):
        if line.startswith("content-length:"):
            length = int(line.split(":", 1)[1].strip())
    body = data.split(b"\r\n\r\n", 1)[1]
    while len(body) < length:
        chunk = conn.recv(65536)
        if not chunk:
            return
        body += chunk


def serve(conn):
    with conn:
        drain_request(conn)
        SEEN["requests"] += 1
        if SEEN["requests"] == 1:
            # Headers, then half a chunk, then EOF mid-body.
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"content-type: text/event-stream\r\n"
                b"transfer-encoding: chunked\r\n\r\n"
                b"1f\r\ndata: {\"type\":\"response.out"
            )
            return  # close without finishing the chunk
        payload = GOOD_EVENTS.encode()
        conn.sendall(
            b"HTTP/1.1 200 OK\r\n"
            b"content-type: text/event-stream\r\n"
            b"content-length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload
        )


def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PORT))
    srv.listen(8)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=serve, args=(conn,), daemon=True).start()


main()
