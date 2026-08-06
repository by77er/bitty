"""sh() and read() in the run_script session.

Covers the two prelude bindings end to end through a real harness: sh() is
already defined (nobody has to write it), a non-zero exit comes back as a
value instead of an error, stdout and stderr arrive merged in write order, the
working directory defaults to the read grant without the script naming it, a
timeout kills the child and keeps what it printed, and read() numbers lines
1-indexed with a negative start meaning a tail.

In the same scenario it re-asserts the api.exec / Deno.Command contracts
(text vs Uint8Array, non-ASCII round trip), because sh() lands beside them in
the same op path and a change there must not break them silently.
"""
import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
REPO = os.environ["BITTY_TEST_REPO"]
# The grant root is canonicalized by the harness (/var -> /private/var on
# macOS), so the last path component is what a pwd can be matched against.
LEAF = os.path.basename(REPO.rstrip("/"))
# See the n == 0 branch: assert that run_script's own description and the
# system prompt name sh() and read(). Off until spec §5 (agent.rs) lands.
DISCOVERABLE = os.environ.get("BITTY_TEST_SH_DISCOVERABLE") == "1"

# 1. sh() exists already, merges the streams in order, runs in the granted
#    directory, and reports a non-zero exit as data.
S_BASICS = """
const r = await sh("echo hello; echo oops 1>&2; pwd");
const bad = await sh("exit 7");
return [
  "sh=" + typeof sh, "read=" + typeof read,
  "code=" + r.code, "timedOut=" + r.timedOut, "truncated=" + r.truncated,
  "hello=" + r.out.includes("hello"), "merged=" + r.out.includes("oops"),
  "order=" + (r.out.indexOf("hello") < r.out.indexOf("oops")),
  "cwd=" + r.out.includes(%s),
  "bad=" + bad.code,
].join(" ");
""" % json.dumps(LEAF)

# 2. An overrunning command is killed and still hands back what it printed,
#    and the session it ran in is usable straight afterwards.
S_TIMEOUT = """
const t = await sh("echo early; sleep 5", { timeout: 1 });
const after = await sh("echo later");
return [
  "timedOut=" + t.timedOut, "code=" + t.code,
  "early=" + t.out.includes("early"),
  "after=" + after.out.trim(), "afterCode=" + after.code,
].join(" ");
"""

# 3. read(): 1-indexed, inclusive, clamped, negative start counts back.
S_READ = """
const P = %s + "/lines.txt";
bitty.fs.write(P, "alpha\\nbravo\\ncharlie\\ndelta\\necho\\n");
return [
  "span=" + (await read(P, 2, 3) === "2| bravo\\n3| charlie"),
  "tail=" + (await read(P, -2) === "4| delta\\n5| echo"),
  "past=" + (await read(P, 99) === ""),
  "all=" + (await read(P)).split("\\n").length,
].join(" ");
""" % json.dumps(REPO)

# 4. The two older exec paths, unchanged: text out of bitty.exec, bytes out of
#    Deno.Command. Codepoints rather than escapes, so a mojibake regression
#    cannot mangle its own expectation.
S_EXEC = """
const CWD = %s;
const CODES = [104, 233, 108, 108, 111, 32, 10003, 32, 8594];
const EXPECT = CODES.map((c) => String.fromCodePoint(c)).join("");
const PY = "import sys; sys.stdout.buffer.write(''.join(chr(c) for c in [" +
  CODES.join(",") + "]).encode('utf-8')); sys.stderr.write('e')";
const t = bitty.exec("python3", ["-c", PY], CWD);
const out = new Deno.Command("python3", { args: ["-c", PY], cwd: CWD }).outputSync();
return [
  "exec.stdout=" + typeof t.stdout,
  "exec.text=" + (t.stdout === EXPECT),
  "exec.errtext=" + (t.stderr === "e"),
  "cmd.stdout=" + out.stdout.constructor.name,
  "cmd.len=" + out.stdout.length,
  "cmd.text=" + (new TextDecoder().decode(out.stdout) === EXPECT),
  "cmd.code=" + out.code,
].join(" ");
""" % json.dumps(REPO)


def sse(e):
    return "".join(f"event: {x['type']}\ndata: {json.dumps(x)}\n\n" for x in e).encode()


def turn(blocks, stop):
    ev = [{"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant",
                                                "content": [], "model": "claude-opus-5",
                                                "usage": {"input_tokens": 1}}}]
    for i, b in enumerate(blocks):
        if b[0] == "text":
            ev += [{"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}},
                   {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b[1]}}]
        else:
            ev += [{"type": "content_block_start", "index": i,
                    "content_block": {"type": "tool_use", "id": b[1], "name": b[2], "input": {}}},
                   {"type": "content_block_delta", "index": i,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(b[3])}}]
        ev.append({"type": "content_block_stop", "index": i})
    ev += [{"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 1}},
           {"type": "message_stop"}]
    return ev


def res(m):
    return [b for b in m.get("content", []) if isinstance(b, dict) and b.get("type") == "tool_result"]


def flat(x):
    return x if isinstance(x, str) else " ".join(b.get("text", "") for b in x)


def script(src):
    return ("tool_use", "t", "run_script", {"script": src})


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def fail(self, m):
        b = json.dumps({"type": "error", "error": {"type": "invalid_request_error",
                                                   "message": "ASSERTION: " + m}}).encode()
        self.send_response(400)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(b)

    def want(self, r, tokens, what):
        """Every token must appear in the result, and the result must not be an error."""
        if r is None or r["is_error"]:
            return self.fail(f"{what} failed outright: {r}")
        got = flat(r["content"])
        missing = [t for t in tokens if t not in got]
        if missing:
            return self.fail(f"{what}: missing {missing} in {got!r}")
        return None

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["content-length"])))
        system, messages = flat(b["system"]), b["messages"]
        n = len([m for m in messages if m["role"] == "assistant"])
        r = res(messages[-1])[0] if messages and res(messages[-1]) else None
        if n == 0:
            names = [t["name"] for t in b["tools"]]
            if "run_script" not in names:
                return self.fail(f"root should be offered run_script: {names}")
            # The bindings are worth nothing if the model is never told they
            # exist — that is exactly how the run_build tool died. The prompt
            # side of the spec lives in agent.rs and is not applied yet, so
            # this is one env var away rather than asserted by default.
            if DISCOVERABLE:
                desc = next(t["description"] for t in b["tools"] if t["name"] == "run_script")
                for token in ["sh(", "read("]:
                    if token not in desc:
                        return self.fail(f"run_script's description should name {token}: {desc[-400:]!r}")
                if "sh(" not in system:
                    return self.fail(f"the system prompt should name sh(): {system[-400:]!r}")
            return self.respond(turn([script(S_BASICS)], "tool_use"))
        if n == 1:
            bad = self.want(r, ["sh=function", "read=function", "code=0", "timedOut=false",
                                "truncated=false", "hello=true", "merged=true", "order=true",
                                "cwd=true", "bad=7"], "sh basics")
            if bad is not None:
                return bad
            return self.respond(turn([script(S_TIMEOUT)], "tool_use"))
        if n == 2:
            bad = self.want(r, ["timedOut=true", "code=null", "early=true",
                                "after=later", "afterCode=0"], "sh timeout")
            if bad is not None:
                return bad
            return self.respond(turn([script(S_READ)], "tool_use"))
        if n == 3:
            bad = self.want(r, ["span=true", "tail=true", "past=true", "all=5"], "read")
            if bad is not None:
                return bad
            return self.respond(turn([script(S_EXEC)], "tool_use"))
        if n == 4:
            bad = self.want(r, ["exec.stdout=string", "exec.text=true", "exec.errtext=true",
                                "cmd.stdout=Uint8Array", "cmd.len=14", "cmd.text=true",
                                "cmd.code=0"], "the older exec contracts")
            if bad is not None:
                return bad
            return self.respond(turn([("text", "SH_OK\n")], "end_turn"))
        return self.respond(turn([("text", "Idle.\n")], "end_turn"))

    def respond(self, ev):
        p = sse(ev)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(p)))
        self.end_headers()
        self.wfile.write(p)


ThreadingHTTPServer(("127.0.0.1", 8747), H).serve_forever()
