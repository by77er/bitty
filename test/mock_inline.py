"""Inline TypeScript: runs in the caller's persistent session with the
caller's own capabilities.

Covers: annotations are stripped (transpiled, not merely ignored), the
precheck is syntax-only by design (a type mismatch runs; a syntax error is
caught before V8), filesystem grants bind the session, g.* persists across
run_script calls, oversized results are stored as g.results.rN handles
instead of landing in context, and run_script is offered independently of
spawn authority (it is this-process authority, not delegation)."""
import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
REPO=os.environ["BITTY_TEST_REPO"]
def sse(e): return "".join(f"event: {x['type']}\ndata: {json.dumps(x)}\n\n" for x in e).encode()
def turn(blocks, stop):
    ev=[{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"claude-opus-5","usage":{"input_tokens":1}}}]
    for i,b in enumerate(blocks):
        if b[0]=="text":
            ev+=[{"type":"content_block_start","index":i,"content_block":{"type":"text","text":""}},
                 {"type":"content_block_delta","index":i,"delta":{"type":"text_delta","text":b[1]}}]
        else:
            ev+=[{"type":"content_block_start","index":i,"content_block":{"type":"tool_use","id":b[1],"name":b[2],"input":{}}},
                 {"type":"content_block_delta","index":i,"delta":{"type":"input_json_delta","partial_json":json.dumps(b[3])}}]
        ev.append({"type":"content_block_stop","index":i})
    ev+=[{"type":"message_delta","delta":{"stop_reason":stop},"usage":{"output_tokens":1}},{"type":"message_stop"}]
    return ev
def res(m): return [b for b in m.get("content",[]) if isinstance(b,dict) and b.get("type")=="tool_result"]
def flat(x): return x if isinstance(x,str) else " ".join(b.get("text","") for b in x)
def script(src): return ("tool_use", "t", "run_script", {"script": src})
class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def fail(self,m):
        b=json.dumps({"type":"error","error":{"type":"invalid_request_error","message":"ASSERTION: "+m}}).encode()
        self.send_response(400); self.send_header("content-type","application/json"); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        b=json.loads(self.rfile.read(int(self.headers["content-length"])))
        system=flat(b["system"]); messages=b["messages"]; names=[t["name"] for t in b["tools"]]
        n=len([m for m in messages if m["role"]=="assistant"])
        if "process proc-2" in system:
            # run_script is this-process authority and rides no other grant;
            # the spawn tools do, and this leaf was denied spawn.
            if "run_script" not in names:
                return self.fail(f"run_script does not require spawn authority, but the leaf lost it: {names}")
            if "spawn_process" in names:
                return self.fail(f"a process without spawn must not be offered spawn_process: {names}")
            return self.respond(turn([("text","leaf ok\n")],"end_turn"))
        if n==0:
            if "run_script" not in names:
                return self.fail(f"root should be offered run_script: {names}")
            return self.respond(turn([script(
                # Annotated on purpose: inline source has to be transpiled, not
                # just parsed, or every annotation reaches V8 as a syntax error.
                "let c: number = 0; const dir: string = %s; for await (const _ of Deno.readDir(dir)) c++; return `count=${c}`;" % json.dumps(REPO+"/src"))],"tool_use"))
        r = res(messages[-1])[0] if res(messages[-1]) else None
        if n==1:
            if r["is_error"] or "count=1" not in r["content"]:
                return self.fail(f"inline script should read the granted dir: {r}")
            # A type mismatch is deliberately a runtime concern: the precheck
            # is syntax-only (no host toolchain), so this runs and returns.
            return self.respond(turn([script("const s: number = \"nope\"; return s;")],"tool_use"))
        if n==2:
            if r["is_error"] or r["content"] != "nope":
                return self.fail(f"annotations are stripped, not type-checked; this should run: {r}")
            return self.respond(turn([script("const = ;")],"tool_use"))
        if n==3:
            if not r["is_error"] or "TypeScript" not in r["content"]:
                return self.fail(f"a syntax error should be caught before V8: {r}")
            return self.respond(turn([script("return Deno.readTextFile(\"/etc/shadow\");")],"tool_use"))
        if n==4:
            if not r["is_error"]:
                return self.fail("inline script escaped the caller's filesystem grant")
            return self.respond(turn([script("g.x = 42; return \"stored\";")],"tool_use"))
        if n==5:
            if r["is_error"] or "stored" not in r["content"]:
                return self.fail(f"storing session state should succeed: {r}")
            return self.respond(turn([script("return g.x + 1;")],"tool_use"))
        if n==6:
            if r["is_error"] or "43" not in r["content"]:
                return self.fail(f"g.x must persist across run_script calls: {r}")
            return self.respond(turn([script("return \"y\".repeat(20000);")],"tool_use"))
        if n==7:
            if r["is_error"] or "g.results.r1" not in r["content"]:
                return self.fail(f"an oversized result should come back as a g.results handle: {r}")
            if len(r["content"]) > 5000:
                return self.fail(f"the handle must not carry the whole payload: {len(r['content'])} chars")
            return self.respond(turn([script("return g.results.r1.length;")],"tool_use"))
        if n==8:
            if r["is_error"] or "20000" not in r["content"]:
                return self.fail(f"the stored result should be sliceable later: {r}")
            return self.respond(turn([("tool_use","t9","spawn_process",{"name":"leaf","instructions":"x","can_spawn":False})],"tool_use"))
        return self.respond(turn([("text","INLINE_OK\n")],"end_turn"))
    def respond(self, ev):
        p=sse(ev); self.send_response(200); self.send_header("content-type","text/event-stream")
        self.send_header("content-length",str(len(p))); self.end_headers(); self.wfile.write(p)
ThreadingHTTPServer(("127.0.0.1",8751),H).serve_forever()
