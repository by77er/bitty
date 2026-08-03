"""Inline TypeScript: runs with the caller's own capabilities, returns a value,
is typechecked, and is hidden from a process that cannot spawn."""
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
            if "run_script" in names:
                return self.fail(f"a process without spawn must not be offered run_script: {names}")
            return self.respond(turn([("text","leaf ok\n")],"end_turn"))
        if n==0:
            if "run_script" not in names:
                return self.fail(f"root holds spawn and should be offered run_script: {names}")
            return self.respond(turn([("tool_use","t1","run_script",
                {"script": "let n = 0; for await (const _ of Deno.readDir(%s)) n++; return `count=${n}`;" % json.dumps(REPO+"/src")})],"tool_use"))
        if n==1:
            r=res(messages[-1])[0]
            if r["is_error"] or "count=1" not in r["content"]:
                return self.fail(f"inline script should read the granted dir: {r}")
            return self.respond(turn([("tool_use","t2","run_script",{"script":"const n: number = \"nope\"; return n;"})],"tool_use"))
        if n==2:
            r=res(messages[-1])[0]
            if not r["is_error"] or "TypeScript" not in r["content"]:
                return self.fail(f"inline script should be typechecked: {r}")
            return self.respond(turn([("tool_use","t3","run_script",{"script":"return Deno.readTextFile(\"/etc/shadow\");"})],"tool_use"))
        if n==3:
            r=res(messages[-1])[0]
            if not r["is_error"]:
                return self.fail("inline script escaped the caller's filesystem grant")
            return self.respond(turn([("tool_use","t4","spawn_process",{"name":"leaf","instructions":"x","can_spawn":False})],"tool_use"))
        return self.respond(turn([("text","INLINE_OK\n")],"end_turn"))
    def respond(self, ev):
        p=sse(ev); self.send_response(200); self.send_header("content-type","text/event-stream")
        self.send_header("content-length",str(len(p))); self.end_headers(); self.wfile.write(p)
ThreadingHTTPServer(("127.0.0.1",8751),H).serve_forever()
