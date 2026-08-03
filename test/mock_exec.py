import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
REPO=os.environ["BITTY_TEST_REPO"]
SCRIPT = """
bitty.onMail((mail, api): string => {
  if (mail.body === "run") {
    const r = api.exec("python3", ["-c", "print(6*7)"], %s);
    return `code=${r.code} out=${r.stdout.trim()}`;
  }
  if (mail.body === "mkdir") { api.fs.mkdir(%s + "/newdir"); return "made"; }
  try { api.exec("rm", ["-rf", "/"], %s); return "RAN_FORBIDDEN"; }
  catch (e) { return `blocked:${e instanceof Error ? e.message : String(e)}`; }
});
""" % (json.dumps(REPO), json.dumps(REPO), json.dumps(REPO))
def sse(e): return "".join(f"event: {x['type']}\ndata: {json.dumps(x)}\n\n" for x in e).encode()
def turn(blocks, stop):
    ev=[{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"claude-opus-5","usage":{"input_tokens":10}}}]
    for i,b in enumerate(blocks):
        if b[0]=="text":
            ev+=[{"type":"content_block_start","index":i,"content_block":{"type":"text","text":""}},
                 {"type":"content_block_delta","index":i,"delta":{"type":"text_delta","text":b[1]}}]
        else:
            ev+=[{"type":"content_block_start","index":i,"content_block":{"type":"tool_use","id":b[1],"name":b[2],"input":{}}},
                 {"type":"content_block_delta","index":i,"delta":{"type":"input_json_delta","partial_json":json.dumps(b[3])}}]
        ev.append({"type":"content_block_stop","index":i})
    ev+=[{"type":"message_delta","delta":{"stop_reason":stop},"usage":{"output_tokens":5}},{"type":"message_stop"}]
    return ev
def results(m): return [b for b in m.get("content",[]) if isinstance(b,dict) and b.get("type")=="tool_result"]
def flat(x): return x if isinstance(x,str) else " ".join(b.get("text","") for b in x)
class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def fail(self,m):
        b=json.dumps({"type":"error","error":{"type":"invalid_request_error","message":"ASSERTION: "+m}}).encode()
        self.send_response(400); self.send_header("content-type","application/json"); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers["content-length"])))
        system,messages=flat(body["system"]),body["messages"]
        n=len([m for m in messages if m["role"]=="assistant"])
        if n==0:
            if "Running: only python3" not in system:
                return self.fail(f"root should hold a scoped run grant: {system[-300:]!r}")
            ev=turn([("tool_use","t1","spawn_process",{"name":"tools","instructions":"x","script":SCRIPT})],"tool_use")
        elif n==1:
            ev=turn([("tool_use","t2","call_process",{"process_id":"proc-2","message":"run"})],"tool_use")
        elif n==2:
            r=results(messages[-1])[0]
            if r["is_error"] or "out=42" not in r["content"]:
                return self.fail(f"python3 should have run: {r}")
            ev=turn([("tool_use","t3","call_process",{"process_id":"proc-2","message":"mkdir"})],"tool_use")
        elif n==3:
            r=results(messages[-1])[0]
            if r["is_error"]:
                return self.fail(f"mkdir should work with write access: {r}")
            ev=turn([("tool_use","t4","call_process",{"process_id":"proc-2","message":"forbidden"})],"tool_use")
        elif n==4:
            r=results(messages[-1])[0]
            if "RAN_FORBIDDEN" in r["content"]:
                return self.fail("a program outside the allowlist was executed")
            if "blocked:" not in r["content"]:
                return self.fail(f"expected a refusal: {r['content']!r}")
            ev=turn([("text","Done.\n")],"end_turn")
        else:
            ev=turn([("text","Idle.\n")],"end_turn")
        p=sse(ev); self.send_response(200); self.send_header("content-type","text/event-stream")
        self.send_header("content-length",str(len(p))); self.end_headers(); self.wfile.write(p)
ThreadingHTTPServer(("127.0.0.1",8744),H).serve_forever()
