import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
REPO=os.environ["BITTY_TEST_REPO"]
# Idiomatic Deno: the API a model actually reaches for.
GOOD = """
bitty.onMail(async (mail, api): Promise<string> => {
  await Deno.mkdir(%s + "/made-by-deno-api");
  const text: string = await Deno.readTextFile(%s + "/src/main.rs");
  return `deno-api-ok:${text.trim().length}`;
});
""" % (json.dumps(REPO), json.dumps(REPO))
BAD = """
bitty.onMail((mail): string => {
  const n: number = "not a number";
  return n;
});
"""
def sse(e): return "".join(f"event: {x['type']}\ndata: {json.dumps(x)}\n\n" for x in e).encode()
def turn(blocks, stop):
    ev=[{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"m","usage":{"input_tokens":1}}}]
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
        body=json.loads(self.rfile.read(int(self.headers["content-length"])))
        messages=body["messages"]; n=len([m for m in messages if m["role"]=="assistant"])
        if n==0:   # a script with a type error must be refused at spawn
            ev=turn([("tool_use","t1","spawn_process",{"name":"broken","instructions":"x","script":BAD})],"tool_use")
        elif n==1:
            r=res(messages[-1])[0]
            if not r["is_error"]:
                return self.fail("a script with a type error should not spawn")
            if "TypeScript" not in r["content"]:
                return self.fail(f"error should name the typecheck: {r['content'][:200]!r}")
            ev=turn([("tool_use","t2","spawn_process",{"name":"bad-model","instructions":"x","model":"gpt-4o"})],"tool_use")
        elif n==2:
            r=res(messages[-1])[0]
            if not r["is_error"] or "not a model tier" not in r["content"]:
                return self.fail(f"a vendor model id should be refused: {r}")
            ev=turn([("tool_use","t3","spawn_process",{"name":"good","instructions":"x","script":GOOD})],"tool_use")
        elif n==3:
            r=res(messages[-1])[0]
            if r["is_error"]:
                return self.fail(f"valid script should spawn: {r['content'][:300]}")
            ev=turn([("tool_use","t4","call_process",{"process_id":"proc-2","message":"go"})],"tool_use")
        elif n==4:
            r=res(messages[-1])[0]
            if r["is_error"] or "deno-api-ok" not in r["content"]:
                return self.fail(f"standard Deno API should work: {r}")
            ev=turn([("text","Done.\n")],"end_turn")
        else:
            ev=turn([("text","Idle.\n")],"end_turn")
        p=sse(ev); self.send_response(200); self.send_header("content-type","text/event-stream")
        self.send_header("content-length",str(len(p))); self.end_headers(); self.wfile.write(p)
ThreadingHTTPServer(("127.0.0.1",8746),H).serve_forever()
