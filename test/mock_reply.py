import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
def txt(m):
    c=m.get("content",[])
    return c if isinstance(c,str) else " ".join(b.get("text","") for b in c if isinstance(b,dict) and b.get("type")=="text")
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
        if "process proc-2" in system:
            incoming=txt(messages[-1])
            if 'reply_to="call-' in incoming:
                # Deliberately a plain send, with no in_reply_to: this is what a
                # real worker did, and it used to strand the caller until timeout.
                ev=turn([("tool_use","r1","send_message",{"to":"proc-1","message":"ENV REPORT: python 3.12"})],"tool_use")
            elif n>=1 and res(messages[-1]) and "Answered the call" in res(messages[-1])[0]["content"]:
                ev=turn([("text","Done.\n")],"end_turn")
            else:
                ev=turn([("text","Working.\n")],"end_turn")
        else:
            if n==0:
                ev=turn([("tool_use","t1","spawn_process",{"name":"recon","instructions":"scout"})],"tool_use")
            elif n==1:
                ev=turn([("tool_use","t2","call_process",{"process_id":"proc-2","message":"Status?","timeout_seconds":15})],"tool_use")
            elif n==2:
                r=res(messages[-1])[0]
                if r["is_error"]:
                    return self.fail(f"agent-to-agent reply failed: {r['content']!r}")
                if "ENV REPORT" not in r["content"]:
                    return self.fail(f"caller did not receive the reply: {r['content']!r}")
                ev=turn([("text","GOT_REPLY\n")],"end_turn")
            else:
                ev=turn([("text","Idle.\n")],"end_turn")
        p=sse(ev); self.send_response(200); self.send_header("content-type","text/event-stream")
        self.send_header("content-length",str(len(p))); self.end_headers(); self.wfile.write(p)
ThreadingHTTPServer(("127.0.0.1",8749),H).serve_forever()
