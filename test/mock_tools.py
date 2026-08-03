import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
def sse(e): return "".join(f"event: {x['type']}\ndata: {json.dumps(x)}\n\n" for x in e).encode()
def flat(x): return x if isinstance(x,str) else " ".join(b.get("text","") for b in x)
class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def fail(self,m):
        b=json.dumps({"type":"error","error":{"type":"invalid_request_error","message":"ASSERTION: "+m}}).encode()
        self.send_response(400); self.send_header("content-type","application/json"); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        b=json.loads(self.rfile.read(int(self.headers["content-length"])))
        system=flat(b["system"]); names=[t["name"] for t in b["tools"]]
        n=len([m for m in b["messages"] if m["role"]=="assistant"])
        if "process proc-2" in system:
            if "spawn_process" in names:
                return self.fail(f"a process without the spawn capability must not be offered spawn_process: {names}")
            if "stop_process" not in names:
                return self.fail(f"it may stop itself, so stop_process should remain: {names}")
            if names[0]!="send_message" or names[1]!="list_processes":
                return self.fail(f"always-tools must lead so the prefix stays shared: {names}")
            if not b["tools"][1].get("cache_control"):
                return self.fail("the breakpoint must sit on the last always-tool")
            ev=[{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"claude-opus-5","usage":{"input_tokens":1}}},
                {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}},
                {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"TOOLS_OK\n"}},
                {"type":"content_block_stop","index":0},
                {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}},{"type":"message_stop"}]
        elif n==0:
            if "spawn_process" not in names:
                return self.fail("root holds spawn and should be offered it")
            blocks={"name":"leaf","instructions":"x","can_spawn":False}
            ev=[{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"claude-opus-5","usage":{"input_tokens":1}}},
                {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"spawn_process","input":{}}},
                {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":json.dumps(blocks)}},
                {"type":"content_block_stop","index":0},
                {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":1}},{"type":"message_stop"}]
        else:
            ev=[{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"claude-opus-5","usage":{"input_tokens":1}}},
                {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}},
                {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"done\n"}},
                {"type":"content_block_stop","index":0},
                {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}},{"type":"message_stop"}]
        p=sse(ev); self.send_response(200); self.send_header("content-type","text/event-stream")
        self.send_header("content-length",str(len(p))); self.end_headers(); self.wfile.write(p)
ThreadingHTTPServer(("127.0.0.1",8750),H).serve_forever()
