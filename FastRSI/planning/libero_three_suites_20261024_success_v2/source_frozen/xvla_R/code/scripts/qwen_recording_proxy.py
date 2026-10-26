"""Loopback-only transparent model-call recorder, used equally by all arms."""
import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error, request
from cost_trace import span, emit

OUT=Path(os.environ.get('FASTRSI_MODEL_CALLS','/project/runs/pilot_20261001/model_calls'))
OUT.mkdir(parents=True,exist_ok=True)
opener=request.build_opener(request.ProxyHandler({}))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path=='/healthz' else 404)
        self.end_headers();self.wfile.write(b'{}')

    def do_POST(self):
        if self.path!='/v1/chat/completions':
            self.send_error(404);return
        call_id=uuid.uuid4().hex
        data=self.rfile.read(int(self.headers.get('Content-Length','0')))
        (OUT/(call_id+'.request.json')).write_bytes(data)
        status=200
        with span('qwen_call',call_id=call_id):
            try:
                req=request.Request('http://127.0.0.1:49510'+self.path,data=data,
                                    headers={'Content-Type':'application/json'})
                with opener.open(req,timeout=600) as response:
                    status=response.status;body=response.read()
            except error.HTTPError as exc:
                status=exc.code;body=exc.read()
            except Exception as exc:
                status=504;body=json.dumps({'error':str(exc)}).encode()
        (OUT/(call_id+'.response.json')).write_bytes(body)
        try:usage=json.loads(body).get('usage')
        except (ValueError,AttributeError):usage=None
        emit({'event':'model_usage','call_id':call_id,'status':status,'usage':usage})
        self.send_response(status);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)


ThreadingHTTPServer(('127.0.0.1',49511),Handler).serve_forever()
