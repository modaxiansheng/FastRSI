"""Twelve fixed real-model calls: identical inputs, balanced decoder order."""
import json
import os
import statistics
import time
from pathlib import Path
from urllib.request import Request,urlopen,build_opener,ProxyHandler

root=Path(os.environ['FASTRSI_STAGE_RUN']);project=Path('/project');rows=[];started=time.time()
opener=build_opener(ProxyHandler({}))
for n,row in enumerate(json.loads((root/'DECODE_REQUESTS.json').read_text())):
    body=json.loads((project/row['service_call']/'REQUEST.json').read_text())
    order=['reference','optimized','optimized','reference'] if n%2==0 else ['optimized','reference','reference','optimized']
    for variant in order:
        request={**body,'fastrsi_decoder':variant}
        tick=time.time()
        with opener.open(Request('http://127.0.0.1:49510/v1/chat/completions',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'}),timeout=240) as r:
            answer=json.load(r)
        call=root/'stream/service_calls'/answer['id']
        receipt=json.loads((call/'RECEIPT.json').read_text())
        rows.append({'label':row['label'],'variant':variant,'call':answer['id'],'seconds':time.time()-tick,
            'receipt':receipt,'token_ids':json.loads((call/'OUTPUT_TOKEN_IDS.json').read_text()),'message':answer['choices'][0]['message']})
        (root/'DECODER_PROGRESS.json').write_text(json.dumps({'rows':rows},indent=2))
summary=[]
for source in json.loads((root/'DECODE_REQUESTS.json').read_text()):
    group=[r for r in rows if r['label']==source['label']]
    same=all(r['token_ids']==group[0]['token_ids'] for r in group)
    means={v:statistics.mean(r['receipt']['generation_seconds'] for r in group if r['variant']==v) for v in ['reference','optimized']}
    summary.append({'label':source['label'],'all_output_tokens_equal':same,'generation_mean_seconds':means,
        'relative_generation_reduction':1-means['optimized']/means['reference']})
result={'status':'COMPLETED','summary':summary,'rows':rows,'seconds':time.time()-started,
    'scope':'Public decoder comparison on saved requests, not FastRSI efficacy or full-task speedup'}
(root/'DECODER_RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps({'summary':summary,'seconds':result['seconds']}),flush=True)
