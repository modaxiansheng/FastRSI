"""Replay saved token prefixes against both masks; no model generation."""
import hashlib
import json
import os
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
from fastrsi.constrained_tools import ToolJSONLogitsProcessor

root=Path(os.environ['FASTRSI_STAGE_RUN']);project=Path('/project')
torch.set_num_threads(2)
tokenizer=AutoTokenizer.from_pretrained('/model',trust_remote_code=False)
data=build_token_enforcer_tokenizer_data(tokenizer)
data.eos_token_id=json.loads(Path('/model/generation_config.json').read_text())['eos_token_id']
rows=[];started=time.time()
for row in json.loads((root/'DECODE_REQUESTS.json').read_text()):
    folder=project/row['service_call'];body=json.loads((folder/'REQUEST.json').read_text())
    ids=json.loads((folder/'OUTPUT_TOKEN_IDS.json').read_text())
    names=[t['function']['name'] for t in body['tools']]
    processors=[ToolJSONLogitsProcessor(tokenizer,data,1024,names,variant=v) for v in ['reference','optimized']]
    scores=torch.arange(len(tokenizer),dtype=torch.float32).unsqueeze(0)
    for size in range(len(ids)+1):
        prefix=torch.tensor([[0]*1024+ids[:size]])
        a,b=[p(prefix,scores) for p in processors]
        assert torch.equal(a,b),(row['label'],size)
    rows.append({'label':row['label'],'prefixes_checked':len(ids)+1,'exact_masks_equal':True,
        'stats':[p.stats for p in processors]})
out={'status':'PASSED','rows':rows,'seconds':time.time()-started,'scope':'Exact CPU masks on saved real token prefixes; no speed or model-quality claim'}
(root/'DECODER_EQUIVALENCE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
