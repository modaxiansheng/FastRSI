"""CPU-only native token counting with all network connections blocked."""
import ast,hashlib,inspect,json,os,socket,time
from pathlib import Path
import tiktoken_ext.openai_public as public
from PhyAgentOS.utils.helpers import estimate_prompt_tokens_chain

root=Path('/project');cache=Path(os.environ['TIKTOKEN_CACHE_DIR']);out=Path(os.environ['FASTRSI_TOKEN_CACHE_RECEIPT'])
fn=ast.parse(inspect.getsource(public.cl100k_base));expected=next(k.value.value for n in ast.walk(fn) if isinstance(n,ast.Call) for k in n.keywords if k.arg=='expected_hash')
asset=cache/'9b5ad71b2ce5302211f9c61530b329a4922fc6a4';actual=hashlib.sha256(asset.read_bytes()).hexdigest();assert actual==expected
attempts=[]
def blocked(*args,**kwargs):
    attempts.append('connect');raise AssertionError('Token counting attempted network access')
socket.socket.connect=blocked;socket.create_connection=blocked
rows=[];start=time.perf_counter()
for arm in ['N','F']:
    run=root/'runs/sampling_budget_v2_20261002/pair0'/arm/'stream'
    for p in sorted(run.glob('service_calls/*/REQUEST.json')):
        request=json.loads(p.read_text())
        if not request.get('tools'):continue
        t=time.perf_counter();count,source=estimate_prompt_tokens_chain(object(),request.get('model'),request['messages'],request['tools'])
        assert count>0 and source=='tiktoken'
        rows.append({'input':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'count':count,'source':source,'seconds':time.perf_counter()-t})
assert len(rows)==6 and not attempts
short,source=estimate_prompt_tokens_chain(object(),None,[{'role':'user','content':'native token counter check'}],None)
long,_=estimate_prompt_tokens_chain(object(),None,[{'role':'user','content':'hello '*70000}],None)
assert 0<short<65536<long and not attempts
result={'status':'PASSED','scope':'Native counter with same verified cl100k asset, offline. Trigger thresholds and core code unchanged; no Agent efficacy claim.',
    'asset_sha256':actual,'asset_expected_sha256':expected,'asset_bytes':asset.stat().st_size,'rows':rows,'threshold_fixture':{'short':short,'long':long,'unchanged_threshold':65536},'network_attempts':len(attempts),'seconds':time.perf_counter()-start}
with out.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
