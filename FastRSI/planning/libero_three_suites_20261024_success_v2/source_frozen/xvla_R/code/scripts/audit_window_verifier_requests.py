"""Saved-request-only audit. No replays, relabelling, or robot executions."""
import hashlib,json,difflib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def leaves(a,b,path=''):
 if type(a)!=type(b):return [path]
 if isinstance(a,dict):return sum([leaves(a.get(k),b.get(k),path+'.'+k) for k in sorted(set(a)|set(b))],[])
 if isinstance(a,list):
  if len(a)!=len(b):return [path+'.length']
  return sum([leaves(x,y,path+f'[{i}]') for i,(x,y) in enumerate(zip(a,b))],[])
 return [] if a==b else [path]
def embedded(req):
 for m in req['messages']:
  for p in m['content'] if isinstance(m['content'],list) else []:
   if p.get('type')=='text' and 'VERIFICATION_BUNDLE:\n' in p['text']:return json.loads(p['text'].split('VERIFICATION_BUNDLE:\n',1)[1])
 raise ValueError('No saved bundle')
def main():
 audit=read(R/'planning/window_learning_v2_20261003/DRAWER_EVIDENCE_AUDIT.json');rows=[];requests=[];bundles=[];identities=[];sources={}
 for row in audit['rows']:
  if row['arm'] not in ('N','R'):continue
  p=R/row['request'];request=read(p);receipt=read(p.parent/'RECEIPT.json');response=read(p.parent/'RESPONSE.json');identity=read(p.parents[1]/'SERVICE_IDENTITY.json');raw=(p.parent/'RAW.txt').read_text();parsed=json.loads(raw)
  content=response['choices'][0]['message']['content'];assert json.loads(content)==parsed and parsed['verdict']==row['verdict']
  assert receipt['generation_finish_reason']=='eos' and not receipt['reached_token_limit']
  sources.update({str(v.relative_to(R)):sha(v) for v in [p,p.parent/'RAW.txt',p.parent/'RESPONSE.json',p.parent/'RECEIPT.json',p.parent/'MODEL_PROMPT.txt',p.parents[1]/'SERVICE_IDENTITY.json']})
  rows.append({'arm':row['arm'],'request_options':{k:v for k,v in request.items() if k!='messages'},'decoding':receipt['decoding'],'prompt_tokens':receipt['prompt_tokens'],'completion_tokens':receipt['completion_tokens'],'end':receipt['generation_finish_reason'],'parsed_label':parsed['verdict'],'raw_api_and_saved_label_agree':True,'identity':identity})
  requests.append(request);bundles.append(embedded(request));identities.append(identity)
 bind=R/'runs/window_learning_v2_20261003/MODEL_BINDING.json';binding=read(bind);sources[str(bind.relative_to(R))]=sha(bind)
 histdiff=list(difflib.unified_diff(bundles[0]['history_md'].splitlines(),bundles[1]['history_md'].splitlines(),n=0))
 # No ephemeral lease values exported; field names/hashes suffice.
 histdiff=[line for line in histdiff if 'claim_token' not in line]
 report={'scope':'Saved whole-input audit only; no verdict replacement','rows':rows,'full_requests_equal':requests[0]==requests[1],'full_request_differing_paths':leaves(requests[0],requests[1]),'bundle_differing_paths':leaves(bundles[0],bundles[1]),'history_diff':histdiff,'service_identity_differences':leaves(identities[0],identities[1]),'checkpoint':{'name':binding['model'],'revision':binding['revision'],'binding_status':binding['status']},'same_checkpoint_config_and_template':all(identities[0][k]==identities[1][k] for k in ('config_sha256','chat_template_sha256','model_class','transformers','torch')),'sources':sources,'conclusions':['Images identical; whole requests and rendered model prompts differ.','Same recorded Qwen3-VL checkpoint/config/template, Transformers 4.57.6 and Torch 2.7.1; SDPA backend in deployed source, not vLLM.','Both greedy, temperature zero, max 384; both EOS without truncation; opposing labels originated in raw generation, not parser.','Changed metadata/history and different GPUs remain confounders. No causal attribution or reproducibility guarantee from this audit.','Native task facts, original semantic outcomes and quarantine remain separate; no training on conflicting narrative.']}
 out=R/'planning/task_window_v1_20261003/VERIFIER_FULL_REQUEST_AUDIT.json'
 with out.open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps({k:report[k] for k in ('full_requests_equal','bundle_differing_paths','history_diff','service_identity_differences','conclusions')},indent=2))
if __name__=='__main__':main()
