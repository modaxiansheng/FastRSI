"""Owned five-inference protocol probe on currently idle GPU136."""
import fcntl,hashlib,json,shutil,subprocess,time,socket
from pathlib import Path
from urllib.request import urlopen
from run_expert_first_change_batch_v2 import gate
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI');O=P/'runs/native_complete_flow_check_h136_v1_20261010'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 lock=Path('/var/tmp/yuchen-fastrsi-complete-flow-check-h136-v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 O.mkdir(exist_ok=False);gate(O);before=sha(P/'CURRENT_NATIVE_ENTRY.json');(O/'code').mkdir();(O/'service').mkdir();(O/'code/fastrsi').mkdir()
 (O/'code/fastrsi/__init__.py').write_text('')
 for n in ['serve_native_complete_flow_v1.py','check_native_complete_flow_v1.py','serve_native_value_v1.py','phase_iql_feedback_update_v1.py','phase_iql_value_v1.py','prefix_feature_capture_v1.py']:shutil.copy2(P/'scripts'/n,O/'code'/n)
 for n in ['native_value_binding_v1.py','complete_flow_memory.py']:shutil.copy2(P/'src/fastrsi'/n,O/'code/fastrsi'/n)
 shutil.copytree(P/'models/openpi_data',O/'tokenizer_cache')
 registry=P/'planning/complete_flow_memory_v1_20261010/REGISTRY.json';r=json.loads(registry.read_text());model=P/'planning/native_paired_q_update_v1_20261010/training/paired_router.npz';assert sha(model)==r['teacher_model_sha256']
 cases=[]
 for v,i in zip(range(12,16),range(32,36)):
  base=P/'planning'/f'native_value_phase_v{v}_20261010';paths={}
  for arm in ['N','R']:
   d=base/arm/'stream/value_service';call=json.loads((d/'VALUE_POLICY_CALLS.jsonl').read_text().splitlines()[0]);paths[arm]=(d/(call['evidence_name']+'_request.msgpack'),d/(call['evidence_name']+'_response.msgpack'))
  mode='R' if i==34 else 'N';q=paths['R'][0];e=paths[mode][1]
  cases.append(dict(initial=i,request=str(q.relative_to(P)),request_sha256=sha(q),expected_mode=mode,expected_response=str(e.relative_to(P)),expected_response_sha256=sha(e)))
 save(O/'SPEC.json',dict(endpoint='ws://127.0.0.1:57300',registry=str(registry.relative_to(P)),registry_sha256=sha(registry),model_sha256=sha(model),cases=cases))
 template=P/'planning/phase_iql_rollout_gpu0_v1_20261010';old=str(P/'runs/phase_iql_rollout_gpu0_v1_20261010')
 policy=json.loads((template/'POLICY_LAUNCH.json').read_text());policy=[x.replace(old,str(O)).replace('serve_phase_iql_candidates_v1.py','serve_native_complete_flow_v1.py') for x in policy];policy[policy.index('--name')+1]='yuchen-fastrsi-complete-flow-check-policy-v1-g136-20261010'
 image='sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81';index=policy.index(image)
 policy[index:index]=['-e','FASTRSI_NATIVE_VALUE_SERVICE=/check/service','-e','FASTRSI_NATIVE_VALUE_MODEL=/project/'+str(model.relative_to(P)),'-e','FASTRSI_COMPLETE_FLOW_REGISTRY=/project/'+str(registry.relative_to(P)),'-e','FASTRSI_COMPLETE_FLOW_REGISTRY_SHA256='+sha(registry),'-e','FASTRSI_COMPLETE_FLOW_MODE=C']
 case=json.loads((template/'init28_single_LAUNCH.json').read_text());case=[x.replace(old,str(O)) for x in case];case[case.index('--name')+1]='yuchen-fastrsi-complete-flow-check-case-v1-g136-20261010';index=case.index('/check/code/check_phase_iql_rollout_v1.py');case=case[:index]+['/check/code/check_native_complete_flow_v1.py']
 save(O/'REGISTRATION.json',dict(status='REGISTERED_BEFORE_REAL_PROTOCOL_OUTPUTS',spec_sha256=sha(O/'SPEC.json'),default_sha256=before,robot_actions=0,new_quality_or_efficiency_claim=False,source_sha256={str(f.relative_to(O/'code')):sha(f) for f in (O/'code').rglob('*') if f.is_file()}));save(O/'POLICY_LAUNCH.json',policy);save(O/'CASE_LAUNCH.json',case)
 owned=[];started=time.perf_counter()
 try:
  cid=subprocess.check_output(policy,text=True).strip();owned.append(cid);deadline=time.monotonic()+420
  while True:
   try:
    with urlopen('http://127.0.0.1:57300/healthz',timeout=2) as response:assert response.status==200
    break
   except Exception:
    assert json.loads(subprocess.check_output(['docker','inspect',cid]))[0]['State']['Running']
    if time.monotonic()>deadline:raise TimeoutError('policy readiness')
    time.sleep(2)
  save(O/'STARTUP.json',dict(seconds=time.perf_counter()-started));owned.append(case[case.index('--name')+1])
  with (O/'CASE.log').open('x') as f:run=subprocess.run(case,stdout=f,stderr=subprocess.STDOUT,timeout=180)
  assert run.returncode==0;assert json.loads((O/'RESULT.json').read_text())['status']=='PASSED_REAL_NATIVE_COMPLETE_FLOW_PACKET_BINDING';assert before==sha(P/'CURRENT_NATIVE_ENTRY.json')
 finally:
  stops=[]
  for cid in owned:
   (O/('LOG_'+cid+'.txt')).write_text(subprocess.check_output(['docker','logs',cid],text=True,stderr=subprocess.STDOUT));state=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
   if state['State']['Running']:subprocess.run(['docker','stop','-t','10',cid],check=True,capture_output=True)
   state=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert not state['State']['Running'] and state['State']['Pid']==0;stops.append(dict(name=state['Name'],state=state['State'],image=state['Image']))
  save(O/'STOP_RECEIPT.json',stops);save(O/'HOST.json',dict(seconds=time.perf_counter()-started,default_unchanged=before==sha(P/'CURRENT_NATIVE_ENTRY.json')))
 print((O/'RESULT.json').read_text(),flush=True)
if __name__=='__main__':main()
