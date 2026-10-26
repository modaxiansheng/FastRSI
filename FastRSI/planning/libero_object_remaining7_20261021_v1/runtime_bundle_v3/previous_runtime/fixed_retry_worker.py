"""Native baseline plus bounded unchanged-policy retries on frozen dev states."""
import asyncio,hashlib,importlib.util,json,os,time,traceback
from pathlib import Path
import numpy as np
from fastrsi.resident_client import ResidentClient
from retry_protocol.records import plain,atomic_receipt,Ledger
from retry_protocol.fixed_retry import Identity,Attempt,run_identity
from cost_trace import emit,span
from boundary_render_v2 import install as install_boundary

ROOT=Path(os.environ['FASTRSI_NATIVE_RUN']);case=json.loads((ROOT.parent/'ROSTER.json').read_text())[0]
OUT=ROOT/case['id'];OUT.mkdir(exist_ok=False)
identity=Identity(**case['development_identity']);identity.validate()
model=os.environ['FASTRSI_VLA_MODEL'];port=int(os.environ['FASTRSI_RESIDENT_POLICY_PORT'])
client=ResidentClient(port,case['id'],expected_action_horizon=int(os.environ['FASTRSI_RESIDENT_INFERENCE_HORIZON']))
path=Path('/core/PhyAgentOS/runtime/targets/remote/libero/server.py')
assert hashlib.sha256(path.read_bytes()).hexdigest()=='c89b483a3d5bf830b98a6470a6fa6c41e70ab12930fee4b7eab99b0d99fbba41'
spec=importlib.util.spec_from_file_location('fixed_retry_target',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
install_boundary(module,enabled=True,emit=emit)
runtime=module.LiberoRealRuntime();loop=asyncio.new_event_loop();asyncio.set_event_loop(loop)
protocol=Ledger(OUT/'PROTOCOL_EVENTS.jsonl',sync=True)
policy_ledger=ROOT.parent.parent/'POLICY_COST_EVENTS.jsonl'
current={};receipts=[];benchmarks=[];attempt_actions=[];started=time.time()
def digest(x):return hashlib.sha256(json.dumps(plain(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def state_digest(rt):return digest(np.asarray(rt.env.get_sim_state())[1:])
original_reset=module.LiberoRealRuntime.reset
def reset(rt,ctx):
    with span('environment_reset'):obs=original_reset(rt,ctx)
    raw=np.asarray(rt.init_states[0]);saved=np.load('/eval/generated/'+case['development_state_path'],allow_pickle=False)
    assert np.array_equal(raw,saved),'initial state array differs from frozen development state'
    current['reset_state_sha256']=state_digest(rt)
    if current['index']>1:assert current['reset_state_sha256']==receipts[0]['reset_state_sha256'],'retry reset did not restore same settled physical state'
    emit(dict(event='development_reset',attempt=current['index'],identity=identity.identity_id,initial_state_sha256=identity.initial_state_sha256,settled_state_sha256=current['reset_state_sha256'],settling_actions=10))
    from PIL import Image
    Image.fromarray(obs['agentview_image']).save(OUT/('attempt_%d'%current['index'])/'initial_front.png')
    return obs
module.LiberoRealRuntime.reset=reset
original_action=module.LiberoRealRuntime.action_chunk
def action_chunk(rt,chunk):
    before=rt.step_idx
    try:
        with span('environment_execution'):value=original_action(rt,chunk)
    finally:
        actions=np.asarray(chunk['actions'],dtype=np.float32)[:rt.step_idx-before]
        attempt_actions.extend(plain(actions));current['actions']=rt.step_idx
        emit(dict(event='actual_executed_actions',attempt=current['index'],actions=plain(actions),count=len(actions),step=rt.step_idx,native_success=bool(rt.env.check_success())))
    if rt.step_idx==50:
        from PIL import Image
        Image.fromarray(rt.observe()['agentview_image']).save(OUT/('attempt_%d'%current['index'])/'step50_front.png')
    return value
module.LiberoRealRuntime.action_chunk=action_chunk
original_infer=module._PolicyWsClient.infer
async def infer(policy,obs):
    if current.get('first_input_sha256') is None:
        current['first_input_sha256']=digest({k:v for k,v in obs.items() if k not in ('session_id','timestamp_ns','observation_id')})
    with span('policy_client'):value=await original_infer(policy,obs)
    current['client_success_calls']+=1
    return value
module._PolicyWsClient.infer=infer

def execute(index,budget,op):
    current.clear();current.update(index=index,client_success_calls=0,actions=0)
    attempt_actions.clear();attempt_start=time.time();rawdir=OUT/('attempt_%d'%index);rawdir.mkdir(exist_ok=False)
    offset=policy_ledger.stat().st_size if policy_ledger.exists() else 0
    physical=None;error=None;completed=False;status='ERROR';result=None
    try:
        if index>1 and runtime.env is not None:
            runtime.env.close();runtime.env=None;runtime._env_key=None
        admin=client.begin('retry-begin-'+case['id'],1600) if op=='begin' else client.handoff('retry-handoff-%d-'%index+case['id'])
        atomic_receipt(rawdir/'RESIDENT_ENTRY.json',admin)
        payload=dict(suite=case['suite'],episodes=[dict(task_id=case['task_id'],init_state_id=0,task_description=case['task_description'])],session_id=case['id'],attempt_index=index-1,max_steps=budget,seed=7,control_mode='absolute' if model=='xvla' else 'relative',policy_endpoint='ws://127.0.0.1:'+str(port),replan_every_steps=10,verification={'profile':'strict'},camera_height=256,camera_width=256,num_steps_wait=10,evidence_mode='all')
        with span('fixed_retry_native_attempt',attempt=index):result=loop.run_until_complete(runtime.benchmark(payload))
        assert len(result['episodes'])==1
        episode=result['episodes'][0];benchmarks.append(episode)
        if episode.get('error_code') is None:physical=bool(episode['success']);completed=True;status='OK'
        else:error=episode.get('error_message') or episode['error_code']
    except BaseException:error=traceback.format_exc()
    events=[];tail=b''
    if policy_ledger.exists():
        with policy_ledger.open('rb') as f:f.seek(offset);tail=f.read()
        (rawdir/'POLICY_EVENTS.jsonl').write_bytes(tail)
        events=[json.loads(line) for line in tail.splitlines()]
    counts={name:sum(e.get('event')==name for e in events) for name in ['policy_rpc_begin','policy_rpc_end','action_generation_begin','action_generation_end','neural_forward_begin','neural_forward_end']}
    raw=dict(index=index,result=result,error=error,physical_success=physical,software_status=status,episode_completed=completed,actions=current['actions'],executed_actions=attempt_actions.copy(),reset_state_sha256=current.get('reset_state_sha256'),first_input_sha256=current.get('first_input_sha256'),final_state_sha256=state_digest(runtime) if runtime.env is not None else None,policy_counts=counts,client_success_calls=current['client_success_calls'],seconds=time.time()-attempt_start,initial_state_sha256=identity.initial_state_sha256,policy_ledger_offset=offset,policy_ledger_bytes=len(tail),policy_ledger_sha256=hashlib.sha256(tail).hexdigest(),qwen_calls=0,qwen_tokens=0,money=None)
    atomic_receipt(rawdir/'RAW.json',raw);receipts.append(raw)
    rawsha=hashlib.sha256((rawdir/'RAW.json').read_bytes()).hexdigest()
    return Attempt(index=index,physical_success=physical,software_status=status,verification_status='NOT_RUN',initial_state_sha256=identity.initial_state_sha256,actions=current['actions'],policy_inferences=counts['action_generation_end'] if tail else None,forward_calls=counts['neural_forward_end'] if tail else None,seconds=raw['seconds'],qwen_calls=0,qwen_tokens=0,observation_sha256=current.get('first_input_sha256'),action_trace_sha256=digest(attempt_actions),final_state_sha256=raw['final_state_sha256'],raw_receipt_sha256=rawsha,rng_operation=op,episode_completed=completed,error=error)

try:
    result=run_identity(identity,model,execute,protocol.emit,deterministic_decoding_verified=model=='openvla')
    atomic_receipt(OUT/'PROTOCOL_RESULT.json',result)
    atomic_receipt(OUT/'RESULT.json',dict(case=case,chain_status='PASSED' if result['software_ok'] else 'FAILED',status='PASSED' if result['software_ok'] else 'FAILED',environment_success=result['final_success'],physical_first_success=result['first_success'],physical_final_success=result['final_success'],semantic_success=None,seconds=time.time()-started,benchmark={'episodes':benchmarks},scope='fixed simulator-feedback retry on new development states, first attempt shared baseline',protocol_result=result,agent_model_calls=0,verifier_model_calls=0,qwen_tokens=0))
finally:
    close_error=None;end=None
    try:
        if runtime.env is not None:runtime.env.close()
        if client.status().get('active') is not None:end=client.end('retry-end-'+case['id'])
    except BaseException:close_error=traceback.format_exc()
    atomic_receipt(OUT/'CASE_LIFECYCLE.json',dict(seconds=time.time()-started,resident_end=end,close_error=close_error,recorded_attempts=len(receipts),recorded_actions=sum(r['actions'] for r in receipts),money=None))
    loop.close()
    if close_error:raise RuntimeError(close_error)
