"""Isolated fourth-attempt server. Native full sampling; no environment access."""
import argparse,asyncio,copy,hashlib,json,os,random,sys,time,types
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(os.environ['LIBEROPRO_CAMPAIGN_CODE'])/'scripts'))
from PhyAgentOS.runtime.policy.openpi.native_openpi_server import NativeOpenPIPolicy
from fastrsi.resident_policy import ResidentPolicy,OpenPIStateGuard,RNGSnapshot
from resident_openpi_server import serve
from pi_native_meter_v8 import install
from candidate_selector import select,seed_for
from session_namespace import resolve_session

def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def obs_hash(obs):
    h=hashlib.sha256()
    for k in sorted(obs):
        h.update(k.encode());v=obs[k]
        if isinstance(v,np.ndarray):h.update(str(v.dtype).encode()+str(v.shape).encode()+v.tobytes())
        else:h.update(json.dumps(v,sort_keys=True).encode())
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight-only',action='store_true');ap.add_argument('--model',choices=['pi05'],required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,required=True);ap.add_argument('--receipt-dir',type=Path,required=True);ap.add_argument('--xvla-wire-chunk',type=int);a=ap.parse_args()
    manifest=json.loads(Path('/eval/FROZEN_PROTOCOL.json').read_text());rows={r['identity_id']:r for r in manifest['rows']}
    a.receipt_dir.mkdir(parents=True,exist_ok=True);ledger=a.receipt_dir.parent/'POLICY_COST_EVENTS.jsonl';owner=None
    context=dict(phase='load',block=None,candidate=None,branch=None,identity=None)
    def event(kind,**fields):
        with ledger.open('a') as f:f.write(json.dumps(dict(event=kind,time=time.time(),mono_ns=time.monotonic_ns(),session_id=owner.active['session_id'] if owner and owner.active else None,**context,**fields))+'\n')
    random.seed(7);np.random.seed(7);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
    native=NativeOpenPIPolicy(policy_config='pi05_libero',checkpoint_dir=a.checkpoint,default_prompt=None,pytorch_device='cuda:0',disable_torch_compile=True,torch_seed=7)
    assert native.metadata['action_horizon']==10
    model=native.policy._model;original_noise=model.sample_noise;captured=[]
    def noise(self,shape,device):
        out=original_noise(shape,device);arr=out.detach().cpu().numpy().copy();captured.append(ahash(arr));return out
    model.sample_noise=types.MethodType(noise,model)
    meter=install(native,event);guard=OpenPIStateGuard(native);postload=RNGSnapshot();native_infer=native.infer
    shape=(1,model.config.action_horizon,model.config.action_dim)
    assert shape==(1,10,32)
    restored={}
    for identity,row in rows.items():
        assert postload.hashes==row['postload_rng_hashes'],('Postload RNG mismatch',identity)
        postload.restore();t=time.perf_counter()
        for _ in range(row['prior_generations']):original_noise(shape,'cuda:0')
        snapshot=RNGSnapshot();assert snapshot.hashes==row['after_third_rng_hashes'],('Fourth RNG reconstruction failed',identity,snapshot.hashes,row['after_third_rng_hashes'])
        restored[identity]=snapshot
        event('fourth_rng_reconstruction',reconstructed_identity=identity,noise_draws=row['prior_generations'],neural_inferences=0,seconds=time.perf_counter()-t,hashes=snapshot.hashes,verified=True)
    def one(obs):
        guard.check();before=obs_hash(obs);captured.clear();t=time.perf_counter()
        output=native_infer(copy.deepcopy(obs));torch.cuda.synchronize()
        assert obs_hash(obs)==before and len(captured)==1
        guard.check();actions=np.asarray(output['actions']);assert actions.shape==(10,7) and np.isfinite(actions).all()
        evidence=dict(noise_sha256=captured[0],actions_sha256=ahash(actions),seconds=time.perf_counter()-t,input_sha256=before,guard_state_sha256=guard.digest,actions=actions.tolist())
        event('candidate_generated',**evidence)
        return output,evidence
    # Synthetic protocol preflight only: no environment, task transitions or outcome labels.
    context.update(phase='preflight');postload.restore();guard.reset()
    obs={'observation/image':np.zeros((256,256,3),dtype=np.uint8),'observation/wrist_image':np.zeros((256,256,3),dtype=np.uint8),'observation/state':np.zeros(8,dtype=np.float32),'prompt':next(iter(rows.values()))['source']['language']}
    original_snapshot=RNGSnapshot();o1,e1=one(obs);after=RNGSnapshot();original_snapshot.restore();guard.reset();o2,e2=one(obs)
    assert np.array_equal(o1['actions'],o2['actions']) and e1['noise_sha256']==e2['noise_sha256']
    assert RNGSnapshot().hashes==after.hashes
    diversity=[]
    for i in [1,2,3]:
        original_snapshot.restore();guard.reset();torch.cuda.manual_seed_all(seed_for('preflight',0,i));out,ev=one(obs);diversity.append(ev)
    assert len({e1['noise_sha256'],*[e['noise_sha256'] for e in diversity]})==4
    assert len({e1['actions_sha256'],*[e['actions_sha256'] for e in diversity]})>1
    postload.restore();guard.reset()
    (a.receipt_dir/'PREFLIGHT_ACCEPTANCE.json').write_text(json.dumps(dict(no_environment=True,physical_attempts=0,real_neural_generations=5,same_rng_same_actions=True,independent_noise_hashes=[e1['noise_sha256']]+[e['noise_sha256'] for e in diversity],action_hashes=[e1['actions_sha256']]+[e['actions_sha256'] for e in diversity],reconstructed_identities=len(rows),guard=guard.reset()),indent=2))
    class FourthResident(ResidentPolicy):
        def admin(self,command):
            with self.lock:
                fresh=command.get('request_id') not in self.receipts
                if command.get('op')=='begin':
                    resolve_session(command['session_id'],rows)
                result=super().admin(command)
                if command.get('op')=='begin' and fresh:
                    identity,branch=resolve_session(command['session_id'],rows);restored[identity].restore()
                    result.update(fourth_rng_hashes=RNGSnapshot().hashes,prior_generations=rows[identity]['prior_generations'],branch=branch)
                    self.receipts[command['request_id']]=(copy.deepcopy(command),copy.deepcopy(result))
                    context.update(phase='physical_attempt',identity=identity,branch=branch,block=None,candidate=None)
                    event('fourth_begin',hashes=result['fourth_rng_hashes'])
                return result
    def measured(obs):
        identity,branch=resolve_session(owner.active['session_id'],rows);block=owner.active['inferences']-1
        context.update(phase='physical_attempt',identity=identity,branch=branch,block=block,candidate=0)
        event('policy_rpc_begin');t=time.perf_counter();start=RNGSnapshot();guard.reset()
        outputs=[];proofs=[];o,e=one(obs);outputs.append(o);proofs.append(e);main_after=RNGSnapshot()
        if branch=='candidate4':
            for i in [1,2,3]:
                start.restore();guard.reset();torch.cuda.manual_seed_all(seed_for(identity,block,i));context['candidate']=i
                o,e=one(obs);outputs.append(o);proofs.append(e)
            assert len({x['noise_sha256'] for x in proofs})==4,'candidate noises duplicated'
            idx,details=select([o['actions'] for o in outputs],manifest['selection']['scale'])
        else:idx=0;details={}
        # Only the ordinary main stream is carried forward; auxiliary streams do not contaminate it.
        main_after.restore();guard.check();assert RNGSnapshot().hashes==main_after.hashes
        out=outputs[idx];context['candidate']=None
        event('candidate_selection',selected=idx,candidates=proofs,selection=details,selected_actions=out['actions'].tolist(),selected_sha256=ahash(out['actions']),main_rng_after=main_after.hashes,unmodified_original=True,seconds=time.perf_counter()-t)
        event('policy_rpc_end',returned_actions=10,seconds=time.perf_counter()-t)
        return out
    native.infer=measured
    native.metadata.update(system_vla_model='pi05',native_meter=meter,fourth_candidate4=True,checkpoint_config_sha256=hashlib.sha256((Path(a.checkpoint)/'config.json').read_bytes()).hexdigest())
    owner=FourthResident(native,guard=guard,rng=postload,max_sessions=32,max_inferences=2000)
    (a.receipt_dir/'STARTUP.json').write_text(json.dumps(dict(metadata=owner.metadata,rng_hashes=owner.rng.hashes,policy_state=guard.reset(),arguments={k:str(v) for k,v in vars(a).items()}),indent=2))
    context.update(phase='ready',identity=None,branch=None,block=None,candidate=None)
    if a.preflight_only:return
    asyncio.run(serve(owner,host=a.host,port=a.port,receipt_dir=a.receipt_dir))
if __name__=='__main__':main()
