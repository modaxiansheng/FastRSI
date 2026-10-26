"""Transport-only, exact in-memory baseline-end RNG checkpoint for paired extra attempts."""
import copy,hashlib,json,time
from pathlib import Path
from fastrsi.resident_policy import ResidentPolicy,RNGSnapshot

class CheckpointResidentOriginal(ResidentPolicy):
    def __init__(self,native,*,guard,receipt_dir,allowed,**kw):
        super().__init__(native,guard=guard,**kw)
        self.checkpoints={};self.allowed=set(allowed);self.receipt_dir=Path(receipt_dir)
    def identify(self,sid):
        condition,mode=sid.rsplit('--',1)
        assert condition in self.allowed and mode in ['base','ordinary','visual'],sid
        return condition,mode
    def admin(self,command):
        with self.lock:
            fresh=command.get('request_id') not in self.receipts
            operation=command.get('op');identity=mode=None;snapshot=None
            if operation in ['begin','end']:
                identity,mode=self.identify(command['session_id'])
            if operation=='end' and mode=='base' and fresh:
                assert self.active and self.active['session_id']==command['session_id']
                snapshot=RNGSnapshot()
            result=super().admin(command)
            if operation=='end' and mode=='base' and fresh:
                assert snapshot.hashes==RNGSnapshot().hashes,'End unexpectedly consumed RNG'
                assert identity not in self.checkpoints
                self.checkpoints[identity]=snapshot
                p=self.receipt_dir/('BASELINE_END_'+hashlib.sha256(identity.encode()).hexdigest()+'.json')
                with p.open('x') as f:json.dump(dict(time=time.time(),condition=identity,rng_hashes=snapshot.hashes,end_receipt=result,source='Actual native baseline-end state, no noise-draw reconstruction'),f,indent=2)
            if operation=='begin' and fresh:
                if mode!='base':
                    assert identity in self.checkpoints,'Missing completed baseline checkpoint'
                    self.checkpoints[identity].restore();self.guard.reset()
                result.update(paired_start_rng_hashes=RNGSnapshot().hashes,paired_condition=identity,paired_mode=mode)
                self.receipts[command['request_id']]=(copy.deepcopy(command),copy.deepcopy(result))
            return result


class CheckpointResident(CheckpointResidentOriginal):
    """IO v2: serialize actual checkpoints; restore old hashes using audited noise draws only."""
    def __init__(self,native,**kw):
        import os
        super().__init__(native,**kw)
        self.model=os.environ['IO2_MODEL']
        plan=json.loads(Path('/pilot/RESTORE_PLAN.json').read_text())
        assert self.rng.hashes==plan['old_startup_rng_hashes'],'Postload state changed'
        restored=[]
        for item in plan['checkpoints']:
            self.rng.restore();self.guard.reset()
            count=item['true_generations']
            if self.model in ['pi0','pi05']:
                m=native.policy._model
                for _ in range(count):m.sample_noise((1,m.config.action_horizon,m.config.action_dim),native.policy._pytorch_device)
            else:
                assert self.model=='openvla','No unpersisted X-VLA baseline requires reconstruction'
            snap=RNGSnapshot()
            assert snap.hashes==item['expected_rng_hashes'],('Exact prior baseline RNG reconstruction failed',item['condition'],snap.hashes,item['expected_rng_hashes'])
            self.checkpoints[item['condition']]=snap
            restored.append(self.persist(item['condition'],snap,'Prior closed native baseline; original noise draws only, zero neural forwards and zero environment actions',item))
        self.rng.restore();self.guard.reset()
        assert RNGSnapshot().hashes==plan['old_startup_rng_hashes']
        self.write_json(self.receipt_dir/'IO2_RESTORE_ACCEPTANCE.json',dict(time=time.time(),postload_rng_match=True,restored=restored,neural_forwards=0,environment_actions=0,source='Bounded exact reconstruction, all RNG domains equal original hashes'))
    def write_json(self,path,v):
        with path.open('x') as f:json.dump(v,f,indent=2)
    def persist(self,identity,snapshot,source,evidence=None):
        import pickle,os
        key=hashlib.sha256(identity.encode()).hexdigest()
        payload=dict(python=snapshot.python,numpy=snapshot.numpy,cpu=snapshot.cpu,cuda=snapshot.cuda,cuda_count=snapshot.cuda_count,hashes=snapshot.hashes)
        raw=pickle.dumps(payload,protocol=4);path=self.receipt_dir/('CHECKPOINT_'+key+'.pickle');tmp=path.with_suffix('.tmp')
        assert not path.exists()
        with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        tmp.replace(path)
        import random
        before=RNGSnapshot();back=pickle.loads(path.read_bytes());assert back['hashes']==snapshot.hashes
        random.setstate(back['python']);snapshot.np.random.set_state(back['numpy']);snapshot.torch.set_rng_state(back['cpu']);snapshot.torch.cuda.set_rng_state_all(back['cuda'])
        assert RNGSnapshot().hashes==snapshot.hashes
        before.restore()
        receipt=dict(time=time.time(),condition=identity,path=str(path),sha256=hashlib.sha256(raw).hexdigest(),rng_hashes=snapshot.hashes,source=source,evidence=evidence,roundtrip_verified=True)
        self.write_json(path.with_suffix('.json'),receipt);return receipt
    def admin(self,command):
        with self.lock:
            fresh=command.get('request_id') not in self.receipts
            result=super().admin(command)
            if fresh and command.get('op')=='end':
                identity,mode=self.identify(command['session_id'])
                if mode=='base':self.persist(identity,self.checkpoints[identity],'Actual native baseline-end state, all RNG domains serialized')
            return result
