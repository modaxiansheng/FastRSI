"""Transport-only, exact in-memory baseline-end RNG checkpoint for paired extra attempts."""
import copy,hashlib,json,time
from pathlib import Path
from fastrsi.resident_policy import ResidentPolicy,RNGSnapshot

class CheckpointResident(ResidentPolicy):
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
