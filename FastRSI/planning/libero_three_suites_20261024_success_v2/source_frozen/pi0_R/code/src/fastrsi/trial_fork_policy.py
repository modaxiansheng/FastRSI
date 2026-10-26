"""Same-service development trial checkpoint; no cross-model or online routing.

Only the declared source Session creates a checkpoint. Only a fresh child
Session, before connecting any inference owner, may restore it. Everything is
held in memory; no untrusted pickle or client-supplied RNG is deserialized.
"""
import copy
import hashlib
import json
import time
from .resident_policy import ResidentPolicy, ResidentProtocolError, RNGSnapshot, VERSION


class TrialForkPolicy(ResidentPolicy):
    def __init__(self, native, *, source_session, prefix_inferences, **kwargs):
        if not isinstance(source_session,str) or not source_session:
            raise ValueError('Explicit source Session required')
        if type(prefix_inferences) is not int or not 0 < prefix_inferences < 52:
            raise ValueError('Bounded nonterminal prefix required')
        super().__init__(native, **kwargs)
        self.source_session=source_session
        self.prefix_inferences=prefix_inferences
        self.checkpoint=None
        self.prefix_bindings=[]
        self.fork_receipts={}

    def infer(self, observation, *, connection_id):
        with self.lock:
            result=super().infer(observation, connection_id=connection_id)
            active=self.active
            if active['session_id']==self.source_session and active['inferences']<=self.prefix_inferences:
                started=time.perf_counter()
                binding=dict(prompt=observation.get('prompt'),
                    inputs={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in observation.items() if hasattr(v,'tobytes')},
                    actions=hashlib.sha256(result['actions'].tobytes()).hexdigest())
                self.prefix_bindings.append(binding)
                if active['inferences']==self.prefix_inferences:
                    rng=RNGSnapshot(numpy_module=self.rng.np,torch_module=self.rng.torch)
                    digest=hashlib.sha256(json.dumps(self.prefix_bindings,sort_keys=True).encode()).hexdigest()
                    self.checkpoint=dict(rng=rng,source_session=self.source_session,
                        source_inferences=self.prefix_inferences,prefix_sha256=digest,
                        attention=[x._attn_implementation for x in getattr(self.guard,"configs",[])])
                    result['policy_meta']['trial_checkpoint']=dict(prefix_sha256=digest,
                        rng_hashes=rng.hashes,source_inferences=self.prefix_inferences)
                result['policy_meta']['trial_checkpoint_capture_seconds']=time.perf_counter()-started
        return result

    def admin(self, command):
        if command.get('op')!='restore_trial_prefix':
            with self.lock:
                if command.get('request_id') in self.fork_receipts:
                    raise ResidentProtocolError('Request identity belongs to trial restore')
                return super().admin(command)
        with self.lock:
            if set(command)!={'op','request_id','session_id','source_session','source_inferences'}:
                raise ResidentProtocolError('Unexpected trial restore fields')
            request_id=command['request_id']
            if not isinstance(request_id,str) or not 0<len(request_id)<=160:
                raise ResidentProtocolError('Invalid request identity')
            if request_id in self.receipts:
                raise ResidentProtocolError('Request identity belongs to another operation')
            if request_id in self.fork_receipts:
                previous,receipt=self.fork_receipts[request_id]
                if previous!=command:raise ResidentProtocolError('Conflicting restore retry')
                return copy.deepcopy(receipt)
            a=self.active;c=self.checkpoint
            if (self.broken or a is None or c is None or a['session_id']!=command['session_id']
                    or a['session_id']==self.source_session or a['owner'] is not None or a['inferences']!=0
                    or command['source_session']!=c['source_session']
                    or type(command['source_inferences']) is not int
                    or command['source_inferences']!=c['source_inferences']
                    or a['max_inferences']<=c['source_inferences']):
                raise ResidentProtocolError('Restore requires matching paid prefix and fresh child Session')
            tick=time.perf_counter()
            try:
                self.guard.check()
                for cfg,attention in zip(getattr(self.guard,"configs",[]),c["attention"]):
                    cfg._attn_implementation=attention
                hashes=c['rng'].restore()
            except BaseException:
                self.broken=True
                raise
            a['inferences']=c['source_inferences']  # Virtual prefix still consumes original attempt budget.
            receipt=dict(version=VERSION,op=command['op'],request_id=request_id,
                session_id=a['session_id'],generation=a['generation'],
                inherited_inferences=c['source_inferences'],prefix_sha256=c['prefix_sha256'],
                restored_rng_hashes=hashes,seconds=time.perf_counter()-tick,
                scope='Paid development prefix reuse; not newly executed model calls')
            self.fork_receipts[request_id]=(copy.deepcopy(command),copy.deepcopy(receipt))
            return receipt
