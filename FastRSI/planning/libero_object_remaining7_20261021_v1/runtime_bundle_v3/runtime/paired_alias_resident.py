"""Distinct branch ownership with one actual baseline-end RNG snapshot; no sampling changes."""
from pathlib import Path
import os,json,time,hashlib
from paired_resident import CheckpointResident as FrozenResident
from paired_resident_io2 import CheckpointResident as PersistMethods

class CheckpointResident(FrozenResident):
 def __init__(self,native,**kw):
  super().__init__(native,**kw);self.aliases=json.loads(os.environ['PAIRED_ALIAS_MAP'])
  assert set(self.aliases).issubset(self.allowed)
  assert set(self.aliases.values()).issubset(self.allowed)
  assert not set(self.aliases).intersection(self.aliases.values())
 def write_json(self,path,value):
  with path.open('x') as f:json.dump(value,f,indent=2)
 persist=PersistMethods.persist
 def admin(self,command):
  with self.lock:
   fresh=command.get('request_id') not in self.receipts
   result=super().admin(command)
   if fresh and command.get('op')=='end':
    condition,mode=self.identify(command['session_id'])
    if mode=='base':
     assert condition not in self.aliases
     snapshot=self.checkpoints[condition];receipt=self.persist(condition,snapshot,'Actual closed native baseline; shared immutable state for ordinary, frozen and candidate extra attempts')
     for alias,canonical in self.aliases.items():
      if canonical==condition:
       assert alias not in self.checkpoints;self.checkpoints[alias]=snapshot
       self.write_json(self.receipt_dir/('ALIAS_'+hashlib.sha256(alias.encode()).hexdigest()+'.json'),dict(time=time.time(),alias=alias,canonical=canonical,checkpoint_sha256=receipt['sha256'],rng_hashes=snapshot.hashes,policy_sampling_modified=False))
   return result
