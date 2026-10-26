"""Bounded retries for owned transient Session reads; original parser remains authoritative."""
import errno,json,os,time
from pathlib import Path
DELAYS=(0.025,0.05,0.1,0.2,0.4)
def install():
 from PhyAgentOS.runtime.watchdog.registry import SessionRegistry
 if getattr(SessionRegistry,'_liberopro_io_retry_v3',False):return
 # New case processes install only this version, never stack adapters.
 assert not getattr(SessionRegistry,'_liberopro_io_retry_v2',False)
 original=SessionRegistry._load_unlocked
 root=Path(os.environ['FASTRSI_NATIVE_RUN']).absolute()
 def load(self):
  path=Path(self.sessions_path).absolute()
  bounded=path.name=='SESSIONS.md' and root in path.parents
  for attempt in range(len(DELAYS)+1):
   try:return original(self)
   except (FileNotFoundError,ValueError) as exc:
    missing=isinstance(exc,FileNotFoundError) and exc.errno==errno.ENOENT
    fence=type(exc) is ValueError and str(exc)==str(self.sessions_path)+' does not contain a fenced YAML block'
    if not bounded or not (missing or fence) or attempt==len(DELAYS):raise
    receipt=dict(event='owned_session_read_retry',adapter='session_io_retry_v3',path=str(path),
      reason='ENOENT' if missing else 'MISSING_YAML_FENCE',attempt=attempt+1,
      delay_seconds=DELAYS[attempt],wall_ns=time.time_ns())
    with (root/'SESSION_IO_RETRIES.jsonl').open('a') as out:out.write(json.dumps(receipt)+'\n')
    time.sleep(DELAYS[attempt])
 SessionRegistry._load_unlocked=load
 SessionRegistry._liberopro_io_retry_v3=True
