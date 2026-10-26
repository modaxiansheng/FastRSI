"""Bounded ENOENT retry for an owned Session registry; original parser stays authoritative."""
import errno,json,os,time
from pathlib import Path
DELAYS=(0.025,0.05,0.1,0.2,0.4)
def install():
 from PhyAgentOS.runtime.watchdog.registry import SessionRegistry
 if getattr(SessionRegistry,'_liberopro_io_retry_v2',False):return
 original=SessionRegistry._load_unlocked
 root=Path(os.environ['FASTRSI_NATIVE_RUN']).absolute()
 def load(self):
  path=Path(self.sessions_path).absolute()
  bounded=path.name=='SESSIONS.md' and root in path.parents
  for attempt in range(len(DELAYS)+1):
   try:return original(self)
   except FileNotFoundError as exc:
    if not bounded or exc.errno!=errno.ENOENT or attempt==len(DELAYS):raise
    receipt=dict(event='owned_session_read_retry',adapter='session_io_retry_v2',path=str(path),attempt=attempt+1,delay_seconds=DELAYS[attempt],wall_ns=time.time_ns())
    with (root/'SESSION_IO_RETRIES.jsonl').open('a') as out:out.write(json.dumps(receipt)+'\n')
    time.sleep(DELAYS[attempt])
 SessionRegistry._load_unlocked=load
 SessionRegistry._liberopro_io_retry_v2=True
