"""Only the transport owner class changes; frozen native loading and metering remain exact."""
from pathlib import Path
import hashlib,os
p=Path(__file__).with_name('backend_server_frozen.py');assert hashlib.sha256(p.read_bytes()).hexdigest()==os.environ['FROZEN_BACKEND_WRAPPER_SHA256'];s=p.read_text();old='from paired_resident import CheckpointResident';assert s.count(old)==1;s=s.replace(old,'from paired_alias_resident import CheckpointResident');exec(compile(s,str(p),'exec'),dict(__name__='__main__',__file__=str(p)))
