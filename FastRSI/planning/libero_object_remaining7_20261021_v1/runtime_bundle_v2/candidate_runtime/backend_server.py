"""Pinned native loaders, unchanged model sampling; only metering and paired transport extended."""
from pathlib import Path
import os,hashlib
source=Path('/source/multi_vla_server.py');text=source.read_text()
assert hashlib.sha256(source.read_bytes()).hexdigest()==os.environ['FROZEN_MULTI_VLA_SHA256']
def replace(old,new,count=1):
    global text
    assert text.count(old)==count,(old,text.count(old))
    text=text.replace(old,new)
replace("choices=['pi0', 'xvla', 'openvla']","choices=['pi0','pi05','xvla','openvla']")
replace("if args.model == 'pi0':","if args.model in ['pi0','pi05']:")
replace("policy_config='pi0_libero'","policy_config=args.model+'_libero'")
replace('guard = OpenPIStateGuard(native)','guard = None  # Bind guard after the existing component meter is installed.')
replace("unnorm_key='libero_10'","unnorm_key=os.environ['LIBEROPRO_BASE_SUITE']")
replace("if args.model != 'xvla':","if args.model == 'openvla':",2)
replace('    infer = native.infer',"    if args.model in ['pi0','pi05']:\n        from pi_native_meter_v8 import install\n        native.metadata['native_meter']=install(native,event)\n        guard=OpenPIStateGuard(native)\n    infer = native.infer")
replace('    owner = ResidentPolicy(native, guard=guard, max_sessions=1024, max_inferences=2000)',"    from paired_resident import CheckpointResident\n    owner=CheckpointResident(native,guard=guard,receipt_dir=args.receipt_dir,allowed=json.loads(os.environ['PAIRED_ALLOWED_CONDITIONS']),max_sessions=1024,max_inferences=2000)")
exec(compile(text,str(source),'exec'),dict(__name__='__main__',__file__=str(source),os=os))
