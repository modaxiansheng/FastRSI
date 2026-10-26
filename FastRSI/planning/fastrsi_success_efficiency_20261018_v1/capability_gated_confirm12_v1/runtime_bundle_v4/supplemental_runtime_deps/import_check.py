"""Zero environment construction and zero policy RPC validation of actual baseline imports."""
from pathlib import Path
import importlib,hashlib,json,os,time
start=time.time();modules=[]
for name in ['retry_protocol.records','retry_protocol.fixed_retry','cost_trace','boundary_render_v2','fastrsi.resident_client','vision_skill','visual_kernel','control_mode_interface']:
 m=importlib.import_module(name);p=Path(m.__file__);modules.append(dict(module=name,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
from retry_protocol.fixed_retry import Identity
rows=json.loads(Path('/eval/FROZEN_PROTOCOL.json').read_text())['conditions']
for row in rows:Identity(**row['identity']).validate()
source=Path('/core/PhyAgentOS/runtime/targets/remote/libero/server.py');assert hashlib.sha256(source.read_bytes()).hexdigest()=='c89b483a3d5bf830b98a6470a6fa6c41e70ab12930fee4b7eab99b0d99fbba41'
Path('/pilot/IMPORT_CHECK.json').open('x').write(json.dumps(dict(time=time.time(),seconds=time.time()-start,model=os.environ['FASTRSI_VLA_MODEL'],modules=modules,identities_validated=len(rows),environment_constructors=0,physical_actions=0,model_calls=0,accepted=True),indent=2))
