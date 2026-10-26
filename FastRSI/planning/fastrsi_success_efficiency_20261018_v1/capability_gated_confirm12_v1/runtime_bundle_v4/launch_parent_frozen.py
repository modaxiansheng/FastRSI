"""Exact model/device and CPU environment launch descriptions for this development version."""
from pathlib import Path
import json,hashlib,shlex
M=Path(__file__).resolve().parent;B=M.parent
P=B/'fastrsi_fixed_retry32_20261016_v1';O=B/'liberopro_eval_20261012_v1';H=B/'fastrsi_visual_grasp_place_20261017_v1'
AS=Path('/mnt/data/task/yuchen/FastPhyagentos/frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1')
POLICY='sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81'
ENV='sha256:3171fda2362d773433482f79c4fd53f6f428413efbf41c3fdfb43d8c799d6c95'
def docker(name,cpus,memory,network='none'):
 return ['docker','run','-d','--name',name,'--network',network,'--cpus',str(cpus),'--memory',memory,'--pids-limit','2048','--shm-size','4g']
def add(args,env,volumes):
 for x in env:args+=['-e',x]
 for source,dest in volumes:
  assert Path(source).exists(),str(source)
  args+=['-v',str(source)+':'+dest]
 return args
def policy_command(root,node,gpu,model,conditions):
 tag=hashlib.sha256(str(root).encode()).hexdigest()[:8];a=docker('yuchen-vmb-'+node+'-'+model+'-g'+str(gpu)+'-'+tag+'-policy',8,'64g','host')
 a+=['--device','/dev/kfd','--device','/dev/mkfd','--device','/dev/dri','--group-add','video','--group-add','render']
 code=P/'source_frozen'/(model+'_R')/'code';checkpoint=AS/'weights'/dict(pi0='pi0_libero_pytorch_fp32',pi05='pi05_libero_pytorch_fp32',openvla='openvla_object',xvla='xvla_libero')[model]
 paths='/code:/workspace/PhyAgentOS-core:/frozen-code/scripts:/frozen-code/src'
 volumes=[(M/'runtime','/code:ro'),(M/'sources','/source:ro'),(root,'/pilot:rw'),(AS/'dependencies/PhyAgentOS-core','/workspace/PhyAgentOS-core:ro'),(code,'/frozen-code:ro'),(checkpoint,'/model:ro'),('/opt/hyhal','/opt/hyhal:ro'),('/etc/hfm','/etc/hfm:ro'),('/mnt/data/task/yuchen/FastPhyagentos/node_preservation_20261008_v1/node187/runtime_patch/ndarray.pyi','/usr/local/lib/python3.11/site-packages/numpydantic/ndarray.pyi:ro')]
 env=['HIP_VISIBLE_DEVICES='+str(gpu),'CUDA_VISIBLE_DEVICES='+str(gpu),'PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','LIBEROPRO_BASE_SUITE=libero_object','PAIRED_ALLOWED_CONDITIONS='+json.dumps(conditions,separators=(',',':')),'FROZEN_MULTI_VLA_SHA256='+hashlib.sha256((M/'sources/multi_vla_server.py').read_bytes()).hexdigest()]
 if model in ['pi0','pi05']:
  paths+=':/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src';volumes.extend([(AS/'dependencies/openpi','/workspace/openpi:ro'),(B/'fastrsi_object_frame_recovery_20261017_v1/tokenizer_cache','/tokenizer-cache:rw')]);env+=['OPENPI_DATA_HOME=/tokenizer-cache']
 else:
  paths='/overlay/site:'+paths;volumes.extend([(AS/'dependencies'/(model+'_overlay')/'site','/overlay/site:ro'),(AS/'dependencies','/vla-dependencies:ro'),(P/'source_frozen'/(model+'_R')/'vla_native','/vla-native:ro')]);env+=['HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','HF_HOME=/vla-dependencies/'+model+'_hf_cache','HF_MODULES_CACHE=/tmp/hf_modules','USE_TF=0','TF_NUM_INTRAOP_THREADS=2','TF_NUM_INTEROP_THREADS=2']
 env+=['PYTHONPATH='+paths];add(a,env,volumes)
 bootstrap="import torch,json,os,runpy; from pathlib import Path; assert torch.cuda.is_available() and torch.cuda.device_count()==1; Path('/pilot/DEVICE_BINDING.json').write_text(json.dumps(dict(device_count=1,name=torch.cuda.get_device_name(0),hip=os.environ['HIP_VISIBLE_DEVICES'],cuda=os.environ['CUDA_VISIBLE_DEVICES']))); runpy.run_path('/code/backend_server.py',run_name='__main__')"
 cmd=['python','-c',bootstrap,'--model',model,'--checkpoint','/model','--host','127.0.0.1','--port',str(31000+100*gpu),'--receipt-dir','/pilot/resident_receipts','--xvla-wire-chunk','1']
 return a+[POLICY,'timeout','14400','bash','-lc','source /opt/dtk/env.sh && source /opt/hyhal/env.sh && exec '+shlex.join(cmd)]
def environment_command(root,lane,node,gpu,row,kind):
 tag=hashlib.sha256(str(root).encode()).hexdigest()[:10];model=row['model'];s=row['source'];a=docker('yuchen-vmb-'+node+'-'+tag+'-'+kind,8,'16g','host')
 env=['PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=/eval/runtime_vendor:/eval/runtime:/eval/upstream:/core:/frozen-code/scripts:/frozen-code/src','LIBERO_CONFIG_PATH=/eval/libero-config','MUJOCO_GL=egl','LIBGL_ALWAYS_SOFTWARE=1','LIBEROPRO_BIND_ENV=1','LIBEROPRO_BASE_SUITE='+s['suite'],'LIBEROPRO_VARIANT='+s['variant'],'FASTRSI_DEV_ID='+row['physical_identity_id'],'FASTRSI_VLA_MODEL='+model,'OMP_NUM_THREADS=4','PI05_POLICY_PORT='+str(31000+100*gpu),'FASTRSI_RESIDENT_POLICY_PORT='+str(31000+100*gpu),'FASTRSI_RESIDENT_INFERENCE_HORIZON='+str(row['wire_horizon']),'FASTRSI_NATIVE_RUN=/pilot/'+row['physical_identity_id']+'/stream']
 volumes=[(M,'/eval:ro'),(root,'/pilot:rw'),(lane/'POLICY_COST_EVENTS.jsonl','/pilot/POLICY_COST_EVENTS.jsonl:ro'),(AS/'dependencies/PhyAgentOS-core','/core:ro'),(P/'source_frozen'/(model+'_R')/'code','/frozen-code:ro')]
 volumes.extend([(O/n,'/eval/'+n+':ro') for n in ['assets','upstream','runtime_vendor','libero-config']]);add(a,env,volumes)
 return a+[ENV,'timeout','3600' if kind=='base' else '1800','python','/eval/runtime/fixed_retry_worker.py' if kind=='base' else '/eval/runtime/multibackend_actor.py']
def vision_command(root,node):
 tag=hashlib.sha256(str(root).encode()).hexdigest()[:10];Z=B/'fastrsi_rgb_pose_recovery_20261017_v1';a=docker('yuchen-vmb-'+node+'-'+tag+'-vision',8,'16g')
 env=['PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=/code:/pose_deps/pose_aux:/pose_deps/megapose_cpu/src','CONDA_PREFIX=/pose_deps/venv','MEGAPOSE_DATA_DIR=/data','LIBGL_ALWAYS_SOFTWARE=1','EGL_PLATFORM=surfaceless','OMP_NUM_THREADS=8']
 volumes=[(root/'vision_rpc','/rpc:rw'),(M/'runtime','/code:ro'),(M/'vision_bundle','/input:ro'),(Z/'megapose_stage_v2/venv','/pose_deps/venv:ro'),(Z/'pose_aux','/pose_deps/pose_aux:ro'),(Z/'megapose_cpu','/pose_deps/megapose_cpu:ro'),(Z/'megapose_stage_v2/data/megapose-models','/data/megapose-models:ro'),(O/'upstream/libero/libero/assets','/geometry:ro')]
 add(a,env,volumes);return a+[ENV,'timeout','1200','/pose_deps/venv/bin/python','/code/vision_service_v3.py']

# Frozen implementations above; exact additional routing/admission interface below.
_original_policy_command=policy_command
_original_environment_command=environment_command
_original_vision_command=vision_command

def policy_command(root,node,gpu,model,conditions):
 base=list(conditions)+['wire-'+model]
 aliases={condition+'-'+branch:condition for condition in base for branch in ['frozen','candidate']}
 a=_original_policy_command(root,node,gpu,model,base+list(aliases));i=a.index(POLICY)
 a[i:i]=['-e','PAIRED_ALIAS_MAP='+json.dumps(aliases,separators=(',',':')),'-e','FROZEN_BACKEND_WRAPPER_SHA256='+hashlib.sha256((M/'runtime/backend_server_frozen.py').read_bytes()).hexdigest()]
 return a

def environment_command(root,lane,node,gpu,row,kind,branch=None):
 a=_original_environment_command(root,lane,node,gpu,row,kind)
 i=a.index(ENV);a[i:i]=['-v',str(M/'confirmation_generated')+':/eval/generated:ro']
 if branch=='candidate':
  i=a.index(ENV);a[i:i]=['-v',str(M/'candidate_runtime')+':/eval/runtime:ro']
 return a

def vision_command(root,node,branch):
 a=_original_vision_command(root,node)
 if branch=='candidate':a=[x.replace(str(M/'runtime')+':/code:ro',str(M/'candidate_runtime')+':/code:ro') for x in a]
 return a

def wire_command(root,node,gpu,model):
 a=docker('yuchen-ltc-wire-'+node+'-g'+str(gpu)+'-'+hashlib.sha256(str(root).encode()).hexdigest()[:8],2,'4g','host')
 env=['PYTHONPATH=/vendor:/code:/core:/frozen-code/scripts:/frozen-code/src','PYTHONDONTWRITEBYTECODE=1','FASTRSI_VLA_MODEL='+model,'PI05_POLICY_PORT='+str(31000+100*gpu)]
 add(a,env,[(O/'runtime_vendor','/vendor:ro'),(M/'runtime','/code:ro'),(root,'/out:rw'),(AS/'dependencies/PhyAgentOS-core','/core:ro'),(P/'source_frozen'/(model+'_R')/'code','/frozen-code:ro')])
 return a+[ENV,'timeout','300','python','/code/wire_contract_three.py']
