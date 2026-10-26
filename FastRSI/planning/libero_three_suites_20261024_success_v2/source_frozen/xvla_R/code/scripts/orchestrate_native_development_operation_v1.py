"""Freeze/preflight/dispatch matched development consequences; no Agent claim.

Owner explicitly runs each mode. Freeze is local filesystem only; preflight is
CPU-only and never resets a simulator. Execute consumes a successful frozen CPU
receipt, and launches at most two paired seeds with fresh policy services.
"""
import argparse
from datetime import datetime
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from urllib.request import build_opener, ProxyHandler

NAME='native_development_operation_v2_20261007'
REMOTE_PROJECT=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
CPU_IMAGE='sha256:3171fda2362d773433482f79c4fd53f6f428413efbf41c3fdfb43d8c799d6c95'
POLICY_IMAGE='sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81'
COLLECTION='planning/task64_native_development_sampler_v1_20261007/outputs/samples'
REGISTRATION='planning/task64_native_development_sampler_v1_20261007/REGISTRATION.json'
SOURCE_PLAN='planning/dual_anchor_operation_value_v1_20261007/EXECUTION_PLAN_V2.json'
MODEL_FILES={
 'config.json':'06bbdc28bc7b7cf8cdae483099ee626054c0f085c71ed5fc398aa1d7da64715f',
 'conversion_manifest.json':'66d46396ff5e645cebe303ff8989ff47a738d6da60cc7a9fac95aeb71e541179',
 'assets/physical-intelligence/libero/norm_stats.json':'b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84',
 'model.safetensors':'2e967a34251e5f69559314d3f1dd999212993226797f87e67e9ee34084f2fd69'}
SCRIPTS=['probe_native_development_operation_v1.py','probe_dual_anchor_operation_value_v1.py','replay_stack_progress.py','stage_trace.py','orchestrate_native_development_operation_v1.py']
MODULES=['__init__.py','native_sampler_draw_restore_v1.py','native_sampler_restore_v1.py','experience_compiler.py','live_controller_trace.py','dual_anchor_retarget_v1.py','goal_skill.py','planar_binding.py','retarget_geometry.py','visual_binding.py','native_validation_request_v1.py','native_goal_component_v1.py']
TESTS=['test_native_sampler_restore_v1.py','test_native_sampler_draw_restore_v1.py','test_native_development_orchestrator_v1.py']
NATIVE='third_party/PhyAgentOS-core/PhyAgentOS/runtime/targets/remote/libero/server.py'


def sha(path):
    out=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):out.update(chunk)
    return out.hexdigest()

def save(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def parse_docker_timestamp(value):
    """RFC3339 nanoseconds accepted on Python3.8; raw strings stay in receipts."""
    match=re.fullmatch(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})',value)
    if match is None:raise ValueError('Unsupported Docker timestamp')
    whole,fraction,zone=match.groups()
    micros=(fraction or '').ljust(6,'0')[:6]
    return datetime.fromisoformat(whole+'.'+micros+('+00:00' if zone=='Z' else zone))


def resolve_execution_request(args,plan):
    """Bind an actual exported model request; parsing never invents a request."""
    supplied=[args.validation_request,args.validation_request_sha,args.request_workspace,
        args.current_component_state_sha256,args.request_model_config,args.request_model_config_sha256]
    if args.operation_diagnostic_only:
        if any(value is not None for value in supplied):raise ValueError('Diagnostic mode cannot impersonate a model request')
        return dict(mode='OWNER_OPERATION_DIAGNOSTIC_ONLY',development_seeds=list(plan['development_seeds']),
            model_requested=False,request_sha256=None,full_agent_chain=False,learning_gain_claimed=False)
    if any(value is None for value in supplied):raise ValueError('Exact exported model request, workspace, state and MODEL_CONFIG bindings required')
    request_path=Path(args.validation_request).resolve();workspace=Path(args.request_workspace).resolve()
    if request_path.name!='VALIDATED_VALIDATION_REQUEST.json' or workspace!=request_path.parent/'agent_session/workspace':
        raise ValueError('Request/workspace must retain the actual model proposal layout')
    for path in (Path(args.validation_request),Path(args.request_workspace),Path(args.request_model_config)):
        if path.is_symlink():raise ValueError('Request input symlink rejected')
    raw=request_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.validation_request_sha:raise ValueError('Exported request raw SHA differs')
    model_path=Path(args.request_model_config).resolve()
    if sha(model_path)!=args.request_model_config_sha256:raise ValueError('Original Qwen MODEL_CONFIG SHA differs')
    proposal_path=request_path.parent/'RESULT.json';proposal_raw=proposal_path.read_bytes();proposal=json.loads(proposal_raw)
    if (proposal.get('version'),proposal.get('phase'),proposal.get('status'),proposal.get('provider_mode'))!=('native_goal_learning_cycle_v1','propose','AWAITING_EXTERNAL_VALIDATION','ORIGINAL_MEASURED_QWEN'):
        raise ValueError('No completed actual-Qwen validation-request proposal')
    state_path=workspace/'skills/goal-component-feedback/CURRENT_COMPONENT_STATE.json'
    if sha(state_path)!=args.current_component_state_sha256:raise ValueError('Actual current component state differs')
    original_path=workspace/'reports/VALIDATION_REQUEST.json';original_raw=original_path.read_bytes()
    request=json.loads(raw)
    if (proposal['request']['export_sha256']!=args.validation_request_sha or
        hashlib.sha256(original_raw).hexdigest()!=proposal['request']['original_sha256'] or json.loads(original_raw)!=request):
        raise ValueError('Original model-written request and exported request differ')
    if proposal['decision']['current_component_state_sha256']!=args.current_component_state_sha256:
        raise ValueError('Proposal state provenance differs')
    if sha(workspace/'SKILLRUNTIME.md')!=proposal['decision']['registry_sha256']:
        raise ValueError('Proposed workspace registry changed')
    state=json.loads(state_path.read_text())
    for rel,expected in state['registry_sha256'].items():
        if rel not in ('SKILLRUNTIME.md','TARGETS.md') or sha(workspace/rel)!=expected:raise ValueError('Current native registry facts changed')
    from fastrsi.native_validation_request_v1 import validate_request
    request=validate_request(request,workspace=workspace,expected_model_config_sha256=args.request_model_config_sha256,
        current_component_state_sha256=args.current_component_state_sha256)
    declaration=request['candidate_declaration'];transform=declaration['transform']
    for key,reference in [('candidate',declaration['source_artifact']),('source_image',transform['source_rgb']),('calibration',transform['calibration'])]:
        if plan['bindings'][plan[key]]!=reference['sha256']:raise ValueError('Requested source differs from executable plan')
    if transform['version']!='dual_anchor_v1' or any(plan[key]!=transform[key] for key in ('source_rois_xyxy','anchor_plan','max_offset_m')):
        raise ValueError('Requested geometry differs from executable plan')
    return dict(mode='ACTUAL_MODEL_EXPORTED_VALIDATION_REQUEST',development_seeds=request['development_seeds'],
        model_requested=True,request_path=str(request_path),request_sha256=args.validation_request_sha,
        original_request_sha256=hashlib.sha256(original_raw).hexdigest(),proposal_result_path=str(proposal_path),
        proposal_result_sha256=hashlib.sha256(proposal_raw).hexdigest(),request_workspace=str(workspace),
        current_component_state_sha256=args.current_component_state_sha256,original_qwen_model_config_path=str(model_path),
        original_qwen_model_config_sha256=args.request_model_config_sha256,
        selected_registry_sha256=sha(workspace/'SKILLRUNTIME.md'),full_agent_chain=False,learning_gain_claimed=False)


def command(argv,timeout=45):
    start=time.monotonic();p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
    result=dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,seconds=time.monotonic()-start)
    if p.returncode:raise RuntimeError(result)
    return result


def freeze(args):
    project=args.project.resolve();plan=args.plan_dir.resolve()
    if (plan/'MANIFEST.json').exists():raise ValueError('Already frozen; never replace a registered campaign')
    manifest_path=project/COLLECTION/'MANIFEST.json'
    if sha(manifest_path)!=args.collection_manifest_sha256:raise ValueError('Collected development manifest differs')
    reg=json.loads((project/REGISTRATION).read_text())
    if reg['version']!='task64_native_development_sampler_v1' or reg['seeds']!=[640711,640712] or reg['role']!='DEVELOPMENT_INPUT':raise ValueError('Development partition differs')
    collected=json.loads((project/COLLECTION/'RESULT.json').read_text())
    if collected['status']!='COMPLETED' or collected['role']!='DEVELOPMENT_INPUT':raise ValueError('Development collection incomplete')
    for rel,digest in json.loads(manifest_path.read_text()).items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts or sha(project/COLLECTION/rel)!=digest:raise ValueError('Development artifact changed: '+rel)
    plan.mkdir(parents=True,exist_ok=True);(plan/'code').mkdir(exist_ok=False)
    paths=['scripts/'+s for s in SCRIPTS]+['src/fastrsi/'+s for s in MODULES]+['tests/'+s for s in TESTS]+[NATIVE]
    for rel in paths:
        src=project/rel;dst=plan/'code'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        ast.parse(src.read_text(),feature_version=(3,8));shutil.copy2(src,dst)
    old=json.loads((project/SOURCE_PLAN).read_text())
    protocol={k:old[k] for k in ('suite','task_id','max_steps','candidate','source_image','calibration','source_rois_xyxy','anchor_plan','max_offset_m')}
    bindings=dict(old['bindings']);bindings[REGISTRATION]=sha(project/REGISTRATION);bindings[COLLECTION+'/MANIFEST.json']=sha(manifest_path)
    # Bind model-independent public source bytes now. Models are stream-hashed
    # once on the execution host before any policy service is started.
    for rel,digest in bindings.items():
        if sha(project/rel)!=digest:raise ValueError('Frozen source input mismatch: '+rel)
    protocol.update(version='native_development_operation_v1',role='DEVELOPMENT_INPUT',development_seeds=[640711,640712],
        arms=['native_pi05','dual_anchor_v1'],program_steps=253,replan_steps=10,policy_config='pi05_libero',torch_seed=7,
        registration=REGISTRATION,collection=COLLECTION,bindings=bindings,
        model_files_sha256=MODEL_FILES,policy_image=POLICY_IMAGE,target_image=CPU_IMAGE,
        resources=dict(host='flowwam-gpu-187',gpus=[0,1],policy_cpus=8,policy_memory='64g',target_cpus=8,target_memory='24g',ports=[49871,49872,49873,49874]),
        limits=dict(max_action_episodes=4,max_policy_steps_each=400,max_settling_steps_each=10,policy_startup_seconds=420,episode_seconds=1500,campaign_seconds=4500),
        source_scope='Source31 manual dual-anchor operation; not a trained selector, full Agent, independent evaluation, or official fixed50 score',
        comparison='Same original development initial inputs; fresh seeded original pi05 against source253 program then same policy fallback within400; no hidden object state in control',
        failure_policy='Preserve failures and all paid work. No automatic episode retry or seed substitution. A native task failure is an outcome; infrastructure or strict-input mismatch stops subsequent dispatch.',
        cost_relationship='Phase costs are nested; GPU service residency is allocation time, distinct from measured inference; do not sum overlapping two-arm wall times.')
    save(plan/'EXECUTION_PLAN.json',protocol)
    shutil.copy2(project/'runs/pilot_20261001/LAUNCH_pi05_v2.json',plan/'POLICY_LAUNCH_TEMPLATE.json')
    save(plan/'FREEZE_RECEIPT.json',dict(status='FROZEN_NOT_DISPATCHED',source_files={p:sha(project/p) for p in paths},collection_manifest_sha256=args.collection_manifest_sha256,default_pointer_sha256=sha(project/'CURRENT_NATIVE_ENTRY.json')))
    preflight='''import ast,importlib.util,json,sys,time,unittest\nfrom pathlib import Path\nroot=Path(__file__).resolve().parent/'code'\nsys.path[:0]=[str(root/'scripts'),str(root/'src'),str(root/'tests')]\nstart=time.perf_counter()\nfor path in root.rglob('*.py'):ast.parse(path.read_text(),feature_version=(3,8))\nimport probe_native_development_operation_v1\nspec=importlib.util.spec_from_file_location('actual_native_import','/core/PhyAgentOS/runtime/targets/remote/libero/server.py')\nmodule=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\nsuite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName(name) for name in ('test_native_sampler_draw_restore_v1','test_native_development_orchestrator_v1')])\nresult=unittest.TextTestRunner(verbosity=2).run(suite)\nrecord=dict(status='PASSED' if result.wasSuccessful() and not result.skipped else 'FAILED',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),seconds=time.perf_counter()-start,actual_native_import=True,model_calls=0,resets=0,policy_actions=0,python=sys.version)\nPath('/out/CHECK.json').write_text(json.dumps(record,indent=2)+'\\n')\nraise SystemExit(0 if record['status']=='PASSED' else 1)\n'''
    (plan/'CPU_PREFLIGHT.py').write_text(preflight)
    manifests={str(p.relative_to(plan)):sha(p) for p in sorted(plan.rglob('*')) if p.is_file() and p.name!='MANIFEST.json'}
    save(plan/'MANIFEST.json',manifests)
    print(json.dumps(dict(status='FROZEN',plan=str(plan),manifest_sha256=sha(plan/'MANIFEST.json'),plan_sha256=sha(plan/'EXECUTION_PLAN.json'))))


class Dispatch:
    def __init__(self,args,mode):
        self.args=args;self.project=args.project.resolve();self.plan=args.plan_dir.resolve();self.mode=mode;self.hash=args.manifest_sha256
        self.out=self.plan/('cpu_preflight' if mode=='cpu-preflight' else 'outputs')
        if self.project!=REMOTE_PROJECT:raise ValueError('Dispatch requires registered remote project host')
        if sha(self.plan/'MANIFEST.json')!=self.hash:raise ValueError('Frozen manifest differs')
        for rel,digest in json.loads((self.plan/'MANIFEST.json').read_text()).items():
            if Path(rel).is_absolute() or '..' in Path(rel).parts or sha(self.plan/rel)!=digest:raise ValueError('Frozen input changed: '+rel)
        self.protocol=json.loads((self.plan/'EXECUTION_PLAN.json').read_text())
        for rel,digest in self.protocol['bindings'].items():
            if sha(self.project/rel)!=digest:raise ValueError('Live source binding differs: '+rel)
        self.lock=(self.plan/'dispatch.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.out.mkdir(exist_ok=False);self.started=time.monotonic();self.records=[]
        self.deps=self.project.parent/'frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1/dependencies'
        self.weights=self.deps.parent/'weights/pi05_libero_pytorch_fp32'
        self.base='/project/'+str(self.plan.relative_to(self.project))
        self.state=dict(status='CHECKING',mode=mode,manifest_sha256=self.hash,plan_sha256=sha(self.plan/'EXECUTION_PLAN.json'),role='DEVELOPMENT_INPUT',model_calls=0 if mode=='cpu-preflight' else 'RECORDED_PER_ARM',arms=[],started_at=time.time())
        self.deadline=self.started+(300 if mode=='cpu-preflight' else 4500)
        def interrupt(signum,frame):raise KeyboardInterrupt('Owned dispatch signal '+str(signum))
        signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)

    def log(self):save(self.out/'STATUS.json',self.state)
    def container(self,name):
        p=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=25)
        if p.returncode:return None
        obj=json.loads(p.stdout)[0]
        if (obj['Config'].get('Labels') or {}).get('fastrsi.dispatch_manifest')!=self.hash:raise ValueError('Container ownership mismatch; never stop foreign work')
        return obj

    def launch(self,role,folder,args):
        name=args[args.index('--name')+1]
        if subprocess.run(['docker','inspect',name],capture_output=True,timeout=25).returncode==0:raise ValueError('Container already exists; no repeat dispatch')
        args[2:2]=['--label','fastrsi.dispatch_manifest='+self.hash]
        row=dict(role=role,name=name,folder=str(folder),argv=args,started_at=time.time(),status='LAUNCH_ATTEMPT',launch_monotonic=time.monotonic())
        self.records.append(row)
        try:
            receipt=command(args,60);cid=receipt['stdout'].strip();row.update(container_id=cid,launch_seconds=receipt['seconds'])
            obj=self.container(name)
            if obj is None or obj['Id']!=cid:raise ValueError('Launch ID does not match owned container')
            row.update(status='RUNNING',container_started_at=obj['State']['StartedAt'])
        finally:save(folder/('LAUNCH_'+role+'.json'),row)
        return row

    def stop(self,row):
        if row.get('stopped'):return
        tick=time.monotonic()
        try:
            obj=self.container(row['name'])
            if obj is not None:
                cid=obj['Id'];row['container_id']=cid
                if obj['State']['Running']:command(['docker','stop','-t','10',cid],40)
                logs=subprocess.run(['docker','logs',cid],capture_output=True,timeout=30)
                folder=Path(row['folder']);(folder/(row['role']+'.stdout.log')).write_bytes(logs.stdout);(folder/(row['role']+'.stderr.log')).write_bytes(logs.stderr)
                after=self.container(cid);row.update(container_finished_at=after['State']['FinishedAt'],exit_code=after['State']['ExitCode'],running=after['State']['Running'])
                if row['running']:raise RuntimeError('Owned container remains running')
                row['lifecycle_seconds_inclusive']=time.monotonic()-row['launch_monotonic']
                row['timestamp_precision']='Original Docker RFC3339 strings preserved; duration truncates each endpoint to microseconds for Python3.8'
                row['container_running_seconds']=(parse_docker_timestamp(after['State']['FinishedAt'])-parse_docker_timestamp(obj['State']['StartedAt'])).total_seconds()
            row['stopped']=True
        finally:
            row['stop_seconds']=time.monotonic()-tick
            save(Path(row['folder'])/('STOP_'+row['role']+'.json'),row)

    def cpu_args(self,name,folder,network):
        return ['docker','run','-d','--name',name,'--network',network,'--cpus','8','--memory','24g','--pids-limit','1024','--shm-size','2g',
            '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/workspace/openpi/third_party/libero:'+self.base+'/code/scripts:'+self.base+'/code/src:/core',
            '-e','LIBERO_CONFIG_PATH=/libero-config','-e','MUJOCO_GL=egl','-e','LIBGL_ALWAYS_SOFTWARE=1',
            '-v',str(self.project)+':/project:ro','-v',str(folder)+':/out:rw','-v',str(self.deps/'openpi')+':/workspace/openpi:ro',
            '-v',str(self.deps/'PhyAgentOS-core')+':/core:ro','-v',str(self.project/'configs/libero')+':/libero-config:ro',CPU_IMAGE]

    def wait_exit(self,row,limit):
        end=min(self.deadline,time.monotonic()+limit)
        while time.monotonic()<end:
            obj=self.container(row['container_id'])
            if not obj['State']['Running']:
                if obj['State']['ExitCode']!=0:raise RuntimeError('Process failed: '+row['name'])
                return
            time.sleep(2)
        raise TimeoutError('Bounded process wait expired; no restart')

    def resource_gate(self,folder,gpu=False):
        inv=command(['docker','ps','--format','{{.ID}} {{.Names}}'])
        memory={k.split(':')[0]:int(k.split()[1])*1024 for k in Path('/proc/meminfo').read_text().splitlines() if k.startswith(('MemAvailable:','MemTotal:'))}
        record=dict(time=time.time(),containers=inv,cpu_count=os.cpu_count(),memory=memory,disk_free_bytes=shutil.disk_usage(self.project).free)
        for image in ([CPU_IMAGE,POLICY_IMAGE] if gpu else [CPU_IMAGE]):
            if command(['docker','image','inspect',image,'--format','{{.Id}}'])['stdout'].strip()!=image:raise ValueError('Image identity differs')
        if gpu:
            # The registered pilot mounts the driver globally; require no other
            # container or GPU holder instead of claiming device isolation.
            record['hcu']=command(['bash','-lc','source /opt/dtk/env.sh && source /opt/hyhal/env.sh && hy-smi --showuse --showmemuse'])
            record['holders']=[]
            for device in ['/dev/kfd','/dev/mkfd']+[str(p) for p in sorted(Path('/dev/dri').glob('render*'))]:
                res=subprocess.run(['fuser',device],capture_output=True,text=True,timeout=15)
                if res.returncode not in (0,1):raise RuntimeError('Device-holder query failed')
                for token in res.stdout.split():
                    if not token.isdigit():raise ValueError('Unparsed device holder')
                    pid=int(token);proc=Path('/proc')/str(pid)
                    comm=(proc/'comm').read_text().strip();argv=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode();exe=os.readlink(proc/'exe')
                    allowed=device=='/dev/mkfd' and pid==2478 and comm=='hymgr' and argv.strip()=='/usr/local/hyhal/bin/hymgr' and Path(exe).name=='hymgr'
                    record['holders'].append(dict(device=device,pid=pid,comm=comm,argv=argv,exe=exe,allowed_system_daemon=allowed))
            mem=re.findall(r'HCU\[(\d+)\].*?HCU memory use \(%\):\s*([0-9.]+)',record['hcu']['stdout'])
            use=re.findall(r'HCU\[(\d+)\].*?HCU use \(%\):\s*([0-9.]+)',record['hcu']['stdout'])
            record['parsed_hcu_memory']=mem;record['parsed_hcu_use']=use
            for port in self.protocol['resources']['ports']:
                with socket.socket() as sock:
                    if sock.connect_ex(('127.0.0.1',port))==0:raise ValueError('Reserved policy port occupied')
            save(folder/'RESOURCE_RECEIPT.json',record)
            if len(mem)!=8 or len(use)!=8 or any(float(v)!=0 for _,v in mem+use):raise ValueError('GPU resource state busy or unparsed')
            if any(not x['allowed_system_daemon'] for x in record['holders']):raise ValueError('Unrelated GPU device holder; no preemption')
        else:save(folder/'RESOURCE_RECEIPT.json',record)
        if inv['stdout'].strip():raise ValueError('Existing live container; inspect ownership, do not preempt')
        required=(180 if gpu else 26)*1024**3
        if memory['MemAvailable']<required or (os.cpu_count() or 0)<(32 if gpu else 8):raise ValueError('Insufficient CPU/memory headroom')
        if record['disk_free_bytes']<4*1024**3:raise ValueError('Insufficient scratch storage')

    def preflight(self):
        self.resource_gate(self.out)
        args=self.cpu_args('yuchen-fastrsi-'+NAME+'-cpu',self.out,'none')+['timeout','180','python',self.base+'/CPU_PREFLIGHT.py']
        row=self.launch('cpu',self.out,args);self.wait_exit(row,210);self.stop(row)
        check=json.loads((self.out/'CHECK.json').read_text())
        if check['status']!='PASSED':raise ValueError('Actual CPU import/tests failed')
        self.state.update(status='PASSED',check=check,container_id=row['container_id'])

    def policy_args(self,seed,arm,index,folder,port):
        args=json.loads((self.plan/'POLICY_LAUNCH_TEMPLATE.json').read_text())['argv']
        old_cache=str(self.project/'models/openpi_data');old_run=str(self.project/'runs/pilot_20261001')
        cache=folder/'tokenizer_cache';shutil.copytree(self.project/'models/openpi_data',cache)
        save(folder/'TOKENIZER_CACHE.json',dict(files={str(p.relative_to(cache)):sha(p) for p in sorted(cache.rglob('*')) if p.is_file()},bytes=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())))
        args=[v.replace(old_cache,str(cache)).replace(old_run,str(folder)) for v in args]
        args[args.index('--name')+1]='yuchen-fastrsi-'+NAME+'-'+str(seed)+'-'+arm+'-pi05'
        args=[('HIP_VISIBLE_DEVICES='+str(index)) if v.startswith('HIP_VISIBLE_DEVICES=') else ('CUDA_VISIBLE_DEVICES='+str(index)) if v.startswith('CUDA_VISIBLE_DEVICES=') else v for v in args]
        args[-1]=args[-1].replace('--port 49500','--port '+str(port))
        if '--torch-seed 7' not in args[-1] or '--disable-torch-compile' not in args[-1] or '--pytorch-device cuda:0' not in args[-1]:raise ValueError('Original policy command changed')
        if POLICY_IMAGE not in args:raise ValueError('Wrong policy image')
        return args

    def health(self,row,port):
        tick=time.monotonic();end=min(self.deadline,tick+420);opener=build_opener(ProxyHandler({}))
        while time.monotonic()<end:
            if not self.container(row['container_id'])['State']['Running']:raise RuntimeError('Policy service exited before readiness')
            try:
                with opener.open('http://127.0.0.1:'+str(port)+'/healthz',timeout=2) as response:
                    if response.status==200:
                        row['readiness_wait_seconds']=time.monotonic()-tick;row['ready_at']=time.time();return
            except Exception:pass
            time.sleep(1)
        raise TimeoutError('Policy startup expired')

    def validated_execution_binding(self):
        if self.args.operation_diagnostic_only:
            return resolve_execution_request(self.args,self.protocol)
        # The typed validator decodes a fixed reference image. Use the actual
        # CPU image whose dependencies were checked, not an assumed host cv2.
        folder=self.out/'request_validation';folder.mkdir()
        def mapped(path):
            if path is None:raise ValueError('Missing model request input')
            return '/project/'+str(Path(path).resolve().relative_to(self.project))
        cmd=['timeout','90','python',self.base+'/code/scripts/orchestrate_native_development_operation_v1.py','validate-request',
            '--project','/project','--plan-dir',self.base,'--output','/out/VALIDATED_BINDING.json',
            '--validation-request',mapped(self.args.validation_request),'--validation-request-sha',str(self.args.validation_request_sha),
            '--request-workspace',mapped(self.args.request_workspace),
            '--current-component-state-sha256',str(self.args.current_component_state_sha256),
            '--request-model-config',mapped(self.args.request_model_config),
            '--request-model-config-sha256',str(self.args.request_model_config_sha256)]
        row=self.launch('request_validation',folder,self.cpu_args('yuchen-fastrsi-'+NAME+'-request',folder,'none')+cmd)
        self.wait_exit(row,105);self.stop(row)
        binding=json.loads((folder/'VALIDATED_BINDING.json').read_text())
        if (binding['mode']!='ACTUAL_MODEL_EXPORTED_VALIDATION_REQUEST' or
                binding['request_sha256']!=self.args.validation_request_sha or
                binding['current_component_state_sha256']!=self.args.current_component_state_sha256 or
                binding['original_qwen_model_config_sha256']!=self.args.request_model_config_sha256):
            raise ValueError('CPU request-validator output identity differs')
        if sha(self.args.validation_request)!=self.args.validation_request_sha:raise ValueError('Request changed after validation')
        if binding['development_seeds']!=json.loads(Path(self.args.validation_request).read_text())['development_seeds']:
            raise ValueError('Requested development sequence differs')
        binding['validator_container_id']=row['container_id']
        binding['validator_lifecycle']=row
        return binding

    def validate_pair(self,values,pair):
        """Optional stricter diagnostic-specific check after real paired outcomes."""
        return None

    def execute(self):
        cpu=json.loads((self.plan/'cpu_preflight/STATUS.json').read_text())
        if cpu['status']!='PASSED' or cpu['manifest_sha256']!=self.hash:raise ValueError('Exact frozen CPU preflight required')
        if sha(self.project/'CURRENT_NATIVE_ENTRY.json')!=json.loads((self.plan/'FREEZE_RECEIPT.json').read_text())['default_pointer_sha256']:raise ValueError('Default pointer differs before dispatch')
        sys.path.insert(0,str(self.plan/'code/src'))
        binding=self.validated_execution_binding()
        save(self.out/'REQUEST_EXECUTION_BINDING.json',binding)
        self.state['request_execution_binding']=binding
        tick=time.monotonic();model={}
        for rel,digest in MODEL_FILES.items():
            path=self.weights/rel;actual=sha(path);model[rel]=dict(expected=digest,actual=actual,size=path.stat().st_size)
            if actual!=digest:raise ValueError('Frozen pi05 model bytes differ: '+rel)
        save(self.out/'MODEL_BINDING.json',dict(files=model,stream_hash_seconds=time.monotonic()-tick))
        for seed in binding['development_seeds']:
            round_idx=self.protocol['development_seeds'].index(seed)
            pair=self.out/('seed_'+str(seed));pair.mkdir();pair_tick=time.monotonic()
            self.resource_gate(pair,gpu=True)
            policies={};targets={};rows={}
            try:
                arms=self.protocol['arms']
                for index,arm in enumerate(arms):
                    folder=pair/arm;folder.mkdir();port=self.protocol['resources']['ports'][round_idx*len(arms)+index]
                    gpu=self.protocol['resources']['gpus'][index]
                    row=dict(seed=seed,arm=arm,port=port,gpu=gpu,status='STARTING',folder=str(folder),started_at=time.time())
                    rows[arm]=row;self.state['arms'].append(row)
                    policies[arm]=self.launch('pi05',folder,self.policy_args(seed,arm,gpu,folder,port))
                for arm,row in rows.items():self.health(policies[arm],row['port'])
                for arm,row in rows.items():
                    folder=Path(row['folder'])
                    probe_script=self.protocol.get('probe_script','probe_native_development_operation_v1.py')
                    if Path(probe_script).name!=probe_script:raise ValueError('Probe must be a frozen script basename')
                    args=self.cpu_args('yuchen-fastrsi-'+NAME+'-'+str(seed)+'-'+arm+'-target',folder,'host')+['timeout','1500','python',self.base+'/code/scripts/'+probe_script,
                        '--plan',self.base+'/EXECUTION_PLAN.json','--project','/project','--seed',str(seed),'--arm',arm,'--output','/out/result','--policy-endpoint','ws://127.0.0.1:'+str(row['port'])]+self.protocol.get('probe_extra_args',[])
                    targets[arm]=self.launch('target',folder,args);row['status']='RUNNING'
                self.state['status']='RUNNING';self.log()
                end=min(self.deadline,time.monotonic()+1530)
                while time.monotonic()<end:
                    all_done=True
                    for arm,target in targets.items():
                        obj=self.container(target['container_id'])
                        if obj['State']['Running']:all_done=False
                        elif not rows[arm].get('finished_at'):
                            rows[arm]['target_terminal_observed_at']=time.time()
                            self.stop(target);self.stop(policies[arm])
                            rows[arm]['finished_at']=time.time()
                            rows[arm]['full_wall_seconds']=rows[arm]['finished_at']-rows[arm]['started_at']
                            if obj['State']['ExitCode']!=0:raise RuntimeError('Probe process failed: '+arm)
                    if all_done:break
                    time.sleep(2)
                else:raise TimeoutError('Paired development execution timeout')
                values={arm:json.loads((Path(row['folder'])/'result/RESULT.json').read_text()) for arm,row in rows.items()}
                for arm,result in values.items():
                    if result['status']!='COMPLETED':raise ValueError('Probe failed: '+arm)
                if len(arms)!=2:raise ValueError('Exactly two matched operations required')
                first,second=[values[a] for a in arms]
                for field in ('pre_settle_state','post_settle_state','public_rgb','public_robot8','xml_sha256'):
                    if first['initial_input_binding'][field]!=second['initial_input_binding'][field]:raise ValueError('Paired initial inputs differ: '+field)
                self.validate_pair(values,pair)
                for arm,row in rows.items():
                    row.update(status='COMPLETED',native_success=values[arm]['native_success'],physical_steps=values[arm]['physical_steps'])
            finally:
                cleanup_errors=[]
                for record in list(targets.values())+list(policies.values()):
                    try:self.stop(record)
                    except BaseException:cleanup_errors.append(dict(name=record['name'],error=traceback.format_exc()))
                for arm,row in rows.items():
                    folder=Path(row['folder']);raw=folder/'result/RESULT.json'
                    probe=json.loads(raw.read_text()) if raw.exists() else None
                    policy=policies.get(arm,{})
                    ready=policy.get('ready_at');launch_time=policy.get('started_at')
                    if not row.get('finished_at'):
                        row['finished_at']=time.time();row['full_wall_seconds']=row['finished_at']-row['started_at']
                    receipt=dict(development_seed=seed,seed=seed,arm=arm,manifest_sha256=self.hash,plan_sha256=sha(self.plan/'EXECUTION_PLAN.json'),
                        raw_probe=dict(path='result/RESULT.json',sha256=sha(raw)) if raw.exists() else None,
                        model_startup_seconds=(ready-launch_time) if ready is not None and launch_time is not None else None,
                        model_stop_seconds=policy.get('stop_seconds'),model_residency_seconds=policy.get('container_running_seconds'),
                        full_wall_seconds=row['full_wall_seconds'],full_agent_chain=False,learning_gain_claimed=False,cleanup_errors=cleanup_errors,
                        request_execution_binding_sha256=sha(self.out/'REQUEST_EXECUTION_BINDING.json'),request_sha256=binding.get('request_sha256'),model_requested=binding['model_requested'],
                        status='COMPLETED' if probe is not None and probe.get('status')=='COMPLETED' and row.get('status')=='COMPLETED' and not cleanup_errors else 'FAILED',probe_result_path=str(raw.relative_to(self.project)),probe_result_sha256=sha(raw) if raw.exists() else None,
                        probe_costs=probe.get('costs') if probe else None,probe_seconds_inclusive=probe.get('seconds') if probe else None,
                        native_success=probe.get('native_success') if probe else None,physical_steps=probe.get('physical_steps') if probe else None,
                        policy_lifecycle=policies.get(arm),target_lifecycle=targets.get(arm),arm_wall_seconds_inclusive=row['full_wall_seconds'],
                        cost_relationship='Probe component costs are nested in probe seconds. Policy allocation overlaps target execution and peer arm; startup/readiness/stop/residency retain separate raw scopes, never summed as method walltime.')
                    save(folder/'ARM_RESULT.json',receipt)
                save(pair/'PAIR_COST.json',dict(seed=seed,pair_wall_seconds_inclusive=time.monotonic()-pair_tick,parallel_arms=True))
                self.log()
                if cleanup_errors:raise RuntimeError('Paired cleanup failed; do not continue to next seed')
        self.state['status']='COMPLETED'

    def run(self):
        try:
            if self.mode=='cpu-preflight':self.preflight()
            else:self.execute()
        except BaseException:self.state.update(status='FAILED',error=traceback.format_exc())
        finally:
            failures=[]
            for row in self.records:
                try:self.stop(row)
                except BaseException:failures.append(dict(name=row['name'],error=traceback.format_exc()))
            if failures:self.state.update(status='CLEANUP_FAILED',cleanup_failures=failures)
            self.state.update(controller_wall_seconds_inclusive=time.monotonic()-self.started,finished_at=time.time(),containers=self.records)
            self.state['cost_relationship']='Controller wall time includes all sequential seeds and parallel arm overlap. Do not add nested or overlapping costs.'
            self.log()
            save(self.out/'OUTPUT_MANIFEST.json',{str(p.relative_to(self.out)):sha(p) for p in sorted(self.out.rglob('*')) if p.is_file() and p.name!='OUTPUT_MANIFEST.json'})
            print(json.dumps(self.state),flush=True)
        if self.state['status'] not in ('PASSED','COMPLETED'):raise SystemExit(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','cpu-preflight','execute','validate-request'])
    p.add_argument('--project',type=Path,required=True);p.add_argument('--plan-dir',type=Path,required=True)
    p.add_argument('--collection-manifest-sha256');p.add_argument('--manifest-sha256')
    p.add_argument('--validation-request',type=Path);p.add_argument('--validation-request-sha')
    p.add_argument('--request-workspace',type=Path);p.add_argument('--current-component-state-sha256')
    p.add_argument('--request-model-config',type=Path);p.add_argument('--request-model-config-sha256')
    p.add_argument('--operation-diagnostic-only',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.mode=='validate-request':
        if args.output is None:p.error('validation output path required')
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
        binding=resolve_execution_request(args,json.loads((args.plan_dir/'EXECUTION_PLAN.json').read_text()))
        if args.output.exists():raise ValueError('Refuse validator output overwrite')
        save(args.output,binding)
    elif args.mode=='freeze':
        if not args.collection_manifest_sha256:p.error('freeze requires final development collection manifest SHA')
        freeze(args)
    else:
        if not args.manifest_sha256:p.error('dispatch requires exact frozen manifest SHA')
        Dispatch(args,args.mode).run()

if __name__=='__main__':main()
