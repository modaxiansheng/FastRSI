"""Matched full native Agent comparison; only the policy action tensors differ."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import orchestrate_native_sampler_agent_arm_v1 as native

P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')

def prepare(project,campaign,seed,slot,adapter):
    project,campaign=Path(project),Path(campaign)
    bindings=native.read(project/'runs/native_successor_operation_B_640715_v3_20261007/SOURCE_BINDINGS.json')
    collection=project/'planning/bowl_action_adaptation_eval6_v1_20261009'
    for key,path in [('sampler_registration',collection/'REGISTRATION.json'),('sampler_manifest',collection/'outputs/samples/MANIFEST.json')]:
        bindings[key]=dict(path=str(path),sha256=native.sha(path))
    bindings.update(mode='evaluation',draw_seed=seed)
    old=Path(bindings['runtime_code_manifest']['path']).parent/'code'
    runtime=campaign.parent/(campaign.name+'_runtime_base')
    shutil.copytree(old,runtime/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for subtree in ('scripts','src/fastrsi'):
        for path in (project/subtree).rglob('*.py'):
            dest=runtime/'code'/path.relative_to(project)
            if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    # Extend only the partition allowlist, keeping old episode and Agent behavior.
    shutil.copyfile(project/'src/fastrsi/native_sampler_restore_v1.py',runtime/'code/src/fastrsi/native_sampler_restore_v1.py')
    native.save(runtime/'CODE_MANIFEST.json',native.manifest(runtime/'code'))
    bindings['runtime_code_manifest']=dict(path=str(runtime/'CODE_MANIFEST.json'),sha256=native.sha(runtime/'CODE_MANIFEST.json'))
    assert slot in range(4);pair=[slot*2,slot*2+1];buses=['09','36','55','77','85','b5','d5','f5'];port=54500+100*slot
    bindings['resources']=dict(gpus=pair,device_map={str(i):dict(pci_bus='0000:'+buses[i]+':00.0',render_node='/dev/dri/renderD'+str(128+i)) for i in pair},ports=dict(policy=port,target=port+2,qwen=port+10,verifier=port+20))
    source=campaign.parent/(campaign.name+'_source_bindings.json');assert not source.exists();native.save(source,bindings)
    native.prepare(project,campaign,source)
    for name in ('serve_action_adapted_pi05_v1.py','orchestrate_anchored_head_agent_v1.py'):
        shutil.copyfile(project/'scripts'/name,campaign/'code/scripts'/name)
    cfg=dict(enabled=bool(adapter),comparison='Fixed anchored head50 versus original; full Agent and fixed readable experience; development regression',persistent_updates=False)
    if adapter:
        result=native.read(adapter.parent/'RESULT.json');assert result['status']=='ANCHORED_HEAD_OFFLINE_PASSED' and result['training_steps']==50 and result['gates']['passed'] and result['trainable_elements']==7175
        assert native.sha(adapter)==result['adapter']['sha256'];cfg.update(path='/project/'+adapter.relative_to(project).as_posix(),sha256=result['adapter']['sha256'],training_result_sha256=native.sha(adapter.parent/'RESULT.json'))
    native.save(campaign/'ACTION_ADAPTER.json',cfg)
    reg=native.read(campaign/'REGISTRATION.json');reg.update(action_adaptation=cfg,runner_timeout_seconds=1500,service_timeout_seconds=1800,controller_timeout_seconds=1800,
        scope='Head50 development regression on prior exposed native draws; not independent generalization or official LIBERO10 score')
    native.save(campaign/'CODE_MANIFEST.json',native.manifest(campaign/'code'));reg['code_manifest_sha256']=native.sha(campaign/'CODE_MANIFEST.json');native.save(campaign/'REGISTRATION.json',reg)
    inputs=native.manifest(campaign);inputs.pop('MANIFEST.json',None);native.save(campaign/'MANIFEST.json',inputs)
    return dict(campaign=str(campaign),manifest_sha256=native.sha(campaign/'MANIFEST.json'),seed=seed,adapter_enabled=bool(adapter),slot=slot)

original_launch=native.launch_argv

def launch(project,campaign,role):
    argv=original_launch(project,campaign,role);campaign=Path(campaign);base='/project/'+campaign.relative_to(project).as_posix()
    if role=='cpu':argv=[s.replace('orchestrate_native_sampler_agent_arm_v1.py','orchestrate_anchored_head_agent_v1.py') for s in argv]
    if role=='pi05':
        argv=[s.replace('python -m PhyAgentOS.runtime.policy.openpi.native_openpi_server','python '+base+'/code/scripts/serve_action_adapted_pi05_v1.py') for s in argv]
        idx=next(i for i,s in enumerate(argv) if s.startswith('sha256:'))
        cfg=native.read(campaign/'ACTION_ADAPTER.json');extra=['-v',str(project)+':/project:ro','-e','FASTRSI_POLICY_LOAD_RECEIPT='+base+'/stream/POLICY_LOAD.json']
        if cfg['enabled']:extra+=['-e','FASTRSI_ACTION_ADAPTER='+cfg['path'],'-e','FASTRSI_ACTION_ADAPTER_SHA256='+cfg['sha256']]
        argv[idx:idx]=extra
    return argv

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,default=P);ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--prepare',action='store_true');ap.add_argument('--cpu-composition',action='store_true');ap.add_argument('--execute',action='store_true')
    ap.add_argument('--seed',type=int);ap.add_argument('--slot',type=int);ap.add_argument('--adapter',type=Path)
    ap.add_argument('--manifest-sha256');ap.add_argument('--allow-real-physical-calls',action='store_true')
    a=ap.parse_args();native.launch_argv=launch
    if a.prepare:result=prepare(a.project,a.campaign,a.seed,a.slot,a.adapter)
    elif a.cpu_composition:result=native.cpu_composition(a.project,a.campaign,a.manifest_sha256)
    else:
        assert a.execute and a.allow_real_physical_calls
        result=native.execute(a.project,a.campaign,a.manifest_sha256)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
