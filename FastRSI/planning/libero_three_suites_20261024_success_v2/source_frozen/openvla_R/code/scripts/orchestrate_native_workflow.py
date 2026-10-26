"""Bounded next decision diagnostic using the shared service and native Agent."""
import fcntl
import argparse
import hashlib
import json
import shutil
import signal
import time
import traceback
import orchestrate_full_os as life
from native_service_config import configure_qwen,model_env

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='native_workflow_v2_20261002')
    parser.add_argument('--protocol',default='configs/native_workflow_v2.json');args=parser.parse_args()
    name=args.run_id
    assert name.startswith('native_workflow_') and all(c.isalnum() or c=='_' for c in name)
    gate=life.PROJECT/'runs/native_stage1_v1_20261002'
    prior=json.loads((gate/'STATUS.json').read_text())
    assert prior['stage']=='native_batch_end_memory' and 'Interrupted 15' in prior['error']
    assert json.loads((gate/'stream/RESULT.json').read_text())['status']=='COMPLETED'
    accepted=gate/'stream/dev_terminal_t4_i23/accepted_experience'
    source_hashes=json.loads((accepted/'RECEIPT.json').read_text())['hashes']
    assert all(hashlib.sha256((accepted/key).read_bytes()).hexdigest()==sha for key,sha in source_hashes.items())
    assert not (accepted/'memory/MEMORY.md').exists(),'Interrupted probe unexpectedly updated its source'
    data=json.loads((gate/'DECODER_RESULT.json').read_text())
    assert all(x['all_output_tokens_equal'] for x in data['summary'])
    for label in [x['label'] for x in data['summary']]:
        rows=[x for x in data['rows'] if x['label']==label]
        last={v:[x for x in rows if x['variant']==v][-1] for v in ['reference','optimized']}
        assert last['optimized']['receipt']['generation_seconds']<last['reference']['receipt']['generation_seconds']
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    campaign=life.PROJECT/'runs'/name;campaign.mkdir(exist_ok=False)
    life.CAMPAIGN=campaign;life.BASE='/project/runs/'+name;started=time.time();life.deadline=started+1200
    status={'status':'PREFLIGHT','started_at':started}
    def stop(signum,frame):raise RuntimeError('Interrupted '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(life.PROJECT/rel,campaign/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(campaign/'CODE_MANIFEST.json',{str(p.relative_to(campaign/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (campaign/'code').rglob('*') if p.is_file()})
    cfg=json.loads((life.PROJECT/'configs/native_agent_qwen3_v1.json').read_text());cfg['decoder_variant']='optimized'
    cfg['common_generation']['memory']['generated_repetition_penalty']=1.1
    cfg['decoder_gate']=str(gate/'DECODER_RESULT.json');life.save(campaign/'MODEL_CONFIG.json',cfg)
    life.save(campaign/'MODEL_BINDING.json',json.loads((gate/'MODEL_BINDING.json').read_text()))
    protocol=json.loads((life.PROJECT/args.protocol).read_text());life.save(campaign/'PROTOCOL.json',protocol)
    memory_gate=protocol.get('memory_source_gate')
    if memory_gate:
        memory=json.loads((life.PROJECT/memory_gate/'stream/terminal_memory/RESULT.json').read_text())
        assert memory['status']=='PASSED' and memory['source_unchanged_after'] is True
        life.save(campaign/'PASSED_MEMORY_BINDING.json',{'run':memory_gate,'result_sha256':hashlib.sha256((life.PROJECT/memory_gate/'stream/terminal_memory/RESULT.json').read_bytes()).hexdigest()})
    life.save(campaign/'ROSTER.json',json.loads((gate/'ROSTER.json').read_text()))
    life.save(campaign/'MEMORY_SOURCE_BINDING.json',{'source':str(accepted),'hashes':source_hashes,'interrupted_run_status':prior,
        'reason':'Complete only the interrupted memory step in an independent clone; do not rerun the successful robot episode.'})
    stream=campaign/'stream';stream.mkdir();life.save(campaign/'STATUS.json',status)
    try:
        life.gate(stream)
        tests=campaign/'tests';tests.mkdir()
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-tests',tests)+[
            life.IMAGE,'timeout','90','bash','-lc','source /opt/dtk/env.sh && /project/envs/qwen3-v1/bin/python '+life.BASE+'/code/tests/test_memory_repetition.py && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_terminal_results.py && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_native_workspace.py']
        cid=life.launch('tests',argv,tests);life.wait_exit(cid,90)
        argv=json.loads((life.OLD/'LAUNCH_qwen.json').read_text())['argv'];argv[argv.index('--name')+1]='yuchen-fastrsi-'+name+'-qwen'
        argv=[x.replace(str(life.OLD),str(stream)) for x in argv]
        life.launch('qwen',configure_qwen(argv,cfg,life),stream);life.health(49510)
        status.update(status='RUNNING',startup_seconds=time.time()-started);life.save(campaign/'STATUS.json',status)
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-runner',stream)+model_env(cfg)+[
            '-e','FASTRSI_NATIVE_RUN='+life.BASE+'/stream',
            '-e','FASTRSI_TERMINAL_SOURCE=/project/runs/native_stage1_v1_20261002/stream/dev_terminal_t4_i23/accepted_experience',
            life.IMAGE,'timeout','900','bash','-lc',
            'source /opt/dtk/env.sh && '+('' if memory_gate else '/project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/probe_terminal_memory.py && ')+'exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/probe_native_workflow.py']
        cid=life.launch('runner',argv,stream);life.wait_exit(cid,900)
        assert json.loads((stream/'RESULT.json').read_text())['status']=='COMPLETED';status['status']='COMPLETED'
    except BaseException:
        status.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(stream/'STOP_RECEIPT.json',life.stop_all());status.update(finished_at=time.time(),full_wall_seconds=time.time()-started)
        life.save(campaign/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':main()
