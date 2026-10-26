"""Bounded CPU integration only; immutable source snapshot and exact-ID cleanup."""
import fcntl,hashlib,json,shutil,time,traceback
import orchestrate_full_os as life

def main():
    name='native_admission_v1_20261002'
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=life.PROJECT/'runs'/name;c.mkdir(exist_ok=False);life.CAMPAIGN=c;life.BASE='/project/runs/'+name
    started=time.time();life.deadline=started+300;s={'status':'PREFLIGHT','started_at':started}
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(life.PROJECT/rel,c/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(c/'CODE_MANIFEST.json',{str(p.relative_to(c/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (c/'code').rglob('*') if p.is_file()})
    life.save(c/'PROTOCOL.json',{'scope':'CPU replay of saved conflict and native Session/evaluation isolation','model_calls':0,'robot_attempts':0,'max_seconds':300,'default_admission':'guarded_v1','native_trigger_changes':False})
    stream=c/'stream';stream.mkdir();life.save(c/'STATUS.json',s)
    try:
        life.gate(stream)
        checks=['test_experience_admission.py','test_evidence_layers.py','test_native_workspace.py','test_native_isolation.py','test_terminal_results.py']
        cmd='source /opt/dtk/env.sh && '+ ' && '.join('/project/envs/os-v1/bin/python '+life.BASE+'/code/tests/'+t for t in checks)
        cmd+=' && /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/probe_experience_admission.py'
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-runner',stream)+['-e','FASTRSI_NATIVE_RUN='+life.BASE+'/stream',life.IMAGE,'timeout','240','bash','-lc',cmd]
        cid=life.launch('runner',argv,stream);s['status']='RUNNING';life.save(c/'STATUS.json',s);life.wait_exit(cid,240)
        assert json.loads((stream/'RESULT.json').read_text())['status']=='PASSED';s['status']='COMPLETED'
    except BaseException:s.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(stream/'STOP_RECEIPT.json',life.stop_all());s.update(finished_at=time.time(),full_wall_seconds=time.time()-started)
        life.save(c/'STATUS.json',s);print(json.dumps(s),flush=True)
    if s['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()
