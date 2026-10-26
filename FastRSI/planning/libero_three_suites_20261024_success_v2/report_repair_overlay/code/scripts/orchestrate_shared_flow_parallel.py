"""Parallel N/F on disjoint card pairs, preserving original absolute pilot budget."""
import fcntl,hashlib,json,pathlib,shutil,signal,subprocess,sys,time,traceback
import orchestrate_full_os as life

def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tree(p):return {str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file()}
def main():
    name='shared_flow_net_parallel_v2_20261002';p=life.PROJECT
    lock=open(p/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=p/'runs/shared_flow_net_v1_20261002';old=load(previous/'STATUS.json');assert old['status']=='FAILED'
    assert (previous/'USER_ACCELERATION_STOP.json').exists()
    master=p/'runs/representative_native_v1_20261002';state=load(master/'STATE.json');assert state['status']=='AUXILIARY_FAILED'
    deadline=load(previous/'PROTOCOL.json')['absolute_deadline'];assert deadline-time.time()>600
    run=p/'runs'/name;run.mkdir(exist_ok=False);life.CAMPAIGN=run;life.BASE='/project/runs/'+name;life.deadline=deadline
    started=time.time();status={'status':'RUNNING','started_at':started,'groups':[]};workers=[]
    life.save(run/'STATUS.json',status);life.save(run/'MASTER_STATE_BEFORE.json',state)
    state.update(status='AUXILIARY_RUNNING',auxiliary_flow_net_parallel=name,auxiliary_roots_reserved=4+len(list(previous.glob('*/stream/net_*/AGENT_EVENTS.jsonl'))));life.save(master/'STATE.json',state)
    def interrupt(n,f):raise KeyboardInterrupt('Interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        life.gate(run)
        for port in [49600,49602,49610,49620]:
            import socket
            with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0
        for rel in ['scripts','src/fastrsi','tests','configs']:shutil.copytree(p/rel,run/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for rel in ['MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json','PLAN.json','FLOW.json','EXPERIENCE_BINDING.json']:shutil.copyfile(previous/rel,run/rel)
        shutil.copytree(previous/'initial_experience',run/'initial_experience')
        assert tree(run/'initial_experience')==load(run/'EXPERIENCE_BINDING.json')
        tested=p/'runs/terminal_read_v1_20261002/code';source=run/'code'
        # Do not rerun unchanged 37-test regression or pay remote tokenizer downloads again.
        for rel in ['src/fastrsi/session_flow.py','src/fastrsi/paged_reads.py','src/fastrsi/terminal_results.py','scripts/run_native_baseline.py']:
            assert sha(source/rel)==sha(tested/rel)
        life.save(run/'CODE_MANIFEST.json',tree(source))
        protocol={**load(previous/'PROTOCOL.json'),'scope':'Parallel static N/F pilot; user requested idle-card acceleration',
          'absolute_deadline':deadline,'previous_run':str(previous),'previous_full_wall_seconds':old['full_wall_seconds'],
          'resources':{'N':{'host':'187','cards':[0,1]},'F':{'host':'187','cards':[2,3]}},
          'concurrency':'Concurrent independent arms, each same per-service quota, two serial cases; host/shared disk contention possible',
          'timing_limitation':'Single pair of card placements, no crossover. Report raw pilot timings only; not established method speedup.',
          'regression_reuse':'37 completed CPU checks and saved-model replay from terminal_read_v1; identical tested decision code; isolated port-only source relocation checked by reverse mapping',
          'updates':False,'root_reservations':'Four planned roots plus any interrupted original attempts, separately retained; no budget reset',
          'previous_attempted_roots':len(list(previous.glob('*/stream/net_*/AGENT_EVENTS.jsonl')))}
        life.save(run/'PROTOCOL.json',protocol)
        for arm in ['N','F']:
            group=run/arm;shutil.copytree(source,group/'code');changes={}
            if arm=='F':
                for rel in ['scripts/run_native_baseline.py','scripts/qwen_agent_server.py']:
                    f=group/'code'/rel;before=f.read_text();after=before
                    for n in [49500,49502,49510,49520]:after=after.replace(str(n),str(n+100))
                    inverse=after
                    for n in [49500,49502,49510,49520]:inverse=inverse.replace(str(n+100),str(n))
                    assert inverse==before;compile(after,str(f),'exec');f.write_text(after);changes[rel]={'before':sha(source/rel),'after':sha(f)}
            life.save(group/'CODE_MANIFEST.json',tree(group/'code'));life.save(group/'PORT_RELOCATION.json',changes)
        for arm in ['N','F']:
            log=open(run/arm/'controller.log','x')
            proc=subprocess.Popen(['python3',str(source/'scripts/run_shared_flow_worker.py'),name,arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            workers.append((arm,proc));life.save(run/arm/'DISPATCH.json',{'pid':proc.pid,'time':time.time(),'arm':arm})
        while any(proc.poll() is None for arm,proc in workers):
            if time.time()>deadline:raise TimeoutError('Original absolute budget exhausted')
            failed=[arm for arm,proc in workers if proc.poll() not in (None,0)]
            if failed:raise RuntimeError('Worker failed: '+str(failed))
            time.sleep(2)
        assert all(proc.returncode==0 for arm,proc in workers)
        assert tree(run/'initial_experience')==load(run/'EXPERIENCE_BINDING.json')
        status['status']='COMPLETED'
    except BaseException:status.update(status='FAILED',error=traceback.format_exc())
    finally:
        for arm,proc in workers:
            if proc.poll() is None:proc.send_signal(signal.SIGTERM)
        for arm,proc in workers:
            try:proc.wait(timeout=50)
            except subprocess.TimeoutExpired:status['cleanup_attention']='Worker cleanup exceeded 50 seconds; inspect exact owned containers'
            f=run/arm/'STATUS.json'
            if f.exists():status['groups'].append(load(f))
        status.update(finished_at=time.time(),full_wall_seconds=time.time()-started);life.save(run/'STATUS.json',status)
        state.update(status='READY' if status['status']=='COMPLETED' else 'AUXILIARY_FAILED',auxiliary_flow_net_parallel_status=status['status']);life.save(master/'STATE.json',state)
        print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()
