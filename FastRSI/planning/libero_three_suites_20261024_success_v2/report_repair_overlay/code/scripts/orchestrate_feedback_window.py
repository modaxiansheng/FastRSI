"""Three equal-resource full-OS arms on predeclared fresh development cases."""
import fcntl, json, shutil, signal, socket, subprocess, time, traceback
from pathlib import Path
import orchestrate_full_os as life
from orchestrate_placement_recovery import read, tree
from window_budget_continuation import scoped_gate_enabled,recheck_dispatch_deadline

NAME='feedback_window_v1_20261003'

def main():
    global NAME
    import argparse,re
    parser=argparse.ArgumentParser();parser.add_argument("--name",default=NAME);args=parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_]+",args.name):raise ValueError("Invalid run name")
    NAME=args.name
    p=life.PROJECT;run=p/'runs'/NAME
    lock=open(p/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run.mkdir(exist_ok=False);started=time.time();life.deadline=started+2100
    life.CAMPAIGN=run;life.BASE='/project/runs/'+NAME
    state={'status':'RUNNING','started_at':started};life.save(run/'STATUS.json',state)
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(p/rel,run/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(run/'CODE_MANIFEST.json',tree(run/'code'))
    protocol=read(p/'planning'/NAME/'PROTOCOL.json');protocol.setdefault('port_offset',-30000);life.save(run/'PROTOCOL.json',protocol)
    if scoped_gate_enabled(protocol):
        if not protocol.get('task_window_study'):
            raise ValueError('Scoped gate requires explicit task window study')
        life.deadline=min(life.deadline,started+protocol['lifecycle_budget_seconds'])
    arms=list(protocol.get('active_arms',['N','F','R']))
    if arms not in (['N','F','R'],['N','F'],['N','F','R','S']):raise ValueError('Unsupported arm layout')
    if set(arms)!=set(protocol['arms']):raise ValueError('Arm modes and active arms differ')
    if protocol.get('window_learning') and arms!=['N','F','R']:raise ValueError('Legacy learner requires all arms')
    learning=protocol.get('window_learning',False)
    if learning:
        from window_learning_evidence import prepare, collect, digest
        from fastrsi.window_learning import update
        if protocol.get('window_learning_version')==2:
            from window_learning_evidence import prepare_paired as prepare
            from fastrsi.paired_window_learning import update
        t=time.perf_counter();receipt=prepare(p,run/'learning_initial')
        receipt['prepare_seconds']=time.perf_counter()-t;life.save(run/'LEARNING_PREPARE.json',receipt)
        learned=read(run/'learning_initial/LEARNED.json')
    source=p/'runs/shared_flow_net_parallel_v2_20261002'
    bindings=['MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json','FLOW.json','EXPERIENCE_BINDING.json']
    for rel in bindings:shutil.copyfile(source/rel,run/rel)
    def interrupt(n,f):raise KeyboardInterrupt('Interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        cpu_checked=False
        state['allocation_skips']=[]
        for index,c in enumerate(protocol['first_batch']):
            if protocol.get('task_window_study') and (index>0 or scoped_gate_enabled(protocol)):
                from task_window_study import continuation
                receipt=continuation(p,run,protocol,index)
                life.save(run/('CONTINUATION_'+str(index)+'.json'),receipt)
                if receipt['status']=='NOT_DISPATCHED':
                    state['allocation_skips'].append({'index':index,'reason':receipt.get('allocation_reason',receipt.get('decision',{}).get('reason'))})
                    if scoped_gate_enabled(protocol):continue
                    break
            if not cpu_checked:
                cpu=run/'cpu';cpu.mkdir();life.gate(cpu)
                command='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && /project/envs/os-v1/bin/python -m unittest test_short_exit_probe test_window_full_actions test_window_prediction_revision test_report_evidence test_terminal_evidence_view test_task_window test_validation_budget test_paired_window_learning test_window_learning test_feedback_window test_action_window test_placement_recovery test_staged_flow test_terminal_results test_native_isolation test_native_memory_budget -v'
                argv=life.cpu_args('yuchen-fastrsi-'+NAME+'-cpu',cpu)+['-e','FASTRSI_NATIVE_RUN=/tmp/window-cpu',life.IMAGE,'timeout','150','bash','-lc',command]
                cid=life.launch('cpu',argv,cpu);life.wait_exit(cid,160);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());state['cpu']='PASSED'
                cpu_checked=True
            folder=run/('pair'+str(index));folder.mkdir();life.gate(folder)
            if scoped_gate_enabled(protocol):
                receipt=recheck_dispatch_deadline(receipt,life.deadline)
                life.save(run/('CONTINUATION_'+str(index)+'.json'),receipt)
                if receipt['status']=='NOT_DISPATCHED':
                    life.save(folder/'ALLOCATION_CANCELLED.json',receipt)
                    state['allocation_skips'].append({'index':index,'reason':receipt['allocation_reason']})
                    continue
            port_offset=protocol.get('port_offset',0)
            for shift in [port_offset+('N','F','R','S').index(a)*100 for a in arms]:
                for port in (49500,49502,49510,49520):
                    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port+shift))!=0
            for rel in bindings:shutil.copyfile(run/rel,folder/rel)
            shutil.copytree(source/'initial_experience',folder/'initial_experience')
            assert tree(folder/'initial_experience')==read(folder/'EXPERIENCE_BINDING.json')
            life.save(folder/'PLAN.json',{'cases':[c]})
            deadline=min(time.time()+850,life.deadline)
            life.save(folder/'PROTOCOL.json',{'slot':0,'absolute_deadline':deadline,'diagnostic_trace':True,'control_kind':'feedback_window','arm_modes':protocol['arms'],'port_offset':port_offset,'gpu_bases':{'N':0,'F':2,'R':4,'S':6},'terminal_evidence_view':bool(protocol.get('terminal_evidence_view',False))})
            if 'planning_entries' in protocol:
                pair_protocol=read(folder/'PROTOCOL.json')
                pair_protocol['planning_entries']=protocol['planning_entries']
                life.save(folder/'PROTOCOL.json',pair_protocol)
            if protocol.get('task_window_study') and not scoped_gate_enabled(protocol):
                table_path=p/protocol['task_window_artifact']
                import hashlib
                assert hashlib.sha256(table_path.read_bytes()).hexdigest()==protocol['task_window_sha256']
                shutil.copyfile(table_path,folder/'TASK_TABLE.json')
                pair_protocol=read(folder/'PROTOCOL.json')
                pair_protocol['task_window_snapshots']={'F':'TASK_TABLE.json'}
                life.save(folder/'PROTOCOL.json',pair_protocol)
            if learning:
                shutil.copyfile(run/'learning_initial/FROZEN.json',folder/'FROZEN.json')
                life.save(folder/'LEARNED.json',learned)
                pair_protocol=read(folder/'PROTOCOL.json')
                pair_protocol['window_learning_snapshots']={'N':'FROZEN.json','R':'LEARNED.json'}
                life.save(folder/'PROTOCOL.json',pair_protocol)
                before={n:digest(folder/n) for n in ('FROZEN.json','LEARNED.json')}
            workers=[];pair_status={'status':'RUNNING','started_at':time.time(),'case':c}
            life.save(folder/'STATUS.json',pair_status)
            try:
                for arm,arm_shift in [(a,('N','F','R','S').index(a)*100) for a in arms]:
                    shift=port_offset+arm_shift
                    group=folder/arm;shutil.copytree(run/'code',group/'code')
                    for rel in ['scripts/run_native_baseline.py','scripts/qwen_agent_server.py']:
                        path=group/'code'/rel;text=path.read_text()
                        for port in (49500,49502,49510,49520):text=text.replace(str(port),str(port+shift))
                        path.write_text(text)
                    life.save(group/'CODE_MANIFEST.json',tree(group/'code'))
                    with (group/'controller.log').open('x') as log:
                        proc=subprocess.Popen(['python3',str(run/'code/scripts/run_placement_worker.py'),NAME+'/pair'+str(index),arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    workers.append(proc);life.save(group/'DISPATCH.json',{'pid':proc.pid,'started_at':time.time()})
                while any(proc.poll() is None for proc in workers):
                    if time.time()>deadline:raise TimeoutError('Batch deadline')
                    if any(proc.poll() not in (None,0) for proc in workers):raise RuntimeError('Worker failure')
                    time.sleep(2)
                assert all(proc.returncode==0 for proc in workers)
                if learning:
                    assert before=={n:digest(folder/n) for n in before}, 'Session changed fixed policy snapshot'
                    t=time.perf_counter()
                    fact=collect(p,folder/'R',protocol['goal_bindings'][str(c['task_id'])])
                    life.save(folder/'LEARNING_FEEDBACK.json',fact)
                    feedback=[fact]
                    probe_seconds=0
                    if protocol.get('window_learning_version')==2:
                        probe=collect(p,folder/'F',protocol['goal_bindings'][str(c['task_id'])])
                        assert probe['comparison_signature']==fact['comparison_signature']
                        feedback.append(probe);probe_seconds=probe['full_wall_seconds']
                    life.save(folder/'LEARNING_FEEDBACK_BATCH.json',feedback)
                    learned=update(learned,feedback,phase=c['mode'])
                    life.save(folder/'LEARNED_NEXT.json',learned)
                    life.save(folder/'LEARNING_UPDATE.json',{'seconds':time.perf_counter()-t,
                        'before_sha256':before['LEARNED.json'],'after_sha256':digest(folder/'LEARNED_NEXT.json'),
                        'frozen_sha256_unchanged':before['FROZEN.json'],'evidence_id':fact['evidence_id'],
                        'input_records':len(learned['records'])-len(feedback),'output_records':len(learned['records']),
                        'candidate_validation_lifecycle_seconds':probe_seconds,
                        'update_scope':'Own development and declared F candidate validation; N excluded, no narrative promotion' if probe_seconds else 'Own completed development episode only; no N/F feedback or narrative promotion'})
                pair_status['status']='COMPLETED'
            except BaseException:pair_status.update(status='FAILED',error=traceback.format_exc());raise
            finally:
                for proc in workers:
                    if proc.poll() is None:proc.send_signal(signal.SIGTERM)
                for proc in workers:
                    try:proc.wait(timeout=45)
                    except subprocess.TimeoutExpired:pair_status['cleanup_attention']='Inspect worker-owned container IDs'
                pair_status.update(finished_at=time.time(),full_wall_seconds=time.time()-pair_status['started_at']);life.save(folder/'STATUS.json',pair_status)
        state['status']='COMPLETED'
        if not cpu_checked:state['cpu']='NOT_NEEDED_NO_VALIDATION_DISPATCH'
    except BaseException:state.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        life.save(run/'STOP_RECEIPT.json',life.stop_all());state.update(finished_at=time.time(),full_wall_seconds=time.time()-started);life.save(run/'STATUS.json',state)
        print(json.dumps(state),flush=True)

if __name__=='__main__':main()
