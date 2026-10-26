"""One paid native-Session wiring check; no repeated frozen arm or gain claim."""
import argparse, copy, fcntl, hashlib, json, os, shutil, signal, socket
import subprocess, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fastrsi.task_window_snapshot import advance_snapshot
from task_window_study import pair_record
import orchestrate_full_os as life
import orchestrate_structural_window as shared
from orchestrate_resident_full_os import BINDINGS, read, tree, freeze
from orchestrate_resident_services import configure_context, check_ports

NAME = 'task_snapshot_native_v1_20261003'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def provenance(root, pair):
    run = Path(pair['evidence_id']).parts[1]
    proto = 'runs/' + run + '/PROTOCOL.json'
    return {'protocol_group': ('legacy_report_bound_cold_v1' if pair['cost_compatible'] else 'legacy_quality_only_' + run),
            'protocol_sources': {proto: sha(root / proto)},
            'independent_instance_key': f"libero_10/task{pair['task']}/init{pair['init']}"}

def prepare(root, folder):
    """Reuse paid dev feedback, then return a new snapshot and next-case protocol."""
    root, folder = Path(root), Path(folder)
    folder.mkdir(exist_ok=False)
    prior = root / 'planning/task_window_v1_20261003/TASK_TABLE_NEXT.json'
    state = read(prior)
    added = pair_record(root, root / 'runs/feedback_window_v1_20261003/pair0', cost_compatible=False)
    added.update(provenance(root, added))
    case = copy.deepcopy(next(c for c in read(root / 'planning/feedback_window_v1_20261003/PROTOCOL.json')['roster'] if (c['task_id'], c['init_state_id']) == (0, 1)))
    assert (case['task_id'], case['init_state_id'], case['mode']) == (0, 1, 'development')
    # Scan real result records, not copied code/plans. A used case must not be relabelled new.
    exposed = []
    for path in (root / 'runs').glob('**/stream/RESULT.json'):
        for row in read(path).get('cases', []):
            c = row.get('case', {})
            if (c.get('suite'), c.get('task_id'), c.get('init_state_id')) == ('libero_10', 0, 1):
                exposed.append(str(path.relative_to(root)))
    if exposed: raise ValueError('Selected next instance already exposed: ' + repr(exposed))
    shutil.copyfile(prior, folder / 'FROZEN.json')
    result = advance_snapshot(phase='development', previous_path=folder / 'FROZEN.json',
        previous_sha256=sha(prior), added_pairs=[added], source_root=root,
        output_path=folder / 'UPDATED.json', next_goal=case['task_description'],
        next_session_id='net_t0_i1',
        previous_provenance={p['evidence_id']: provenance(root, p) for p in state['pairs']})
    receipt = result['receipt']
    assert not receipt['selected_mode_changed'] and receipt['next_decision']['selected_mode'] == 'fixed10'
    receipt.update(historical_feedback_replay=True, newly_acquired_robot_samples=0,
        frozen_execution='not dispatched: same selected mode; no counterfactual time or quality claim')
    life.save(folder / 'UPDATE_RECEIPT.json', receipt)
    life.save(folder / 'ADDED_PAIR.json', added)
    protocol = {'name':NAME, 'case':case, 'maximum_campaign_seconds':1300,
        'maximum_new_robot_cases':1, 'source_old_sha256':sha(prior),
        'updated_sha256':sha(folder/'UPDATED.json'), 'phase':'development',
        'objective':'Native read and execution of updated task table, not learned net benefit',
        'frozen_mode':receipt['previous_decision']['selected_mode'],
        'updated_mode':receipt['next_decision']['selected_mode'],
        'default_changed':False, 'verifier_semantics':'off',
        'experience':'original admitted OS starting snapshot; native triggers and admission unchanged',
        'paid_history':'all source lifecycle fees retained separately; no free cold-start claim',
        'cost':'single full lifecycle includes model startup; no comparison against old cold or resident timing',
        'continuation':'no automatic second case; do not buy identical frozen trajectory'}
    life.save(folder / 'PROTOCOL.json', protocol)
    return protocol


def check_execution(run):
    group = run / 'pair0/N'; report = read(group/'stream/RESULT.json')
    row = report['cases'][0]; proto = read(run/'PROTOCOL.json')
    events = [json.loads(x) for x in (group/'stream/OS_COST_EVENTS.jsonl').read_text().splitlines() if x.strip()]
    agent_path=group/'stream'/row['case']['id']/'AGENT_EVENTS.jsonl'
    agent_events=[json.loads(x) for x in agent_path.read_text().splitlines() if x.strip()]
    reads = [e for e in agent_events if e.get('event') == 'task_window_read']
    choices = [e for e in events if e.get('event') == 'feedback_window_choice']
    inputs = [e for e in events if e.get('event') == 'actual_policy_input']
    assert report['status']=='COMPLETED' and row['chain_status']=='PASSED'
    assert len(reads)==1 and reads[0]['snapshot_sha256']==proto['updated_sha256']
    assert row['task_window_decision']['snapshot_sha256']==proto['updated_sha256']
    assert reads[0]['selected_mode']==proto['updated_mode']
    assert choices and len(choices)==len(inputs) and all(e['mode']==proto['updated_mode'] for e in choices)
    allowed_goals={proto['case']['task_description']} | {b['native_attempt_goal'] for b in row.get('native_window_attempts', [])}
    assert all(e['prompt'] in allowed_goals for e in inputs)
    assert read(run/'ADDED_PAIR.json')['evidence_id'] in reads[0]['evidence_ids']
    assert sha(run/'UPDATED.json')==proto['updated_sha256']
    assert sha(run/'FROZEN.json')==proto['source_old_sha256']
    assert sha(run/'pair0/UPDATED.json')==proto['updated_sha256']
    native=row['native_terminal_by_session']
    return {'status':'PASSED','snapshot_read':reads[0], 'native_session_executed':True,
        'window_choices':len(choices),'vla_calls':len(inputs), 'native_terminal_by_session':native, 'native_recovery_attempts':len(row.get('native_window_attempts',[])),
        'environment_success':row['environment_success'],
        'verification_verdicts':[o['verdict'] for o in row['outcomes']],
        'experience_admission':row['experience_admission'],
        'snapshot_unchanged_during_execution':True,
        'selected_mode_changed_by_new_evidence':False, 'learning_gain_established':False,
        'measured_saving_seconds':None, 'scope':'One new dev initial state; updated SHA used in real native OS; no fixed-arm counterfactual'}


def execute():
    assert socket.gethostname()=='j03r4n13'
    root=life.PROJECT; run=root/'runs'/NAME
    lock=open(root/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run.mkdir(exist_ok=False);started=time.time();deadline=started+1300
    configure_context(run,deadline);worker=None
    state={'status':'RUNNING','started_at':started,'deadline':deadline,'pid':os.getpid()}
    life.save(run/'STATUS.json',state)
    def interrupt(n,f):raise KeyboardInterrupt('Campaign interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        # Prepare again against current shared-drive executions before allocating GPUs.
        prep=run/'preparation';proto=prepare(root,prep)
        for rel in ['FROZEN.json','UPDATED.json','UPDATE_RECEIPT.json','ADDED_PAIR.json']:
            shutil.copyfile(prep/rel,run/rel)
        proto['deadline']=deadline;life.save(run/'PROTOCOL.json',proto)
        for rel in ('scripts','src/fastrsi','tests','configs'):freeze(root/rel,run/'code'/rel)
        life.save(run/'CODE_MANIFEST.json',tree(run/'code'))
        source=root/'runs/shared_flow_net_parallel_v2_20261002'
        for rel in BINDINGS:shutil.copyfile(source/rel,run/rel)
        shutil.copytree(source/'initial_experience',run/'initial_experience')
        assert tree(run/'initial_experience')==read(run/'EXPERIENCE_BINDING.json')
        cpu=run/'cpu';cpu.mkdir();life.gate(cpu);check_ports([22100,22102,22110,22120])
        cmd='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && python -m unittest test_task_window test_task_window_snapshot test_task_snapshot_worker test_resident_worker test_task_snapshot_native -v && /project/envs/os-v1/bin/python -m unittest test_native_window_attempt test_native_isolation test_experience_admission -v'
        cid=life.launch('cpu',life.cpu_args('yuchen-fastrsi-'+NAME+'-cpu',cpu)+['-e','FASTRSI_TEST_PROJECT=/project',life.IMAGE,'timeout','120','bash','-lc',cmd],cpu)
        life.wait_exit(cid,130);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());state['cpu']='PASSED'
        shared.NAME=NAME
        services,state['pool_startup_seconds']=shared.pool(run,run/'resident','N',deadline)
        configure_context(run,deadline)
        pair=run/'pair0';pair.mkdir()
        for rel in BINDINGS:shutil.copyfile(run/rel,pair/rel)
        shutil.copytree(run/'initial_experience',pair/'initial_experience')
        shutil.copyfile(run/'UPDATED.json',pair/'UPDATED.json')
        life.save(pair/'PLAN.json',{'cases':[proto['case']]})
        life.save(pair/'PROTOCOL.json',{'slot':0,'absolute_deadline':deadline,'diagnostic_trace':True,
            'control_kind':'feedback_window','arm_modes':{'N':'fixed10'},'port_offset':-27400,
            'gpu_bases':{'N':0},'terminal_evidence_view':True,
            'task_window_snapshots':{'N':'UPDATED.json'},'task_window_snapshot_sha256':{'N':proto['updated_sha256']},
            'resident_services':{'policy_port':22100,'qwen_port':22110,'policy_container':services['pi05'],
                'qwen_container':services['qwen'],'session_id':NAME+'-net-t0-i1','max_inferences':400}})
        group=pair/'N';freeze(run/'code',group/'code')
        for rel in ['scripts/run_native_baseline.py','scripts/qwen_agent_server.py']:
            path=group/'code'/rel;text=path.read_text()
            for port in (49500,49502,49510,49520):text=text.replace(str(port),str(port-27400))
            path.write_text(text)
        life.save(group/'CODE_MANIFEST.json',tree(group/'code'))
        with (group/'controller.log').open('x') as log:
            worker=subprocess.Popen(['python3',str(run/'code/scripts/run_placement_worker.py'),NAME+'/pair0','N'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        life.save(group/'DISPATCH.json',{'pid':worker.pid,'started_at':time.time()})
        while worker.poll() is None:
            if time.time()>deadline-30:raise TimeoutError('Campaign bound')
            time.sleep(2)
        assert worker.returncode==0,read(group/'STATUS.json')
        checked=check_execution(run);life.save(run/'NATIVE_CHECK.json',checked)
        state['worker']=read(group/'STATUS.json');state['status']='COMPLETED'
    except BaseException:state.update(status='FAILED',error=traceback.format_exc())
    finally:
        if worker is not None and worker.poll() is None:
            worker.send_signal(signal.SIGTERM)
            try:worker.wait(timeout=90)
            except subprocess.TimeoutExpired:state['cleanup_attention']='Worker pending; inspect exact receipts'
        life.save(run/'STOP_RECEIPT.json',life.stop_all())
        audit=[]
        for path in run.rglob('LAUNCH_*.json'):
            cid=read(path).get('stdout','').strip()
            if len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid):continue
            info=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
            assert info['Id']==cid and info['Name'].startswith('/yuchen-fastrsi-'+NAME)
            audit.append({'id':cid,'name':info['Name'],'running':info['State']['Running']})
        life.save(run/'CONTAINER_AUDIT.json',audit)
        if any(x['running'] for x in audit):state['cleanup_attention']='Owned container still running'
        if state.get('cleanup_attention'):state['status']='FAILED'
        state.update(finished_at=time.time(),full_wall_seconds=time.time()-started)
        life.save(run/'STATUS.json',state)
    if state['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',type=Path);args=ap.parse_args()
    if args.prepare:prepare(Path(__file__).resolve().parents[1],args.prepare)
    else:execute()
