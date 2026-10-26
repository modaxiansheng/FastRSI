"""Acquire matched development evidence, fit, freeze, and execute transfer controls."""
import fcntl,hashlib,json,os,shutil,signal,socket,subprocess,time,traceback
from pathlib import Path
import orchestrate_full_os as life
import orchestrate_structural_window as shared
from orchestrate_resident_full_os import BINDINGS,read,tree,freeze
from orchestrate_resident_services import configure_context,check_ports
from learning_transfer_study import prepare,learn,sha,group_record
from task_window_study import pair_record
from fastrsi.task_window import select
NAME='learning_transfer_v2_20261004'
PREVIOUS='learning_transfer_v1_20261004'

def binding(arm):
    i=('N','F','R').index(arm)
    return dict(policy_port=22100+i*100,qwen_port=22110+i*100,gpu_base=i*2,mode='fixed10')


def main():
    assert socket.gethostname()=='j03r4n13'
    root=life.PROJECT;run=root/'runs'/NAME
    lock=open(root/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run.mkdir(exist_ok=False);start=time.time();deadline=start+3000;configure_context(run,deadline)
    state=dict(status='RUNNING',started_at=start,phase='preparation',groups=[]);children=[]
    def stop(n,f):raise KeyboardInterrupt('Interrupted '+str(n))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def checkpoint():life.save(run/'STATUS.json',state)
    def execute_case(case,phase,index,modes,snapshots):
        pair=run/phase/('pair'+str(index));pair.mkdir(parents=True)
        for rel in BINDINGS:shutil.copyfile(run/rel,pair/rel)
        shutil.copytree(run/'initial_experience',pair/'initial_experience')
        # Shared evidence access: each condition receives both versions and the full acquisition ledger.
        for rel in ('FROZEN.json','UPDATED.json','NEW_PAIRS.json'):
            if (run/rel).exists():shutil.copyfile(run/rel,pair/rel)
        life.save(pair/'PLAN.json',{'cases':[case]})
        handles=[]
        for arm,mode in modes.items():
            ports=binding(arm);proto=dict(slot=0,absolute_deadline=deadline,diagnostic_trace=True,control_kind='feedback_window',arm_modes=modes,port_offset=-27400,gpu_bases={a:binding(a)['gpu_base'] for a in modes},terminal_evidence_view=True,
                resident_services=dict(policy_port=ports['policy_port'],qwen_port=ports['qwen_port'],policy_container=pools[arm]['pi05'],qwen_container=pools[arm]['qwen'],session_id=NAME+'-'+phase+'-'+str(index)+'-'+arm,max_inferences=400))
            # Worker reads a common protocol; resident bindings differ, so use an arm-specific campaign folder.
            campaign=pair/('campaign_'+arm);campaign.mkdir()
            for rel in BINDINGS:shutil.copyfile(run/rel,campaign/rel)
            shutil.copytree(run/'initial_experience',campaign/'initial_experience')
            for rel in ('FROZEN.json','UPDATED.json','NEW_PAIRS.json'):
                if (run/rel).exists():shutil.copyfile(run/rel,campaign/rel)
            if arm in snapshots:
                rel=snapshots[arm];proto.update(task_window_snapshots={arm:rel},task_window_snapshot_sha256={arm:sha(campaign/rel)})
            life.save(campaign/'PLAN.json',{'cases':[case]});life.save(campaign/'PROTOCOL.json',proto)
            group=campaign/arm;freeze(run/'code',group/'code')
            shift=-27400+('N','F','R','S').index(arm)*100
            for rel in ('scripts/run_native_baseline.py','scripts/qwen_agent_server.py'):
                p=group/'code'/rel;s=p.read_text()
                for port in (49500,49502,49510,49520):s=s.replace(str(port),str(port+shift))
                p.write_text(s)
            life.save(group/'CODE_MANIFEST.json',tree(group/'code'))
            log=(group/'controller.log').open('x');proc=subprocess.Popen(['python3',str(run/'code/scripts/run_placement_worker.py'),str(campaign.relative_to(root/'runs')),arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            log.close();children.append(proc);handles.append((arm,group,proc));life.save(group/'DISPATCH.json',dict(pid=proc.pid,started_at=time.time()))
        while any(p.poll() is None for _,_,p in handles):
            if time.time()>deadline-40:raise TimeoutError('Bounded campaign expired')
            time.sleep(2)
        for arm,group,proc in handles:
            receipt=read(group/'STATUS.json');state['groups'].append(group_record(receipt,phase=phase,index=index,arm=arm,path=str(group.relative_to(root))));checkpoint()
            assert proc.returncode==0 and receipt['status']=='COMPLETED',receipt
        return pair
    try:
        previous=root/'runs'/PREVIOUS
        assert read(previous/'STATUS.json')['status']=='FAILED'
        assert all(not x['running'] for x in read(previous/'CONTAINER_AUDIT.json'))
        shutil.copytree(previous/'preparation',run/'preparation')
        proto=read(run/'preparation/PROTOCOL.json')
        declared=read(root/'planning'/PREVIOUS/'PROTOCOL.json')
        assert proto==declared,'Predeclared protocol changed'
        assert sha(run/'preparation/FROZEN.json')==sha(root/'planning'/PREVIOUS/'FROZEN.json')
        proto['reused_development']={'0':'runs/'+PREVIOUS+'/development/pair0'}
        proto['resume_from']=PREVIOUS
        shutil.copyfile(run/'preparation/FROZEN.json',run/'FROZEN.json')
        life.save(run/'PROTOCOL.json',dict(proto,deadline=deadline))
        for rel in ('scripts','src/fastrsi','tests','configs'):freeze(root/rel,run/'code'/rel)
        life.save(run/'CODE_MANIFEST.json',tree(run/'code'))
        source=root/'runs/shared_flow_net_parallel_v2_20261002'
        for rel in BINDINGS:shutil.copyfile(source/rel,run/rel)
        shutil.copytree(source/'initial_experience',run/'initial_experience');assert tree(run/'initial_experience')==read(run/'EXPERIENCE_BINDING.json')
        life.gate(run);check_ports([base+offset for base in (22100,22200,22300) for offset in (0,2,10,20)])
        shared.NAME=NAME;shared.binding=binding;pools={};state['pool_startup_seconds']={}
        for arm in ('N','F','R'):
            pools[arm],state['pool_startup_seconds'][arm]=shared.pool(run,run/'resident'/arm,arm,deadline);checkpoint()
        configure_context(run,deadline);state['phase']='development';checkpoint();pairs=[]
        for i,case in enumerate(proto['development']):
            if str(i) in proto.get('reused_development',{}):
                pair=root/proto['reused_development'][str(i)]
                for arm in ('N','F'):
                    oldcase=read(pair/('campaign_'+arm)/'PLAN.json')['cases'][0]
                    assert oldcase==case,'Reused case differs'
                    receipt=read(pair/('campaign_'+arm)/arm/'STATUS.json');assert receipt['status']=='COMPLETED'
                    state['groups'].append(group_record(receipt,phase='development',index=i,arm=arm,reused=True,path=str((pair/('campaign_'+arm)/arm).relative_to(root))))
                checkpoint()
            else:
                pair=execute_case(case,'development',i,{'N':'fixed10','F':'fixed5'},{})
            record=pair_record(root,pair,arm_paths={a:str((pair/('campaign_'+a)/a).relative_to(root)) for a in ('N','F')},cost_compatible=True)
            pairs.append(record);life.save(run/'NEW_PAIRS.json',pairs)
        state['phase']='update';checkpoint();frozen=read(run/'FROZEN.json');updated,receipt=learn(frozen,pairs)
        life.save(run/'UPDATED.json',updated);receipt['decisions']=[dict(task=c['task_id'],frozen=select(frozen,c['task_description']),updated=select(updated,c['task_description'])) for c in proto['evaluation']]
        receipt.update(frozen_sha256=sha(run/'FROZEN.json'),updated_sha256=sha(run/'UPDATED.json'));life.save(run/'UPDATE_RECEIPT.json',receipt)
        state['phase']='evaluation';checkpoint()
        for i,c in enumerate(proto['evaluation']):
            execute_case(c,'evaluation',i,{'N':receipt['fixed_mode'],'F':select(updated,c['task_description'])['selected_mode'],'R':select(frozen,c['task_description'])['selected_mode']},{'F':'UPDATED.json','R':'FROZEN.json'})
        assert sha(run/'FROZEN.json')==receipt['frozen_sha256'] and sha(run/'UPDATED.json')==receipt['updated_sha256']
        state.update(status='COMPLETED',phase='finished')
    except BaseException:state.update(status='FAILED',error=traceback.format_exc())
    finally:
        for p in children:
            if p.poll() is None:p.send_signal(signal.SIGTERM)
        for p in children:
            try:p.wait(timeout=90)
            except subprocess.TimeoutExpired:state.setdefault('cleanup_attention',[]).append(p.pid)
        life.save(run/'STOP_RECEIPT.json',life.stop_all())
        audit=[]
        for path in run.rglob('LAUNCH_*.json'):
            cid=read(path).get('stdout','').strip()
            if len(cid)!=64 or any(x not in '0123456789abcdef' for x in cid):continue
            info=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
            assert info['Id']==cid and info['Name'].startswith('/yuchen-fastrsi-')
            audit.append(dict(id=cid,name=info['Name'],running=info['State']['Running']))
        life.save(run/'CONTAINER_AUDIT.json',audit)
        if any(x['running'] for x in audit):state['cleanup_attention']='Owned resources remain'
        if state.get('cleanup_attention'):state['status']='FAILED'
        state.update(finished_at=time.time(),full_wall_seconds=time.time()-start);checkpoint();print(json.dumps(state),flush=True)
    if state['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()
