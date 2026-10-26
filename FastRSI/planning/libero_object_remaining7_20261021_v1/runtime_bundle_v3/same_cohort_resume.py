"""Same frozen Object14 cohort resume metadata and host-only preparation hooks.

No Env, torch, model, container or SSH imports. Host planner re-reads original
raw bytes and checkpoints but NEVER unpickles. The root coordinator owns launches,
fresh resource admission, deadlines and cleanup. No finished physical result can
be converted into permission to retry it.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import shutil
import time

SCHEMA = 'fastrsi_object7_samecohort_resume_v1'
RESTORE_SCHEMA = 'fastrsi_object7_actual_samecohort_rng_restore_v1'
PLAN_NAME = 'CONTINUATION_PLAN.json'
POLICY = 'sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81'
ENV = 'sha256:3171fda2362d773433482f79c4fd53f6f428413efbf41c3fdfb43d8c799d6c95'
BRANCHES = ('ordinary', 'frozen', 'candidate')
SOURCE = Path(__file__).resolve().parent


def need(value, message):
    if not value: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def pin(path): return dict(path=str(Path(path).resolve()), sha256=sha(path), bytes=Path(path).stat().st_size)
def save(path, value):
    path=Path(path)
    with path.open('x') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())


def checked_path(path, old):
    p=Path(path).resolve();need(p.is_relative_to(old), 'Original source escaped frozen runtime')
    need(p.is_file(), 'Original source unavailable: '+str(p));return p


def pinned(reference, old):
    p=checked_path(reference['path'],old)
    need(sha(p)==reference['sha256'], 'Original byte SHA mismatch: '+str(p))
    if 'bytes' in reference:need(p.stat().st_size==reference['bytes'],'Original byte size mismatch')
    return p


def load_leaf(path, old):
    p=checked_path(path,old);return read(p),pin(p)


def source_closure(old):
    byte_receipt=read(old/'BUNDLE_BYTE_COPY_RECEIPT.json')
    need(byte_receipt['local_metadata_only'] is False and byte_receipt['historical_sources_modified'] is False,
         'Not an actual unchanged scientific byte copy')
    manifest=read(old/'BUILD_INPUT_MANIFEST.json')
    need(byte_receipt['source_manifest_sha256']==sha(old/'BUILD_INPUT_MANIFEST.json'),'Input manifest differs')
    records={}
    for name,digest in byte_receipt['files'].items():
        need(sha(checked_path(old/name,old))==digest,'Frozen runtime byte changed: '+name);records[name]=digest
    for name,digest in manifest['package_files'].items():
        need(sha(checked_path(old/name,old))==digest,'Frozen package byte changed: '+name)
    for item in manifest['files']:
        need(sha(checked_path(old/item['destination'],old))==item['sha256'],'Frozen source byte changed')
    return dict(byte_copy_receipt=pin(old/'BUNDLE_BYTE_COPY_RECEIPT.json'),
                input_manifest=pin(old/'BUILD_INPUT_MANIFEST.json'), scientific_files=records,
                all_current_scientific_file_bytes_verified=True)


def validate_base(case, row, old):
    need(len(case['base'])==1,'Ambiguous/partial baseline; do not restart')
    item=case['base'][0];pp=pinned(item['protocol_file'],old);protocol=read(pp)
    need(protocol==item['protocol'],'Protocol observation changed')
    need(protocol['identity']==row['identity'] and protocol['model']==row['model'],'Baseline identity differs')
    need(protocol['software_ok'] is True and type(protocol['first_success']) is bool
         and type(protocol['final_success']) is bool,'Baseline not normally closed')
    attempts=protocol['attempts'];need(1<=len(attempts)<=3 and len(attempts)==len(item['raws']),'Baseline attempt count')
    raws=[];refs=[pin(pp)]
    for index,(attempt,raw_item) in enumerate(zip(attempts,item['raws']),1):
        need(attempt['index']==index,'Attempt index gap')
        p=pinned(raw_item['file'],old);need(sha(p)==attempt['raw_receipt_sha256'],'Protocol raw SHA differs')
        raw=read(p);need(all(raw.get(k)==v for k,v in raw_item['meta'].items()),'Observed raw metadata changed')
        need(raw['error'] is None and raw['episode_completed'] is True and type(raw['physical_success']) is bool,
             'Physical attempt is incomplete/unknown; never retry by inference')
        need(raw['physical_success']==attempt['physical_success'] and raw['initial_state_sha256']==row['identity']['initial_state_sha256'],
             'Raw physical/source identity differs')
        need(0<=raw['actions']<=row['source']['max_steps'],'Action limit changed')
        counts=raw['policy_counts']
        for stem in ('policy_rpc','action_generation','neural_forward'):
            need(counts[stem+'_begin']==counts[stem+'_end'],'Native cost stream not closed')
        raws.append(raw);refs.append(pin(p))
    need(len({r['reset_state_sha256'] for r in raws})==len({r['first_input_sha256'] for r in raws})==1,'Baseline pairing differs')
    need(protocol['first_success']==raws[0]['physical_success'] and protocol['final_success']==any(r['physical_success'] for r in raws),'Protocol success aggregation differs')
    need(not any(r['physical_success'] for r in raws[:-1]),'Attempt after success')
    need(protocol['final_success'] or len(raws)==3 or row['model']=='openvla', 'Nonterminal retry schedule')
    life,life_pin=load_leaf(pp.parent/'CASE_LIFECYCLE.json',old)
    need(life==item['case_lifecycle'] and life['close_error'] is None,'Baseline lifecycle differs/unfinished')
    need(life['resident_end']['status_after']['active'] is None,'Baseline session still active')
    rng=life['resident_end']['receipt']['end_rng_hashes']
    checkpoint=case['checkpoint'];receipt_path=pinned(checkpoint['receipt_file'],old)
    receipt=read(receipt_path);payload=pinned(checkpoint['pickle'],old)
    need(receipt==checkpoint['receipt'] and receipt['roundtrip_verified'] is True and receipt['condition']==row['identity_id'],'Wrong actual checkpoint producer')
    need(receipt['sha256']==sha(payload) and receipt['rng_hashes']==rng,'Checkpoint byte/RNG lifecycle mismatch')
    refs += [life_pin,pin(receipt_path),pin(payload)]
    return dict(protocol_path=str(pp),protocol_sha256=sha(pp),protocol=protocol,
        first_raw_reset_sha256=raws[0]['reset_state_sha256'],first_raw_input_sha256=raws[0]['first_input_sha256'],
        rng_hashes=rng,checkpoint_receipt=pin(receipt_path),checkpoint_pickle=pin(payload),files=refs,
        normal_attempts=len(raws))


def verify_unstarted_actor(lane, case, row, branch, old):
    folder=Path(case['root'])/branch
    known=set(os.listdir(folder)) if folder.is_dir() else set()
    forbidden={'RAW.json','RESULT.json','RESIDENT_ENTRY.json','RESIDENT_END.json','PHYSICAL_POLICY_EVENTS.jsonl',
               row['identity_id']+'_'+branch+'_actor_BINDING.json'}
    need(not known.intersection(forbidden),'Unclosed/unknown actor evidence blocks restart')
    role=row['identity_id']+'_'+branch+'_actor'
    lane_names=set(os.listdir(lane['lane']))
    need('LAUNCH_'+role+'.json' not in lane_names and 'LAUNCH_INTENT_'+role+'.json' not in lane_names,
         'Actor dispatch was attempted; no automatic retry')
    need(not any(x['role']=='LAUNCH_'+role+'.json' for x in lane['owned']), 'Actor had actual container')
    return dict(original_folder=str(folder),original_files=sorted(known),actor_role=role,
        original_lane=str(lane['lane']),no_actor_launch_or_launch_intent=True,
        source_basis='Frozen isolated_coordinator.launch protect then durable launch intent before Docker create',
        prior_vision_load_cost_retained=True)


def validate_arm(case, row, branch, observed, base, old):
    folder=Path(observed['folder']);need(folder==Path(case['root'])/branch,'Arm folder differs')
    result_path=pinned(observed['RESULT.json_pin'],old);result=read(result_path)
    raw_path=pinned(observed['RAW_pin'],old);raw=read(raw_path)
    need(result==observed['RESULT.json'] and result=={k:v for k,v in raw.items() if k!='executed_actions'},'Raw/result mismatch')
    need(result['software_ok'] is True and result['error'] is None and result['episode_completed'] is True
         and result['native_accounting_complete'] is True and result['GT_control_input'] is False
         and type(result['physical_success']) is bool and 0<=result['official_actions']<=row['source']['max_steps'],
         'Arm not a normal physical closure')
    expected=copy.deepcopy(row);canonical=row['identity_id']
    expected.update(canonical_condition=canonical,identity_id=canonical if branch=='ordinary' else canonical+'-'+branch,
        skill=row['skill_by_branch'][branch],after_baseline_rng_hashes=base['rng_hashes'],
        baseline_reset_state_sha256=base['first_raw_reset_sha256'],baseline_first_input_sha256=base['first_raw_input_sha256'])
    expected['source']['skill']=expected['skill']
    expected['support']='ORIGINAL_NATIVE_FALLBACK_NO_OBJECT_SKILL' if expected['skill'] is None else 'EXACT_IMMUTABLE_SOURCE_CAPABILITY'
    job_path=pinned(observed['JOB.json_pin'],old);job=read(job_path)
    need(job==observed['JOB.json'] and job==dict(row=expected,mode='ordinary_fourth' if branch=='ordinary' else 'paired_fourth'),'Frozen arm JOB differs')
    entry_path=pinned(observed['RESIDENT_ENTRY.json_pin'],old);entry=read(entry_path)
    need(entry==observed['RESIDENT_ENTRY.json'] and entry['receipt']['paired_start_rng_hashes']==base['rng_hashes'],'Arm RNG pairing differs')
    inp_path=pinned(observed['INITIAL_INPUT_ACCEPTANCE.json_pin'],old);inp=read(inp_path)
    need(inp==observed['INITIAL_INPUT_ACCEPTANCE.json'] and inp['reset_state_sha256']==base['first_raw_reset_sha256']
         and inp['initial_policy_input_sha256']==base['first_raw_input_sha256'] and inp['original_prompt']==row['source']['language'],'Initial input pairing differs')
    end,end_pin=load_leaf(folder/'RESIDENT_END.json',old)
    need(end['status_after']['active'] is None,'Arm session did not end')
    role=canonical+'_'+branch+'_actor';stop,stop_pin=load_leaf(folder/('STOP_'+role+'.json'),old)
    need(stop['state']['Running'] is False and stop['state']['ExitCode']==0 and not stop['state']['OOMKilled'] and not stop['state']['Error'],'Arm actual container not normally closed')
    ledger=checked_path(folder/'PHYSICAL_POLICY_EVENTS.jsonl',old)
    need(sha(ledger)==result['policy_ledger_sha256'],'Arm full policy cost SHA differs')
    return dict(result=result,raw_path=str(raw_path),raw_sha256=sha(raw_path),normal_attempts=1,
        original_folder=str(folder),files=[pin(result_path),pin(raw_path),pin(job_path),pin(entry_path),pin(inp_path),end_pin,stop_pin,pin(ledger)])


def build_plan(old_root, scope_sha256, drain_pins):
    old=Path(old_root).resolve();scope=read(old/'FROZEN_RUNTIME_SCOPE.json')
    need(sha(old/'FROZEN_RUNTIME_SCOPE.json')==scope_sha256,'Canonical frozen scope SHA mismatch')
    need(scope['status']=='ALL56_OBJECT7_CONDITIONS_FROZEN_BEFORE_TRIAL' and len(scope['conditions'])==56
         and scope['normal_physical_attempts_max']==336,'Do not bypass original 14/56/336 shape')
    by_id={r['identity_id']:r for r in scope['conditions']}
    need(len(by_id)==56 and len({r['physical_identity_id'] for r in by_id.values()})==14,'Physical cohort changed')
    closure=source_closure(old);snapshots=[];cases={};group_evidence={}
    for reference in drain_pins:
        p=Path(reference['path']).resolve();need(sha(p)==reference['sha256'],'Drain evidence SHA differs')
        observed=read(p);need(observed['scope_sha256']==scope_sha256 and observed['node'] in ('187','188'),'Drain scope differs')
        need(observed['HALT']['source']=='later_models_driver_v1' and observed['HALT']['no_restart'] is True,'Unknown HALT source')
        for lane in observed['lanes']:
            need(lane['identity']['frozen_scope_sha256']==scope_sha256,'Lane source scope differs')
            need(lane['stop']['cleanup_errors']==[] and all(not v['state']['Running'] for v in lane['stop']['states'].values()),'Original cleanup not closed')
            need(all(not v['state']['Running'] for v in lane['owned']),'Original own container remains running')
            group_evidence[lane['lane']]=dict(identity=lane['identity'],stop=lane['stop'],lifecycle=lane['lifecycle'],source_snapshot=pin(p))
            for case in lane['cases']:
                condition=case['condition'];need(condition not in cases and condition in by_id,'Duplicate/foreign historical case')
                row=by_id[condition];need(case['row']==row and row['model']=='pi05','Original ROW changed or outside π0.5 stage')
                need(case['incomplete_attempt_files']==[],'Interrupted physical attempt blocks this continuation')
                base=validate_base(case,row,old);arms={};remaining=[];unstarted={}
                if not base['protocol']['final_success']:
                    for branch in BRANCHES:
                        observed_arm=case['branches'].get(branch)
                        if observed_arm and 'RESULT.json' in observed_arm:
                            arms[branch]=validate_arm(case,row,branch,observed_arm,base,old)
                        elif branch=='candidate' and row['source']['task_id']==2:
                            need(not observed_arm,'Unsupported candidate unexpectedly started')
                        else:
                            unstarted[branch]=verify_unstarted_actor(lane,case,row,branch,old);remaining.append(branch)
                complete=case['closed_condition']
                completed_ref=None
                if complete is not None:
                    actual,completed_ref=load_leaf(Path(case['root'])/'CONDITION_RESULT.json',old)
                    need(actual==complete and actual['software_ok'] is True and actual['condition']==condition,'Closed condition changed')
                    need(not remaining and actual['baseline_protocol_sha256']==base['protocol_sha256'],'Closed condition incomplete')
                    need(actual['shared_baseline_end_rng']==base['rng_hashes'],'Closed condition RNG mismatch')
                    if not base['protocol']['final_success']:
                        need(set(arms)==set(BRANCHES),'Closed failed-native condition lacks arm results')
                        for b in BRANCHES:need(actual['branches'][b]['raw_sha256']==arms[b]['raw_sha256'],'Closed arm pointer differs')
                startup_path=checked_path(Path(lane['lane'])/'resident_receipts/STARTUP.json',old);startup=read(startup_path)
                cases[condition]=dict(condition=condition,row=row,original_case=case['root'],original_lane=lane['lane'],
                    status='REUSE_CLOSED_CONDITION' if complete is not None else 'RESUME_ONLY_UNSTARTED_ARMS',
                    scoped_reuse_counts=dict(closed_condition=(base['normal_attempts']+len(arms) if complete is not None else 0),
                        baseline=base['normal_attempts'],arms={b:1 for b in arms}),
                    condition_result=complete,condition_result_file=completed_ref,baseline=base,arms=arms,
                    remaining_arms=remaining,unstarted_actor_proofs=unstarted,
                    candidate_skipped_unsupported=(not base['protocol']['final_success'] and row['source']['task_id']==2),
                    old_startup=pin(startup_path),old_startup_rng_hashes=startup['rng_hashes'],
                    used_normal_attempts=base['normal_attempts']+len(arms))
        snapshots.append(pin(p))
    need(len(cases)==9 and sum(v['status']=='REUSE_CLOSED_CONDITION' for v in cases.values())==6,
         'Actual fixed drain shape differs; no broadened automatic recovery')
    need(sum(v['used_normal_attempts'] for v in cases.values())==30,'Actual closed attempt accounting changed')
    need(sum(len(v['remaining_arms']) for v in cases.values())==5,'Partial remaining-arm budget differs')
    for condition,row in by_id.items():
        if condition not in cases:
            original_group=next(g for g in scope['groups'] if condition in g['condition_ids'])
            lane=old/'campaigns'/(row['source']['suite']+'__'+row['model'])/('task'+str(row['source']['task_id']))/('node'+row['node'])/('gpu'+str(row['gpu']))
            need(not (lane/condition).exists(),'Unobserved historical case blocks restart')
            cases[condition]=dict(condition=condition,row=row,status='UNSTARTED_ORIGINAL_CONDITION',
                original_case=str(lane/condition),original_lane=str(lane),condition_result=None,condition_result_file=None,
                baseline=None,arms={},remaining_arms=list(BRANCHES),used_normal_attempts=0,
                scoped_reuse_counts=dict(closed_condition=0,baseline=0,arms={}),
                candidate_skipped_unsupported=row['source']['task_id']==2,unstarted_actor_proofs={})
    group_plans=[]
    for group in scope['groups']:
        records=[cases[cid] for cid in group['condition_ids']]
        lane=Path(records[0]['original_lane'])
        complete=all(r['status']=='REUSE_CLOSED_CONDITION' for r in records)
        action='REUSE_COMPLETE_GROUP' if complete else ('RESUME_PARTIAL' if any(r['baseline'] is not None for r in records) else 'NEW_GROUP')
        files={}
        if complete:
            for name in ('GROUP_RESULT.json','OWNED_STOP_RECEIPT.json','LIFECYCLE.json'):
                p=checked_path(lane/name,old);files[name]=pin(p)
            closed=read(lane/'GROUP_RESULT.json')
            need(closed['status']=='FROZEN_GROUP_COMPLETED' and closed['completed']==group['condition_ids'],'Complete group status changed')
        group_plans.append(dict(group,action=action,original_lane=str(lane),reused_group_files=files,
            reused_condition_files={r['condition']:r['condition_result_file'] for r in records if r['condition_result_file'] is not None},
            used_normal_attempts=sum(r['used_normal_attempts'] for r in records)))
    return dict(schema=SCHEMA,time=time.time(),original_root=str(old),original_scope=pin(old/'FROZEN_RUNTIME_SCOPE.json'),
        original_halt=pin(old/'HALT_OBJECT7.json'),scope=scope,source_closure=closure,
        original_drain_snapshots=snapshots,original_group_evidence=group_evidence,conditions=cases,groups=group_plans,
        physical_identities=14,model_conditions=56,normal_physical_attempts_max=336,
        already_closed_normal_attempts=30,partial_unstarted_pi05_arm_count=5,
        closed_conditions_reused=6,untouched_pi05_conditions=5,full_evaluation='PAUSED',
        original_results_or_scientific_algorithms_changed=False,
        helper_sources={p.name:pin(p) for p in [SOURCE/'same_cohort_resume.py',SOURCE/'paired_samecohort_restore.py']})


def recheck_case(plan, record):
    old=Path(plan['original_root']).resolve()
    refs=[]
    if record.get('condition_result_file'):refs.append(record['condition_result_file'])
    if record.get('baseline'):refs.extend(record['baseline']['files']);refs.append(record['old_startup'])
    for arm in record['arms'].values():refs.extend(arm['files'])
    for reference in refs:pinned(reference,old)


def load_plan(ROOT,model,task,node,gpu,continuation_plan=None):
    root=Path(ROOT).resolve();path=Path(continuation_plan) if continuation_plan is not None else root/PLAN_NAME;plan=read(path)
    need(plan['schema']==SCHEMA and plan['physical_identities']==14 and plan['model_conditions']==56
         and plan['normal_physical_attempts_max']==336 and plan['full_evaluation']=='PAUSED','Resume scope differs')
    binding=read(root/'RECOVERY_SOURCE_BINDINGS.json')
    need(binding['continuation_plan_sha256']==sha(path),'New resume plan SHA mismatch')
    for name in ('same_cohort_resume.py','paired_samecohort_restore.py'):
        need(sha(root/name)==binding['files'][name],'Recovery helper source differs')
    need(sha(root/'FROZEN_RUNTIME_SCOPE.json')==plan['original_scope']['sha256'] and read(root/'FROZEN_RUNTIME_SCOPE.json')==plan['scope'],'New canonical scientific scope differs')
    original=Path(plan['original_root']).resolve();source_closure(original)
    for reference in [plan['original_scope'],plan['original_halt']]:pinned(reference,original)
    for reference in plan['original_drain_snapshots']:
        need(sha(reference['path'])==reference['sha256'],'Original stop snapshot changed')
    selected=next(g for g in plan['scope']['groups'] if (g['model'],g['task_id'],g['node'],g['gpu'])==(model,task,node,gpu))
    result=copy.deepcopy(plan);result['selected_group']=selected;result['_runtime_root']=str(root)
    for cid in selected['condition_ids']:recheck_case(plan,plan['conditions'][cid])
    return result


def prepare_lane(plan,lane,rows):
    lane=Path(lane);need([r['identity_id'] for r in rows]==plan['selected_group']['condition_ids'],'Selected cohort changed')
    directory=lane/'resident_receipts';directory.mkdir(exist_ok=False)
    items=[];startup=None
    for row in rows:
        record=plan['conditions'][row['identity_id']];need(record['row']==row,'Frozen ROW mismatch');recheck_case(plan,record)
        if record['status']=='REUSE_CLOSED_CONDITION':
            target=lane/row['identity_id'];need(not target.exists(),'New case path already occupied')
            target.symlink_to(record['original_case'],target_is_directory=True)
        elif record['baseline'] is not None:
            base=record['baseline'];old=Path(plan['original_root']).resolve()
            for reference in [base['checkpoint_receipt'],base['checkpoint_pickle']]:
                p=pinned(reference,old);destination=directory/p.name
                need(not destination.exists(),'Checkpoint destination collision')
                with destination.open('xb') as handle:handle.write(p.read_bytes());handle.flush();os.fsync(handle.fileno())
                need(sha(destination)==reference['sha256'],'Checkpoint copy byte mismatch')
            if startup is None:startup=record['old_startup_rng_hashes']
            need(startup==record['old_startup_rng_hashes'],'Group startup RNG differs')
            items.append(dict(condition=row['identity_id'],relative_path='resident_receipts/'+Path(base['checkpoint_pickle']['path']).name,
                sha256=base['checkpoint_pickle']['sha256'],rng_hashes=base['rng_hashes']))
    if items:
        save(lane/'RESTORE_PLAN.json',dict(schema=RESTORE_SCHEMA,original_scope_sha256=plan['original_scope']['sha256'],
             old_startup_rng_hashes=startup,checkpoints=items,noise_draw_reconstruction=0))
    save(lane/'RESUME_PREPARED.json',dict(time=time.time(),original_scope_sha256=plan['original_scope']['sha256'],
        rows=[r['identity_id'] for r in rows],restored_checkpoints=items,new_normal_attempts=0,
        old_results_modified=False))


def reuse_closed_condition(plan,row,lane):
    record=plan['conditions'][row['identity_id']];recheck_case(plan,record)
    return copy.deepcopy(record['condition_result'])


def reuse_baseline(plan,row,lane,case):
    record=plan['conditions'][row['identity_id']];recheck_case(plan,record)
    if record['baseline'] is None:return None
    target=Path(case)/'base';need(not target.exists(),'Baseline destination already occupied')
    target.symlink_to(Path(record['original_case'])/'base',target_is_directory=True)
    return Path(record['baseline']['protocol_path'])


def reuse_arm(plan,row,branch,case):
    record=plan['conditions'][row['identity_id']];recheck_case(plan,record)
    if branch not in record['arms']:return None
    arm=record['arms'][branch];target=Path(case)/branch;need(not target.exists(),'Arm destination occupied')
    target.symlink_to(arm['original_folder'],target_is_directory=True)
    return dict(result=copy.deepcopy(arm['result']),raw_path=arm['raw_path'],raw_sha256=arm['raw_sha256'])


def policy_restore_command(plan,lane,argv):
    lane=Path(lane);a=list(argv)
    restore_required=any(plan['conditions'][cid]['baseline'] is not None and plan['conditions'][cid]['status']!='REUSE_CLOSED_CONDITION'
                         for cid in plan['selected_group']['condition_ids'])
    if not restore_required:return a
    need((lane/'RESTORE_PLAN.json').is_file(),'Required exact checkpoint restore plan missing; do not load')
    root=Path(plan['_runtime_root']);binding=read(root/'RECOVERY_SOURCE_BINDINGS.json')
    overlay=root/'paired_samecohort_restore.py';need(sha(overlay)==binding['files'][overlay.name],'Restore transport source changed')
    old=Path(plan['original_root']);original=old/'runtime/paired_alias_resident.py'
    need(sha(original)==plan['source_closure']['scientific_files']['runtime/paired_alias_resident.py'],'Frozen alias owner source differs')
    need(a.count(POLICY)==1,'Wrong frozen policy image');i=a.index(POLICY)
    a[i:i]=['-v',str(overlay)+':/code/paired_alias_resident.py:ro',
            '-v',str(original)+':/transport/paired_alias_resident_frozen.py:ro']
    need(sum(v.startswith('PYTHONPATH=') for v in a)==1,'Policy path binding ambiguous')
    a=[v+':/transport' if v.startswith('PYTHONPATH=') else v for v in a]
    save(lane/'POLICY_RESTORE_SOURCE_PROOF.json',dict(time=time.time(),restore_plan=pin(lane/'RESTORE_PLAN.json'),
        overlay=pin(overlay),frozen_alias_owner=pin(original),original_policy_argv=argv,recovery_policy_argv=a,
        native_loader_sampling_actor_or_checkpoint_changed=False,noise_draw_reconstruction=0))
    return a


def main():
    p=argparse.ArgumentParser();p.add_argument('--original-root',required=True);p.add_argument('--scope-sha256',required=True)
    p.add_argument('--drain',action='append',nargs=2,metavar=('PATH','SHA256'),required=True);p.add_argument('--output',required=True)
    a=p.parse_args();plan=build_plan(a.original_root,a.scope_sha256,[dict(path=x,sha256=y) for x,y in a.drain]);save(a.output,plan)
    print(json.dumps(dict(schema=SCHEMA,plan_sha256=sha(a.output),closed_conditions=6,new_physical_attempts=0,
          partial_unstarted_pi05_arms=5,untouched_pi05_conditions=5,full_evaluation='PAUSED')))

if __name__=='__main__':main()
