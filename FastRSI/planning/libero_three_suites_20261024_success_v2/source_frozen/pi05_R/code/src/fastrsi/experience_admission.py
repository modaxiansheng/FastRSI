"""Versioned, whole-workspace admission at the inter-Session boundary.

Uses the runtime result already supplied to the native verifier, never external
benchmark labels. Raw workspaces are retained; unresolved feedback is not
silently repaired. All arms can use this deterministic common infrastructure.
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

SEAL='.FASTRSI_ADMITTED.json'
VERSION='runtime_feedback_agreement_v1'
ASSESSMENT_POLICY='runtime_agreement_resolved_replan_v2'


def resolved_replan_chain(row):
    """Bind a completed native retry chain; a replan proposal alone is not evidence.

    This accepts agreed raw records, not the teacher's causal explanation or a
    transferable skill. The existing policy_loop resets between Sessions.
    """
    sessions=row.get('sessions') or [];outcomes=row.get('outcomes') or []
    by_id={s['session_id']:s for s in sessions};by_out={o['session_id']:o for o in outcomes}
    if len(by_id)!=len(sessions) or set(by_id)!=set(by_out) or len(by_out)!=len(outcomes):
        raise ValueError('recovery_ambiguous_attempts')
    root=by_id[row['case']['id']]
    if root.get('parent_session_id') is not None or root.get('replan_attempt')!=0:
        raise ValueError('recovery_invalid_root')
    if root.get('benchmark',{}).get('execution_mode')!='policy_loop':
        raise ValueError('recovery_unbound_execution_mode')
    invariant=('target_ref','skillruntime_ref','goal_id','benchmark','execution','routing','safety_profile','timeouts','verification_profile')
    current=root;seen=[]
    while True:
        sid=current['session_id'];outcome=by_out[sid]
        if sid in seen or len(seen)>=3:raise ValueError('recovery_cycle_or_limit')
        if current.get('replan_attempt')!=len(seen):raise ValueError('recovery_attempt_mismatch')
        if any(current.get(k)!=root.get(k) for k in invariant):raise ValueError('recovery_changed_constraints')
        seen.append(sid)
        if outcome.get('verdict')!='replan':
            if (outcome.get('verdict')!='success' or outcome.get('status')!='succeeded'
                    or current.get('status')!='succeeded'):
                raise ValueError('recovery_not_resolved_by_success')
            break
        meta=current['result']['metadata'];child_id=outcome.get('replanned_session_id')
        if (outcome.get('status')!='replanned' or current.get('status')!='replanned'
                or not child_id or meta.get('replanned_session_id')!=child_id):
            raise ValueError('recovery_child_binding_missing')
        child=by_id[child_id]
        if child.get('parent_session_id')!=sid:raise ValueError('recovery_parent_mismatch')
        verification=meta.get('verification',{});applied=verification.get('applied_attempt_id')
        records=[a for a in verification.get('attempts',[]) if applied and a.get('attempt_id')==applied]
        if len(records)!=1:raise ValueError('recovery_applied_verdict_missing')
        v=records[0];details=v.get('details') or {}
        if (v.get('source')!='auto' or v.get('mode')!='apply' or v.get('error') is not None
                or v.get('verdict')!='replan' or details.get('verdict')!='replan'
                or not child.get('task_description','').strip()
                or details.get('replan_task_description')!=child['task_description']):
            raise ValueError('recovery_instruction_not_applied')
        current=child
    if set(seen)!=set(by_id):raise ValueError('recovery_orphan_attempt')
    return {'session_ids':seen,'terminal_session_id':seen[-1],
            'scope':'Resolved reset-based native retry observations; not same-state recovery or a validated reusable skill'}

def mode():
    value=os.environ.get('FASTRSI_EXPERIENCE_ADMISSION','guarded_v1')
    if value not in ('guarded_v1','legacy_raw'):raise ValueError('Unknown admission mode')
    return value

def tree_hashes(root):
    root=Path(root)
    if root.is_symlink():raise ValueError('Experience root cannot be a symlink')
    result={}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('Experience cannot contain symlinks: '+str(p))
        if p.is_file() and p.relative_to(root).as_posix()!=SEAL:
            result[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result

def validate_snapshot(root):
    root=Path(root)
    if (root/'.FASTRSI_EVALUATION_CLOSED.json').exists():raise ValueError('Evaluation cannot be promoted')
    seal=root/SEAL
    if not seal.is_file():raise ValueError('Unadmitted experience source; audit/import a clean snapshot first')
    value=json.loads(seal.read_text())
    if value.get('version')!=VERSION or value.get('status')!='ADMITTED':raise ValueError('Invalid admission seal')
    if tree_hashes(root)!=value['files']:raise ValueError('Admitted snapshot changed after sealing')
    return value

def assess(workspace,row):
    """Reject any unresolved attempt; final success cannot wash out an earlier conflict."""
    w=Path(workspace).resolve();reasons=[];attempts=[]
    if row.get('chain_status')!='PASSED':reasons.append('incomplete_chain')
    if row.get('case',{}).get('mode')!='development':reasons.append('not_development')
    outcomes=row.get('outcomes') or [];sessions=row.get('sessions') or []
    ids=[s.get('session_id') for s in sessions];oids=[o.get('session_id') for o in outcomes]
    if not ids or len(ids)!=len(set(ids)) or len(oids)!=len(set(oids)) or set(ids)!=set(oids):reasons.append('missing_or_ambiguous_attempts')
    by_id={s.get('session_id'):s for s in sessions}
    recovery=None
    if any(o.get('verdict')=='replan' for o in outcomes):
        try:recovery=resolved_replan_chain(row)
        except (KeyError,ValueError,TypeError,AttributeError) as exc:reasons.append('recovery:'+str(exc))
    for o in outcomes:
        sid=o.get('session_id');a={'session_id':sid,'protocol_code':o.get('code'),'verdict':o.get('verdict')}
        try:
            if o.get('ok') is not True or o.get('code')!='verified':raise ValueError('verification_not_completed')
            verdict=o.get('verdict')
            if verdict not in ('success','failure') and not (verdict=='replan' and recovery):raise ValueError('unresolved_verdict')
            rel=by_id[sid]['result']['artifact_dir'];p=(w/rel/'verification_bundle.json').resolve()
            if w not in p.parents:raise ValueError('artifact_outside_workspace')
            b=json.loads(p.read_text())
            if b['session']['session_id']!=sid:raise ValueError('bundle_session_mismatch')
            value=b['runtime_result'].get('success')
            if type(value) is not bool:raise ValueError('runtime_success_unknown')
            final=b['runtime_result'].get('metadata',{}).get('final_status',{}).get('success')
            if final is not None and (type(final) is not bool or final!=value):raise ValueError('runtime_evidence_conflict')
            a.update(runtime_success=value,bundle_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bundle=str(p.relative_to(w)))
            if value!=(verdict=='success'):raise ValueError('runtime_semantic_conflict')
        except (KeyError,ValueError,OSError,TypeError) as exc:
            a['rejection']=str(exc);reasons.append(str(sid)+':'+str(exc))
        attempts.append(a)
    return {'version':VERSION,'assessment_policy':ASSESSMENT_POLICY,'eligible':not reasons,'reasons':reasons,'attempts':attempts,'resolved_recovery':recovery,
        'scope':'Agreed observations only; not proof of narrative correctness or transferable skill',
        'evidence_source':'Existing pre-verification runtime_result; no external evaluator input'}

def commit_experience(workspace,row,previous,output):
    """Promote one sealed copy or keep the previous pointer; never inherit quarantine."""
    output=Path(output);w=Path(workspace)
    # Evaluation may read a fixed admitted snapshot, but cannot publish or audit-write it.
    if row.get('case',{}).get('mode')=='evaluation':
        if previous is not None:validate_snapshot(previous)
        return previous,{'status':'EVALUATION_NO_UPDATE','version':VERSION}
    parent=validate_snapshot(previous) if previous is not None else None
    decision=assess(w,row)
    source=row.get('accepted_source')
    if (source is None)!=(previous is None) or (source is not None and Path(source).resolve()!=Path(previous).resolve()):
        decision['eligible']=False;decision['reasons'].append('untrusted_parent_lineage')
    output.mkdir(parents=True,exist_ok=False)
    before=tree_hashes(w)
    receipt={**decision,'source_workspace':str(w),'source_files':before,
        'source_case':row.get('case',{}).get('id'),'parent':str(previous) if previous is not None else None,
        'parent_seal_sha256':hashlib.sha256((Path(previous)/SEAL).read_bytes()).hexdigest() if parent else None}
    if decision['eligible']:
        dest=output/'accepted_experience';shutil.copytree(w,dest)
        inherited=dest/SEAL
        if inherited.exists():inherited.unlink()  # only the new copy; source is untouched
        if tree_hashes(w)!=before or tree_hashes(dest)!=before:raise RuntimeError('Candidate changed during admission')
        seal={'version':VERSION,'assessment_policy':ASSESSMENT_POLICY,'status':'ADMITTED','source_case':receipt['source_case'],
            'parent_seal_sha256':receipt['parent_seal_sha256'],'files':before,
            'scope':'Whole-workspace agreed raw observations, not validated reusable skills'}
        (dest/SEAL).write_text(json.dumps(seal,indent=2));validate_snapshot(dest)
        receipt.update(status='ADMITTED',next_source=str(dest));next_source=dest
    else:
        receipt.update(status='QUARANTINED',next_source=str(previous) if previous is not None else None,
            quarantine='Entire candidate workspace retained as evidence; no files enter the next Session')
        next_source=previous
    (output/'ADMISSION.json').write_text(json.dumps(receipt,indent=2))
    return next_source,receipt
