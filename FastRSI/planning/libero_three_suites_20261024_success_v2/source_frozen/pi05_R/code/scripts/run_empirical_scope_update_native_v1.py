"""Actual native tools commit empirical development updates; no LLM/simulator.

The update originates in measured feedback, never scripted teacher replies.
All registry checks and the frozen next-Session consumer use existing native
interfaces. A successful transaction remains an unadmitted development draft.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import probe_native_goal_component_update_v1 as base
from fastrsi.empirical_scope_update_v1 import load_development_observations, select_update, digest
from fastrsi.native_update_common_case_v1 import common_case_protocol
from fastrsi.paged_reads import read_complete
from fastrsi import typed_component_transaction_v1 as transaction

VERSION='empirical_scope_update_native_v1'


def checked_inputs(campaign, binding_sha256):
    raw=(campaign/'BINDING.json').read_bytes()
    if base.sha(raw)!=binding_sha256:raise ValueError('Explicit update binding changed')
    binding=json.loads(raw)
    manifest=(campaign/'INPUT_MANIFEST.json').read_bytes()
    if base.sha(manifest)!=binding['input_manifest_sha256']:raise ValueError('Update input manifest changed')
    for rel,expected in json.loads(manifest).items():
        path=campaign/rel
        if (Path(rel).is_absolute() or '..' in Path(rel).parts or path.is_symlink()
                or base.sha(path.read_bytes())!=expected):raise ValueError('Frozen update input changed: '+rel)
    return binding


def contained(root, relative):
    root=Path(root).resolve();rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Campaign-relative input required')
    path=root/rel
    if path==root or root not in path.resolve().parents:raise ValueError('Input escapes campaign')
    for part in (path,*path.parents):
        if part==root:break
        if part.is_symlink():raise ValueError('Symlinked continuation input')
    return path


def pinned_json(campaign, reference, inputs):
    if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:
        raise ValueError('Explicit relative raw-SHA reference required')
    path=contained(campaign,reference['path'])
    if inputs.get(reference['path'])!=reference['sha256'] or base.sha(path.read_bytes())!=reference['sha256']:
        raise ValueError('Continuation reference differs from frozen input manifest')
    return json.loads(path.read_bytes())


def directory_manifest(root):
    values={}
    for path in sorted(Path(root).rglob('*')):
        if path.is_symlink():raise ValueError('Symlinked inherited workspace')
        if path.is_file():values[path.relative_to(root).as_posix()]=base.sha(path.read_bytes())
    return values


def merge_observations(batches):
    """Physical records count once only when every measured field agrees."""
    merged={};identities={};duplicates=0
    for rows in batches:
        for row in rows:
            if row.get('role')!='development':raise ValueError('Only actual development evidence may accumulate')
            identity=row['result_sha256']
            if identity in merged:
                if merged[identity]!=row:raise ValueError('Duplicate physical result has conflicting contents')
                duplicates+=1;continue
            key=(row['protocol_sha256'],row['context_sha256'],row['candidate'])
            if key in identities:raise ValueError('Repeated branch requires registered replicate identity')
            identities[key]=identity;merged[identity]=copy.deepcopy(row)
    rows=sorted(merged.values(),key=lambda row:(row['protocol_sha256'],row['context_sha256'],row['candidate']))
    return rows,duplicates


def development_batches(campaign,binding):
    campaign=Path(campaign).resolve()
    inputs=json.loads((campaign/'INPUT_MANIFEST.json').read_bytes())
    batches=[dict(package_root='successor_package',evidence_refs=binding['evidence_refs'])]
    additional=binding.get('additional_evidence_batches',[])
    if not isinstance(additional,list):raise ValueError('Explicit additional evidence batch list required')
    batches.extend(additional);loaded=[];receipts=[];known_catalogue=None
    for batch in batches:
        if not isinstance(batch,dict) or set(batch) not in ({'package_root','evidence_refs'}, {'package_root','evidence_refs','version'}):
            raise ValueError('Explicit package root and evidence references required')
        package=contained(campaign,batch['package_root']);refs=batch['evidence_refs']
        manifest_path=Path(batch['package_root'])/'MANIFEST.json'
        package_manifest=pinned_json(campaign,dict(path=manifest_path.as_posix(),sha256=inputs.get(manifest_path.as_posix())),inputs)
        if package_manifest.get('version')!='native_goal_component_package_v2':raise ValueError('Original bound source package required')
        for relative,h in package_manifest['files'].items():
            file=contained(package,relative);rel=file.relative_to(campaign).as_posix()
            if inputs.get(rel)!=h or base.sha(file.read_bytes())!=h:raise ValueError('Evidence package file is not frozen')
        if batch.get('version')=='native_stream_feedback_v1':
            from fastrsi.native_stream_feedback_v1 import load_stream_observations
            rows=load_stream_observations(package,**refs)
        elif batch.get('version')=='native_failed_operation_feedback_v1':
            from fastrsi.native_failed_operation_feedback_v1 import load_failed_operation_observations
            rows=load_failed_operation_observations(package,**refs)
        elif 'version' not in batch:
            rows=load_development_observations(package,**refs)
        else:raise ValueError('Unknown development evidence format')
        catalogue=json.loads(contained(package,refs['catalogue']['path']).read_bytes())['candidates']
        if known_catalogue is None:known_catalogue=catalogue
        elif any(key not in known_catalogue or known_catalogue[key]!=value for key,value in catalogue.items()):
            raise ValueError('Batched candidate identity differs from current source catalogue')
        loaded.append(rows);receipts.append(dict(package_root=batch['package_root'],evidence_refs=copy.deepcopy(refs),
            package_manifest_sha256=base.sha((package/'MANIFEST.json').read_bytes()),physical_results=[r['result_sha256'] for r in rows]))
        if 'version' in batch:receipts[-1]['version']=batch['version']
    merged,duplicates=merge_observations(loaded)
    return merged,dict(version='empirical_development_batches_v1',batches=receipts,
        duplicate_identical_results=duplicates,unique_physical_results=len(merged),evidence_sha256=digest(merged))


def select_bound_update(campaign,binding,observations,candidates,*,incumbent,previous=None):
    """New aggregate rule is explicit; historical per-instance policy stays default."""
    policy=binding.get('selection_policy')
    if policy is None:return select_update(observations,candidates,incumbent=incumbent,previous=previous)
    if not isinstance(policy,dict) or set(policy)!={'version','cohort','batch'} or policy['version']!='empirical_stream_selection_v1':
        raise ValueError('Explicit supported selection policy required')
    batch=policy['batch']
    if batch not in binding.get('additional_evidence_batches',[]) or batch.get('version')!='native_stream_feedback_v1':
        raise ValueError('Selection cohort must bind a complete validated native branch batch')
    from fastrsi.native_stream_feedback_v1 import load_stream_observations
    from fastrsi.empirical_stream_selection_v1 import select_stream_update
    cohort_rows=load_stream_observations(contained(campaign,batch['package_root']),**batch['evidence_refs'])
    exact=sorted({(row['protocol_sha256'],row['context_sha256']) for row in cohort_rows})
    requested=sorted(tuple(pair) for pair in policy['cohort'])
    if exact!=requested:raise ValueError('Selection omitted or altered preregistered physical input roster')
    if any(row not in observations for row in cohort_rows):raise ValueError('Selection evidence missing from cumulative history')
    return select_stream_update(observations,candidates,incumbent=incumbent,previous=previous,cohort=policy['cohort'])


def previous_state(campaign,binding,observations):
    previous=binding.get('previous_run')
    if previous is None:return None
    if not isinstance(previous,dict) or set(previous)!={'result','update','workspace','workspace_manifest','source_workspace'}:
        raise ValueError('Explicit previous run identity required')
    inputs=json.loads((campaign/'INPUT_MANIFEST.json').read_bytes())
    result=pinned_json(campaign,previous['result'],inputs);update=pinned_json(campaign,previous['update'],inputs)
    manifest=pinned_json(campaign,previous['workspace_manifest'],inputs);workspace=contained(campaign,previous['workspace'])
    if result.get('version')!=VERSION or result.get('update_origin')!='EMPIRICAL_DEVELOPMENT_FEEDBACK' or result.get('status') not in (
            'COMPLETED_EMPIRICAL_SCOPE_UPDATE','COMPLETED_EMPIRICAL_NO_CHANGE','COMPLETED_EMPIRICAL_SCOPE_WITHDRAWAL'):
        raise ValueError('Previous actual completed empirical update required')
    if any(type(result.get(k)) is not int or result[k]!=0 for k in ('model_calls','robot_episodes','policy_inference_calls','training_updates')):
        raise ValueError('Previous empirical origin differs')
    if result.get('evaluation_updates_blocked') is not True or result.get('original_campaign_inputs_unchanged') is not True:
        raise ValueError('Previous input/readonly isolation missing')
    from fastrsi.empirical_scope_update_v1 import VERSION as UPDATE_VERSION
    if update.get('version')!=UPDATE_VERSION:raise ValueError('Unsupported previous selector version')
    if result.get('update')!=update or result['decision']['workspace']!=previous['source_workspace']:
        raise ValueError('Previous RESULT/UPDATE/workspace identity differs')
    if not workspace.is_dir() or directory_manifest(workspace)!=manifest or (workspace/'.FASTRSI_EVALUATION_CLOSED.json').exists():
        raise ValueError('Previous development workspace differs or is evaluation-closed')
    for rel,h in manifest.items():
        if inputs.get((Path(previous['workspace'])/rel).as_posix())!=h:
            raise ValueError('Previous workspace bytes not in frozen campaign inputs')
    decision=result['decision'];registry_sha=base.sha((workspace/'SKILLRUNTIME.md').read_bytes())
    if decision.get('registry_sha256')!=registry_sha:raise ValueError('Previous registry differs from actual decision')
    import yaml,re
    text=(workspace/'SKILLRUNTIME.md').read_text();block=re.search(r'```yaml\s*\n(.*?)\n```',text,re.S)
    if block is None:raise ValueError('Native previous registry missing')
    rows=yaml.safe_load(block.group(1))['skillruntimes']
    from fastrsi.native_goal_component_v1 import extract_component
    skills=[row for row in rows if 'skillruntime://'+row['id']==result['new_session']['planned']['skillruntime_ref']]
    if len(skills)!=1 or extract_component(skills[0])!=decision.get('declaration'):
        raise ValueError('Previous registered declaration differs from actual update')
    transactions=decision.get('transactions',[])
    if (len(transactions)!=1 or transactions[0].get('registry_after_sha256')!=registry_sha
            or transactions[0].get('declaration')!=decision['declaration']
            or transactions[0].get('candidate_id')!=decision.get('candidate')):
        raise ValueError('Previous native transaction does not bind actual registry')
    txn=transactions[0];before_sha=txn.get('registry_before_sha256');after_sha=txn['registry_after_sha256']
    if not isinstance(before_sha,str) or not re.fullmatch(r'[0-9a-f]{64}',before_sha):
        raise ValueError('Previous transaction before-registry SHA missing')
    initial_state=json.loads((workspace/base.STATE_REL).read_bytes())
    if initial_state.get('registry_sha256',{}).get('SKILLRUNTIME.md')!=before_sha:
        raise ValueError('Previous transaction before-registry differs from original current-state evidence')
    changed=before_sha!=after_sha;status=result['status']
    revoked=result.get('withdrawal_performed',False)
    if (type(result.get('registry_version_changed')) is not bool or result['registry_version_changed']!=changed
            or type(revoked) is not bool):
        raise ValueError('Previous outcome flags disagree with transaction SHA change')
    if status=='COMPLETED_EMPIRICAL_NO_CHANGE':
        valid=(txn.get('status')=='UNCHANGED' and txn.get('candidate_id')=='unchanged'
               and not changed and not revoked and update.get('candidate')=='unchanged'
               and not update.get('withdrawal_required'))
    elif status=='COMPLETED_EMPIRICAL_SCOPE_WITHDRAWAL':
        valid=(txn.get('status')=='COMPONENT_REMOVED' and txn.get('candidate_id')=='native'
               and decision['declaration'] is None and changed and revoked
               and (update.get('withdrawal_required') is True or update.get('candidate')=='native'))
    else:
        valid=(txn.get('status')=='UNVALIDATED_DRAFT_REGISTERED' and txn.get('candidate_id') not in ('native','unchanged')
               and decision['declaration'] is not None and changed and not revoked
               and update.get('candidate')==decision['candidate'] and not update.get('withdrawal_required'))
    if not valid:raise ValueError('Previous outcome status disagrees with actual native transaction')
    text=(workspace/'SESSIONS.md').read_text();block=re.search(r'```yaml\s*\n(.*?)\n```',text,re.S)
    if block is None or any(row.get('status') not in {'succeeded','failed','replanned','timed_out','cancelled','rejected'}
            for row in yaml.safe_load(block.group(1)).get('sessions',[])):
        raise ValueError('Previous workspace contains nonterminal native work')
    selection=result['new_session']
    if (selection.get('selected_declaration')!=decision['declaration'] or selection.get('registry_bytes_inherited') is not True
            or selection.get('model_calls')!=0 or selection.get('queued')!=1 or selection.get('task_executed') is not False
            or selection['planned']['benchmark'].get('instance_id') is not None
            or selection['planned']['execution'].get('replan_every_steps')!=10):
        raise ValueError('Previous fixed native consumer differs')
    receipts=result.get('common_protocol_receipts',[])
    if not receipts or receipts[0].get('session_mode')!='development' or receipts[0].get('data_role')!='DEVELOPMENT_INPUT':
        raise ValueError('Previous evaluation source cannot become development')
    old_rows,_=merge_observations([update['observations']])
    if update.get('evidence_sha256')!=digest(old_rows):raise ValueError('Previous evidence SHA does not match actual rows')
    actual={row['result_sha256']:row for row in observations}
    if any(actual.get(row['result_sha256'])!=row for row in old_rows):
        raise ValueError('Previous evidence must be independently revalidated in cumulative batches')
    return dict(workspace=workspace,manifest=manifest,update=update,result=result,bindings=copy.deepcopy(previous))


def stage_update_inputs(workspace,raw,package,model_sha,folder,batch_receipt):
    """Refresh known facts only; inherited source refs and other memory survive."""
    staging=folder/'staged_update_inputs';staging.mkdir()
    manifest,protected=base.stage_inputs(staging,raw)
    catalogue,catalogue_path=base.stage_successor_catalogue(staging,package,model_sha,manifest,protected)
    preserved_catalogues=[]
    for rel,blob in protected.items():
        target=Path(workspace)/rel
        if target.exists() and target.read_bytes()!=blob and rel not in (base.FEEDBACK_REL,base.MANIFEST_REL):
            # An old catalogue was rendered with the bound model on its first
            # staging. Preserve those exact inherited bytes, while the raw
            # package still contains its original placeholder and raw SHA.
            try:
                document=json.loads(blob)
                rendered=(json.dumps(base.finalize_semantics(document,model_sha),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
            except (ValueError,TypeError,UnicodeDecodeError):document={};rendered=None
            if (isinstance(document,dict) and set(document)=={'version','candidates'}
                    and document['version']=='native_goal_component_catalogue_v2'
                    and target.read_bytes()==rendered):
                protected[rel]=target.read_bytes()
                manifest['staged_sha256'][rel]=base.sha(protected[rel])
                preserved_catalogues.append(dict(path=rel,raw_package_sha256=base.sha(blob),
                    inherited_sha256=base.sha(protected[rel]),transformation='Exact bound MODEL_CONFIG placeholder rendering only; inherited bytes preserved'))
            else:raise ValueError('Inherited source/semantics/catalogue path changed: '+rel)
    if preserved_catalogues:
        manifest['inherited_catalogue_renderings']=preserved_catalogues
        base.save(staging/base.MANIFEST_REL,manifest)
        protected[base.MANIFEST_REL]=(staging/base.MANIFEST_REL).read_bytes()
    for rel,blob in protected.items():
        target=Path(workspace)/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(blob)
    batch_rel='configs/empirical_scope_update/BATCHES.json'
    base.save(Path(workspace)/batch_rel,batch_receipt);protected[batch_rel]=(Path(workspace)/batch_rel).read_bytes()
    return manifest,protected,catalogue,catalogue_path,batch_rel


async def run_update(campaign,output,binding_sha256):
    start=time.perf_counter();campaign=Path(campaign).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    result=dict(version=VERSION,status='RUNNING',update_origin='EMPIRICAL_DEVELOPMENT_FEEDBACK',
        model_calls=0,robot_episodes=0,policy_inference_calls=0,training_updates=0,
        empirical_admission=False,teacher_adopted=False,distillation_performed=False,learning_net_benefit=False)
    try:
        binding=checked_inputs(campaign,binding_sha256)
        raw=base.campaign_inputs(campaign);model_sha=base.sha(raw['MODEL_CONFIG.json'])
        common=json.loads((campaign/'COMMON_CASE.json').read_bytes())
        if common['model_config_sha256']!=model_sha:raise ValueError('Common model differs from input contract')
        observations,batches=development_batches(campaign,binding)
        if any(row['protocol']['model_config_sha256']!=model_sha for row in observations):
            raise ValueError('Accumulated feedback model differs from current common model')
        previous=previous_state(campaign,binding,observations)
        result['evidence_batches']=batches
        if previous is not None:result['previous_run']=previous['bindings']
        backend=base.load_backend(output,campaign)
        from PhyAgentOS.providers.base import LLMProvider
        class NoModel(LLMProvider):
            def get_default_model(self):return 'EMPIRICAL_UPDATE_NO_TEACHER'
            async def chat(self,*args,**kw):raise RuntimeError('No LLM calls authorized in empirical update')
        with common_case_protocol(backend,common) as protocol:
            folder=output/'update_session';folder.mkdir()
            case=dict(protocol.common,id='empirical_scope_update_root',mode='development')
            w,spec,_=backend.setup_workspace(folder,case,previous['workspace'] if previous else None)
            manifest,protected,catalogue,catalogue_path,batch_rel=stage_update_inputs(
                w,raw,campaign/'successor_package',model_sha,folder,batches)
            from probe_native_goal_learning_cycle_v1 import current_component_state,readonly_selection
            base.save(w/base.STATE_REL,current_component_state(w,spec,backend));protected[base.STATE_REL]=(w/base.STATE_REL).read_bytes()
            from fastrsi import native_goal_component_v1 as component
            anchor=backend.component_selection_anchor(w,spec)
            baseline=backend.read_yaml_block(w/'SKILLRUNTIME.md')
            incumbent_declaration=component.extract_component(base.selected_skill(w,spec,backend))
            snapshot_start=time.perf_counter()
            prior=output/'prior_workspace';shutil.copytree(w,prior)
            prior_manifest={p.relative_to(prior).as_posix():base.sha(p.read_bytes())
                            for p in sorted(prior.rglob('*')) if p.is_file()}
            base.save(output/'PRIOR_WORKSPACE_MANIFEST.json',prior_manifest)
            result['prior_snapshot']=dict(workspace=str(prior),registry_sha256=base.sha((prior/'SKILLRUNTIME.md').read_bytes()),
                manifest_sha256=base.sha((output/'PRIOR_WORKSPACE_MANIFEST.json').read_bytes()),
                seconds=time.perf_counter()-snapshot_start,scope='Exact pre-update common-information frozen control')
            matches=[key for key,value in catalogue.items() if value==incumbent_declaration]
            if incumbent_declaration is not None and len(matches)!=1:raise ValueError('Incumbent absent from empirical catalogue')
            incumbent=matches[0] if matches else 'native'
            if previous and incumbent_declaration!=previous['result']['decision']['declaration']:
                raise ValueError('Native setup lost the exact previous component')
            # Previous candidate may be unchanged: the persisted registry is the state.
            selector_previous=dict(previous['update'],candidate=incumbent) if previous else None
            update=select_bound_update(campaign,binding,observations,list(catalogue),incumbent=incumbent,previous=selector_previous)
            commit_candidate='native' if update.get('withdrawal_required') else update['candidate']
            base.save(output/'UPDATE.json',update)
            report_rel='configs/empirical_scope_update/UPDATE.json'
            (w/report_rel).parent.mkdir(parents=True,exist_ok=True)
            (w/report_rel).write_bytes((output/'UPDATE.json').read_bytes());protected[report_rel]=(w/report_rel).read_bytes()
            required=tuple(dict.fromkeys((*base.REQUIRED_READS,base.MANIFEST_REL,base.STATE_REL,catalogue_path,
                report_rel,batch_rel,*(value['execution_contract']['evidence']['path'] for value in catalogue.values() if 'execution_contract' in value))))
            tracker=base.CompleteReads(w,required);log=folder/'AGENT_EVENTS.jsonl'
            check=lambda:base.validate_draft(w,spec,anchor,case,baseline,backend)
            agent=backend.bind_agent(w,NoModel(),log,component_updates=True,max_iterations=1,read_files=True)
            original=base.install_proposal_tools(agent,w,log,tracker,protected,check,backend)
            restore,receipts=transaction.install(agent,w,spec,tracker,protected,check,backend,log,
                model_config_sha256=model_sha,candidate_catalogue=catalogue,catalogue_path=catalogue_path,
                allow_native_revoke=bool(update.get('withdrawal_required') or commit_candidate=='native'))
            try:
                from fastrsi.tool_origin import TOOL_ORIGIN
                token=TOOL_ORIGIN.set('empirical_scope_update')
                try:
                    pages=[]
                    # Keep every returned page within the original 16k read
                    # tracker. Larger raw-backed manifests need more small
                    # pages, with the same complete/contiguous/4MB checks.
                    for rel in required:await read_complete(agent.tools.execute,rel,on_page=pages.append,page_lines=32,max_pages=256)
                    outcome=await agent.tools.execute(transaction.TOOL_NAME,dict(candidate_id=commit_candidate))
                finally:TOOL_ORIGIN.reset(token)
                if outcome.startswith('Error'):raise ValueError('Original native transaction rejected: '+outcome)
                tracker.require()
                _,decl=check()
                expected_decl=(None if commit_candidate=='native' else incumbent_declaration
                               if commit_candidate=='unchanged' else catalogue[commit_candidate])
                if decl!=expected_decl:raise ValueError('Actual saved component differs from empirical choice')
                base.save(folder/'TRANSACTION.json',json.loads(outcome))
            finally:
                restore();agent.tools.execute=original;await backend.finish_agent(agent,log)
            result['update']=update
            result['decision']=dict(declaration=decl,registry_sha256=base.sha((w/'SKILLRUNTIME.md').read_bytes()),
                origin='EMPIRICAL_DEVELOPMENT_FEEDBACK_NOT_QWEN',workspace=str(w),
                candidate=commit_candidate,selector_candidate=update['candidate'],incumbent=incumbent,
                complete_reads=tracker.complete(),transactions=receipts)
            result['new_session']=await readonly_selection(output,campaign,protocol.common,w,protected,backend,
                lambda *a:(_ for _ in ()).throw(RuntimeError('No model provider allowed')),public_flow=True)
            selected=result['new_session']['selected_declaration']
            if selected!=decl:raise ValueError('New Session did not consume actual update')
            planned=result['new_session']['planned']
            if planned['execution']['replan_every_steps']!=10 or planned['benchmark']['instance_id'] is not None:
                raise ValueError('New Session must retain real H10/sampler protocol')
            if decl is not None:
                prepared=component.prepare_component(Path(result['new_session']['workspace']),decl,
                    expected_goal=spec['task_description'],expected_model_config_sha256=model_sha,expected_target_ref=spec['target_ref'])
                result['new_session']['program_steps_limit']=prepared['program_steps_limit']
                result['new_session']['prepared_source_sha256']=prepared['source_artifact_sha256']
            result['common_protocol_receipts']=protocol.receipts
            withdrawn=commit_candidate=='native' and incumbent_declaration is not None and decl is None
            result.update(status=('COMPLETED_EMPIRICAL_SCOPE_WITHDRAWAL' if withdrawn
                                  else 'COMPLETED_EMPIRICAL_SCOPE_UPDATE' if decl!=incumbent_declaration else 'COMPLETED_EMPIRICAL_NO_CHANGE'),
                withdrawal_performed=withdrawn,native_fallback_quality_certified=False,
                evaluation_updates_blocked=True,registry_version_changed=decl!=incumbent_declaration,
                original_campaign_inputs_unchanged=checked_inputs(campaign,binding_sha256)==binding)
            if previous and directory_manifest(previous['workspace'])!=previous['manifest']:
                raise ValueError('Original previous workspace changed')
    except Exception:result.update(status='FAILED',error=traceback.format_exc())
    result['full_seconds']=time.perf_counter()-start
    base.save(output/'RESULT.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--binding-sha256',required=True);args=parser.parse_args()
    r=asyncio.run(run_update(args.campaign,args.output,args.binding_sha256))
    print(json.dumps({k:r[k] for k in ('status','model_calls','robot_episodes','full_seconds')}),flush=True)
    return 0 if r['status'].startswith('COMPLETED_') else 1


if __name__=='__main__':raise SystemExit(main())
