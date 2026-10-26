"""Owned lane orchestration around exact frozen LIBERO10 model and Agent workers."""
import argparse,ast,fcntl,hashlib,importlib.util,json,os,re,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI/planning/libero_three_suites_20261024_success_v2')
PROJECT=ROOT.parent.parent
SUITE_LIMIT={'libero_spatial':220,'libero_object':280,'libero_goal':300,'libero_10':520}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.pending.'+str(os.getpid()));tmp.write_text(json.dumps(v,indent=2,default=str)+'\n');tmp.replace(p)
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def campaign_for(model,arm,suite,variant):
 source_group=model+'_'+('P' if arm=='P' else 'R');source=ROOT/'source_frozen'/source_group
 name='libero_original_best_20261024_'+model+'_'+arm+'_'+suite+'_'+variant
 campaign=ROOT/'campaigns'/name
 if not campaign.exists():
  campaign.mkdir(parents=True)
  for n in ('code','templates','vla_native'): (campaign/n).symlink_to(source/n,target_is_directory=True)
  for n in ('MODEL_CONFIG.json','VLA_CONFIG.json','FLOW.json','CODE_MANIFEST.json','CORE_BINDING.json'):shutil.copy2(source/n,campaign/n)
  shutil.copy2(source/'FROZEN_CONTROLLER_SOURCE.py',campaign/'FROZEN_CONTROLLER_SOURCE.py')
  cfg=read(campaign/'VLA_CONFIG.json')
  if model=='openvla':
   ck={'libero_spatial':'openvla_spatial','libero_object':'openvla_object','libero_goal':'openvla_goal','libero_10':'openvla_libero10'}[suite]
   cfg['checkpoint']=str(Path(cfg['checkpoint']).parent/ck);cfg['checkpoint_config_sha256']=sha(Path(cfg['checkpoint'])/'config.json')
   write(campaign/'VLA_CONFIG.json',cfg)
  protocol=dict(benchmark='LIBERO',base_suite=suite,perturbation=variant,model=model,arm=arm,registered_cases=500,source_group=source_group,source_controller_sha256=sha(source/'FROZEN_CONTROLLER_SOURCE.py'),asset_index_sha256=sha(ROOT/'ASSET_INDEX_FINAL.json'),learning='empty independent case workspace; no persistent updates or inheritance',max_steps=SUITE_LIMIT[suite],execution_steps=10,scope='Success-first v2 derived per-model retained R workers; max5 recovery children, official original-suite per-attempt limits. Selection experiment, not matched baseline or cross-episode learning.',technical_attempts='At most2 infrastructure attempts; max5 native recovery children on canonical failure; no post-hoc quality rerun.',report_repair='Exact v9 two-file overlay for pi05 P only')
  write(campaign/'PROTOCOL.json',protocol)
 # Host module resolves the symlink; container mounts frozen source directly at the expected code path.
 return campaign,source

def ports_for(gpu_base,slot_index=None):
 base=21100+100*(gpu_base//2 if slot_index is None else slot_index)
 return dict(zip(('policy','target','qwen','verifier'),(base,base+2,base+10,base+20)))

def adapt(module,source,campaign,model,arm,suite,variant,slot_index=None,resource_profile=None):
 tree=ast.parse((source/'FROZEN_CONTROLLER_SOURCE.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_lane');text=ast.get_source_segment((source/'FROZEN_CONTROLLER_SOURCE.py').read_text(),node)
 port_node=next(n for n in node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ports' for t in n.targets))
 original_ports=ast.get_source_segment((source/'FROZEN_CONTROLLER_SOURCE.py').read_text(),port_node)
 text=text.replace(original_ports,"ports = ports_for(placement['gpu_base'], placement.get('slot_index'))")
 module.ports_for=ports_for
 loop="for role, template in [('pi05', 'LAUNCH_pi05_v2.json'), ('qwen', 'LAUNCH_qwen.json'), ('target', 'LAUNCH_target.json')]:"
 assert loop in text
 text=text.replace(loop,loop+"\n            if arm == 'VLA' and role == 'qwen': continue")
 text=text.replace("for port in (ports['policy'], ports['qwen']):","for port in ((ports['policy'],) if arm == 'VLA' else (ports['policy'], ports['qwen'])):")
 text=text.replace("campaign / 'code/scripts/engine_overlay.py'","campaign / 'code/scripts/engine_overlay.py'")
 text=text.replace("state['cases'].append(row)","state['cases'].append(row)\n            if row['chain_status'] == 'PASSED':\n                seal_case(case_dir, item, row['host_case_seconds'])")
 module.seal_case=lambda d,item,seconds:metrics_for(d,model,arm,suite,variant,item,seconds)
 failure="if exit_code or result.get('chain_status') != 'PASSED':\n                raise RuntimeError('Paid case chain failure retained; pause lane for diagnosis')"
 assert failure in text
 text=text.replace(failure,"if exit_code or result.get('chain_status') != 'PASSED':\n                if not seal_agent_failure(case_dir, item, row['host_case_seconds']):\n                    raise RuntimeError('Paid case chain failure retained; pause lane for diagnosis')")
 module.seal_agent_failure=lambda d,item,seconds:agent_failure_for(d,model,arm,suite,variant,item,seconds)
 exec(compile(text,str(ROOT/'run_lane.py'),'exec'),module.__dict__)
 cbase="/project/runs/"+campaign.name+"/code"
 original_mounts=module.mounts_for_campaign
 def mounts(argv,camp):
  argv=original_mounts(argv,camp)
  # Core/library/weights remain immutable. The only writable locations belong to this campaign.
  pos=next(i for i,v in enumerate(argv) if v in (module.IMAGE,module.TARGET_IMAGE))
  cbase='/project/runs/'+camp.name+'/code'
  source_dirs=['-v',str(source/'code')+':'+cbase+':ro','-v',str(source/'vla_native')+':/project/runs/'+camp.name+'/vla_native:ro','-v',str(ROOT)+':/eval:ro','-e','LIBEROPRO_BASE_SUITE='+suite,'-e','LIBEROPRO_VARIANT='+variant]
  # All external environment identity is captured in PROTOCOL and per-case receipts.
  if model=='pi05' and arm=='P':
   for rel in ('scripts/qwen_agent_server.py','src/fastrsi/terminal_results.py'):
    source_dirs+=['-v',str(ROOT/'report_repair_overlay/code'/rel)+':'+cbase+'/'+rel+':ro']
  source_dirs += ['-v', '/mnt/data/task/yuchen/FastPhyagentos/frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1/dependencies/openpi/third_party/libero' + ':/eval/upstream:ro']
  argv[pos:pos]=source_dirs
  if module.TARGET_IMAGE in argv:
   # Target and pure runner import the isolated official PRO environment package.
   argv=[('PYTHONPATH=/eval/runtime:/eval/upstream:'+v[len('PYTHONPATH='):]) if v.startswith('PYTHONPATH=') else ('LIBERO_CONFIG_PATH=/eval/libero-config') if v.startswith('LIBERO_CONFIG_PATH=') else v for v in argv]
   pos=argv.index(module.TARGET_IMAGE);argv[pos:pos]=['-e','LIBEROPRO_BIND_ENV=1','-e','LIBERO_CONFIG_PATH=/eval/libero-config']
  return argv
 module.mounts_for_campaign=mounts
 def live_check():
  # A stop blocks new launches, but already launched owned runners drain to their normal deadline.
  if (ROOT/'HALT.json').exists() and sys._getframe(1).f_code.co_name!='wait_exit':raise RuntimeError('Campaign halted; no new case dispatch')
 module.validate_group=live_check
 oldlaunch=module.launch
 def launch(argv,owner,role,prefix):
  if role in ('pi05','qwen','target') and 'timeout' in argv:
   argv[argv.index('timeout')+1]=str(500*3600+7200)
  if role=='pi05':
   # No change to model weights or generation; OpenVLA normalization follows its base-suite checkpoint.
   argv[-1]=argv[-1].replace(cbase+'/scripts/multi_vla_server.py','/eval/runtime/liberopro_vla_server.py')
   if 'PYTHONPATH=' not in '\n'.join(argv):raise RuntimeError('Policy path missing')
   argv=[('PYTHONPATH=/eval/runtime:'+v[len('PYTHONPATH='):]) if v.startswith('PYTHONPATH=') else v for v in argv]
   pos=argv.index(module.IMAGE);argv[pos:pos]=['-e','LIBEROPRO_CAMPAIGN_CODE='+cbase]
  if role=='qwen':
   assigned={0:2,1:3,2:2,3:3,4:6,5:7,6:6,7:7,8:int(os.environ.get('RESOURCE06_BACKFILL_GPU','6'))}[slot_index]
   argv=[('HIP_VISIBLE_DEVICES='+str(assigned)) if v.startswith('HIP_VISIBLE_DEVICES=') else ('CUDA_VISIBLE_DEVICES='+str(assigned)) if v.startswith('CUDA_VISIBLE_DEVICES=') else v for v in argv]
  if role=='runner':
   argv=[('FASTRSI_RESIDENT_MAX_INFERENCES=2000' if v.startswith('FASTRSI_RESIDENT_MAX_INFERENCES=') else v) for v in argv]
   pos=argv.index(module.IMAGE)
   argv[pos:pos]=['-v',str(ROOT/'resource05_overlay'/model/'run_native_baseline.py')+':'+cbase+'/scripts/run_native_baseline.py:ro']
   argv=[('SYSTEM_NATIVE_FAILURE_RETRY=1' if v.startswith('SYSTEM_NATIVE_FAILURE_RETRY=') else v) for v in argv]
   case=read(Path(owner)/'ROSTER.json')[0]
   asset=next(r for r in read(ROOT/'ASSET_INDEX_FINAL.json') if r['suite']==suite and r['variant']==variant and r['task_id']==case['task_id'])
   write(Path(owner)/'BENCHMARK_BINDING.json',dict(benchmark='LIBERO',base_suite=suite,variant=variant,task_id=case['task_id'],init_state_id=case['init_state_id'],task_language=case['task_description'],asset=asset,asset_index_sha256=sha(ROOT/'ASSET_INDEX_FINAL.json'),model_config_sha256=sha(campaign/'MODEL_CONFIG.json'),vla_config_sha256=sha(campaign/'VLA_CONFIG.json'),runtime_adapter_sha256={p.name:sha(p) for p in (ROOT/'runtime').glob('*.py')},lane_adapter_sha256=sha(ROOT/'run_lane_backfill_resource06.py'),io_adaptation='Owned Session ENOENT or missing fenced YAML bounded read retry; original parser/model/execution/verification/report unchanged'))
   if slot_index is not None:
    binding=read(Path(owner)/'BENCHMARK_BINDING.json')
    binding['resource_profile']=dict(version=resource_profile or 'boost_10slots_v1',logical_slot=slot_index,co_location=(arm=='VLA' or slot_index>=4) if resource_profile=='vla_parallel32_v1' else ((arm!='VLA' and slot_index>=4) if resource_profile=='vla_single_gpu_v1' else slot_index>=4), gpu_layout=('two_policies_per_gpu' if arm=='VLA' else 'paired_policy_qwen') if resource_profile=='vla_parallel32_v1' else (('one_gpu_per_vla_lane' if arm=='VLA' else 'paired_policy_qwen') if resource_profile=='vla_single_gpu_v1' else 'historical_even_gpu_pair'), co_location_scope='Planned full concurrency; actual overlap and warmup from container timestamps and controller expansion receipt',scope='Additional independent registered group; dedicated model instances; original model generation and case protocol unchanged. Time includes contention.')
    write(Path(owner)/'BENCHMARK_BINDING.json',binding)
   worker='/eval/runtime/liberopro_pure_vla.py' if arm=='VLA' else '/eval/runtime/liberopro_case_worker.py'
   argv[-1]=argv[-1].replace(cbase+'/scripts/system_case_worker.py',worker)
   argv=[('PYTHONPATH=/eval/runtime:'+v[len('PYTHONPATH='):]) if v.startswith('PYTHONPATH=') else v for v in argv]
   pos=argv.index(module.IMAGE);argv[pos:pos]=['-e','LIBEROPRO_CAMPAIGN_CODE='+cbase,'-e','LIBERO_CONFIG_PATH=/eval/libero-config','-e','LIBEROPRO_BASE_SUITE='+suite,'-e','LIBEROPRO_VARIANT='+variant]
   if arm=='VLA':
    # The pure runner uses the frozen simulation dependencies, rather than host torch.
    argv[argv.index(module.IMAGE)]=module.TARGET_IMAGE
    argv[-1]='exec python '+worker
    argv=[('PYTHONPATH=/eval/runtime_vendor:/eval/runtime:/eval/upstream:/core:'+cbase+'/scripts:'+cbase+'/src') if v.startswith('PYTHONPATH=') else v for v in argv]
    pos=argv.index(module.TARGET_IMAGE);argv[pos:pos]=['-e','LIBEROPRO_BIND_ENV=1','-e','MUJOCO_GL=egl','-e','LIBGL_ALWAYS_SOFTWARE=1','-e','FASTRSI_TRACE='+str(Path('/project/runs/'+campaign.name)/Path(owner).relative_to(campaign)/'stream/OS_COST_EVENTS.jsonl')]
   else:
    # The exact existing P/R worker sets this internally; keep its common native model ceiling40.
    pass
  image_pos=next(i for i,v in enumerate(argv) if v in (module.IMAGE,module.TARGET_IMAGE))
  assert argv[image_pos+1] in ('timeout','python','bash'), ('Invalid container entry',argv[image_pos:image_pos+4])
  return oldlaunch(argv,owner,role,prefix)
 module.launch=launch

def agent_failure_for(case_dir,model,arm,suite,variant,item,host_seconds):
 result=case_dir/'stream'/('system_t%d_i%d'%(item['task_id'],item['init_state_id']))/'RESULT.json'
 if not result.exists() or arm=='VLA':return False
 row=read(result);error=row.get('error') or ''
 ledger=case_dir/'stream/OS_COST_EVENTS.jsonl'
 events=[json.loads(s) for s in ledger.read_text().splitlines() if s.strip()] if ledger.exists() else []
 # Only already observed, closed Agent planning protocol failures qualify.
 # Infrastructure and unclassified exceptions still stop; measured Agent failures retain native outcomes.
 failure_kind=None
 if 'AssertionError: Frozen execution fields differ:' in error:failure_kind='AGENT_INVALID_SESSION_BEFORE_DISPATCH'
 elif 'AssertionError: Native benchmarking skill was not read' in error:failure_kind='AGENT_REQUIRED_SKILL_NOT_READ_BEFORE_DISPATCH'
 elif ('pydantic_core._pydantic_core.ValidationError:' in error and 'validation error for SessionSpec' in error
       and 'type=extra_forbidden' in error and 'in effective_execution_view' in error):
  failure_kind='AGENT_SESSION_EXTRA_FIELD_BEFORE_DISPATCH'
 qualifies=(row.get('chain_status')=='FAILED' and failure_kind is not None
  and 'in plan_session' in error and 'in validate_queued_session' in error
  and row.get('planned_session') is None and row.get('environment_success') is None and not row.get('outcomes')
  and row.get('accepted_source') is None and row.get('experience_before')==row.get('experience_after')
  and any(e.get('event')=='end' and e.get('name')=='native_agent_planning' for e in events)
  and not any(e.get('event')=='begin' and e.get('name') in ('os_execution','policy_client','verification_model') for e in events))
 # A terminal Agent response that fails to write its required report is a
 # measured system failure after completed execution, rather than an IO retry.
 if not qualifies and error.rstrip().endswith('ValueError: No fresh benchmark report write observed') and 'in consume_terminal' in error:
  begins={e['span_id']:e for e in events if e.get('event')=='begin' and e.get('span_id')}
  ends={e['span_id']:e for e in events if e.get('event')=='end' and e.get('span_id')}
  relevant=[e for e in begins.values() if e.get('name') in ('os_execution','verification_model')]
  canonical=case_dir/'stream'/row['case']['id']/'CANONICAL_TERMINAL_SNAPSHOT.json'
  sessions=read(canonical).get('sessions',[]) if canonical.exists() else []
  episodes=[]
  for session in sessions:
   artifact=session.get('result',{}).get('artifact_dir')
   if not artifact:return False
   episode=case_dir/'stream'/row['case']['id']/'workspace'/artifact/'episode.json'
   if not episode.exists():return False
   episodes.append(read(episode))
  qualifies=(row.get('chain_status')=='FAILED' and type(row.get('environment_success')) is bool
   and row.get('planned_session') is not None and row.get('accepted_source') is None
   and row.get('experience_before')==row.get('experience_after') and bool(row.get('outcomes'))
   and all(x.get('ok') is True and x.get('code')=='verified' for x in row['outcomes'])
   and bool(episodes) and episodes[-1].get('success') is row['environment_success']
   and any(e.get('name')=='os_execution' for e in relevant)
   and any(e.get('name')=='verification_model' for e in relevant)
   and all(ends.get(e['span_id'],{}).get('status')=='ok' for e in relevant)
   and any(e.get('event')=='end' and e.get('name')=='native_terminal_report' and e.get('status')=='error' for e in events))
  failure_kind='AGENT_REPORT_NOT_WRITTEN_AFTER_VERIFICATION'
 # An Agent-authored extra Session field can terminate the background verifier
 # while the already-dispatched native episode completes. Account both layers.
 # The same schema failure also arrives wrapped by execute_runtime, without
 # the nested verifier traceback. Require its exact document path and wrapper.
 verifier_origin=('in first_awaiting_verification' in error or
  ('in execute_runtime' in error and
   ('Background verifier exited: invalid sessions document '+str(case_dir/'stream'/row['case']['id']/'workspace/SESSIONS.md').replace(str(ROOT/'campaigns'),'/project/runs')+': 1 validation error for SessionsDocument') in error))
 if not qualifies and verifier_origin and all(t in error for t in ('validation error for SessionsDocument', 'sessions.0.construction', 'type=extra_forbidden', 'Background verifier exited:')):
  begins={e['span_id']:e for e in events if e.get('event')=='begin' and e.get('span_id')}
  ends={e['span_id']:e for e in events if e.get('event')=='end' and e.get('span_id')}
  executions=[e for e in begins.values() if e.get('name')=='os_execution']
  canonical=case_dir/'stream'/row['case']['id']/'CANONICAL_TERMINAL_SNAPSHOT.json'
  sessions=read(canonical).get('sessions',[]) if canonical.exists() else []
  episodes=[]
  for session in sessions:
   artifact=session.get('result',{}).get('artifact_dir')
   if not artifact:return False
   episode=case_dir/'stream'/row['case']['id']/'workspace'/artifact/'episode.json'
   if not episode.exists():return False
   episodes.append(read(episode))
  authored=False
  for response in (case_dir/'stream').rglob('RESPONSE.json'):
   if not response.parent.name.startswith('agent_call_'):continue
   for call in read(response).get('tool_calls') or []:
    args=call.get('arguments') or {}
    if not isinstance(args,dict):continue
    payload=args.get('item') if call.get('name')=='append_yaml_item' else args.get('patch') if call.get('name')=='patch_yaml_item' else None
    if (str(args.get('path','')).endswith('/'+row['case']['id']+'/workspace/SESSIONS.md')
        and args.get('list_key')=='sessions' and isinstance(payload,dict)
        and isinstance(payload.get('construction'),str) and payload['construction']):authored=True
  qualifies=(row.get('chain_status')=='FAILED' and type(row.get('environment_success')) is bool
   and row.get('planned_session') is not None and row.get('accepted_source') is None
   and row.get('experience_before')==row.get('experience_after') and not row.get('outcomes')
   and authored and bool(episodes) and episodes[-1].get('success') is row['environment_success']
   and bool(executions) and all(ends.get(e['span_id'],{}).get('status')=='ok' for e in executions)
   and any(e.get('event')=='end' and e.get('name')=='native_agent_planning' and e.get('status')=='ok' for e in events)
   and not any(e.get('name') in ('verification_model','native_terminal_report') for e in begins.values()))
  failure_kind='AGENT_SESSION_EXTRA_FIELD_VERIFIER_ABORT_AFTER_DISPATCH'
 if not qualifies:return False
 value=metrics_for(case_dir,model,arm,suite,variant,item,host_seconds,agent_failure=True,failure_kind=failure_kind)
 return True

def metrics_for(case_dir,model,arm,suite,variant,item,host_seconds,agent_failure=False,failure_kind=None):
 result_file=case_dir/'stream'/('system_t%d_i%d'%(item['task_id'],item['init_state_id']))/'RESULT.json'
 if not result_file.exists():raise RuntimeError('No native RESULT: '+str(case_dir))
 row=read(result_file);assert row.get('chain_status')==('FAILED' if agent_failure else 'PASSED'),row.get('isolation_error') or row.get('error') or row.get('chain_status')
 if arm!='VLA':assert row['accepted_source'] is None and row['experience_before']==row['experience_after']
 osfile=case_dir/'stream/OS_COST_EVENTS.jsonl';events=[json.loads(s) for s in osfile.read_text().splitlines() if s.strip()] if osfile.exists() else []
 starts={e['span_id']:e for e in events if e.get('event')=='begin' and e.get('span_id')}
 agents=[e for e in starts.values() if e.get('name')=='agent_model_call'];verifiers=[e for e in starts.values() if e.get('name')=='verification_model']
 def phase(e):
  parent=e.get('parent_id');seen=set()
  while parent in starts and parent not in seen:
   seen.add(parent);x=starts[parent]
   if x.get('name') in ('native_agent_planning','native_terminal_report'):return x['name']
   parent=x.get('parent_id')
  return 'unclassified'
 phases=[phase(e) for e in agents];usages=[]
 for path in (case_dir/'stream').rglob('RESPONSE.json'):
  if path.parent.name.startswith('agent_call_'):
   usage=read(path).get('usage');usages.append(usage)
 completions=[e.get('usage') for e in events if e.get('event')=='engine_completion']
 usages.extend(completions)
 def usage_valid(u):return isinstance(u,dict) and all(type(u.get(k)) is int and u[k]>=0 for k in ('prompt_tokens','completion_tokens','total_tokens')) and u['prompt_tokens']+u['completion_tokens']==u['total_tokens']
 valid=all(usage_valid(u) for u in usages)
 measured_usages=[u for u in usages if usage_valid(u)]
 usage_complete=valid and len(usages)==len(agents)+len(verifiers)
 # A closed, exact model tool-protocol rejection can omit usage in the API error.
 # Keep that call/time, explicit unknown tokens and a measured lower bound; never impute zero.
 known_protocol_rejection=False
 if agent_failure and not usage_complete and failure_kind in ('AGENT_REQUIRED_SKILL_NOT_READ_BEFORE_DISPATCH','AGENT_REPORT_NOT_WRITTEN_AFTER_VERIFICATION'):
  responses=[read(p) for p in (case_dir/'stream').rglob('RESPONSE.json') if p.parent.name.startswith('agent_call_')]
  missing=[r for r in responses if not usage_valid(r.get('usage'))]
  ended={e.get('span_id') for e in events if e.get('event')=='end' and e.get('name')=='agent_model_call' and e.get('status')=='ok'}
  known_protocol_rejection=(len(responses)==len(agents) and len(usages)==len(agents)+len(verifiers) and bool(missing)
   and all(e.get('span_id') in ended for e in agents)
   and all(r.get('finish_reason')=='error' and ('Fenced tool example is not executable' if failure_kind=='AGENT_REQUIRED_SKILL_NOT_READ_BEFORE_DISPATCH' else 'Native checkpoint call requires a closed envelope') in (r.get('content') or '') and 'tool_protocol_error' in (r.get('content') or '') for r in missing))
  if known_protocol_rejection and failure_kind=='AGENT_REQUIRED_SKILL_NOT_READ_BEFORE_DISPATCH':failure_kind='AGENT_TOOL_PROTOCOL_ERROR_BEFORE_DISPATCH'
 target=case_dir/'TARGET_COST_EVENTS.jsonl';t=[json.loads(s) for s in target.read_text().splitlines() if s.strip()] if target.exists() else []
 observed=[e for e in t if e.get('event')=='native_observation']
 after_execution_failure=agent_failure and failure_kind in ('AGENT_REPORT_NOT_WRITTEN_AFTER_VERIFICATION','AGENT_SESSION_EXTRA_FIELD_VERIFIER_ABORT_AFTER_DISPATCH')
 physical=row['environment_success'];assert (physical is None) if agent_failure and not after_execution_failure else (type(physical) is bool)
 canonical=case_dir/'stream'/row['case']['id']/'CANONICAL_TERMINAL_SNAPSHOT.json'
 sessions=read(canonical)['sessions'] if canonical.exists() else []
 native_attempts=[]
 for s in sessions:
  if agent_failure and not after_execution_failure:continue
  p=case_dir/'stream'/row['case']['id']/'workspace'/s['result']['artifact_dir']/'episode.json'
  native_attempts.append(read(p))
 if arm=='VLA':native_attempts=row['benchmark']['episodes']
 value=dict(version='liberopro_case_metrics_v1',model=model,arm=arm,suite=suite,variant=variant,task_id=item['task_id'],init_state_id=item['init_state_id'],host_case_seconds=host_seconds,native_case_seconds=row.get('seconds'),physical_first_success=native_attempts[0]['success'] if native_attempts else None,physical_final_success=physical,semantic_outcomes=row.get('outcomes'),semantic_first_success=((row.get('outcomes') or [{}])[0].get('verdict')=='success') if row.get('outcomes') else None,semantic_final_success=((row.get('outcomes') or [{}])[-1].get('verdict')=='success') if row.get('outcomes') else None,attempts=len(native_attempts),action_steps=sum(int(x.get('num_steps',0)) for x in native_attempts),planning_model_calls=phases.count('native_agent_planning'),report_model_calls=phases.count('native_terminal_report'),agent_model_calls=len(agents),verifier_model_calls=len(verifiers),qwen_token_usage_complete=valid and len(usages)==len(agents)+len(verifiers),qwen_prompt_tokens=sum(u['prompt_tokens'] for u in usages) if valid and len(usages)==len(agents)+len(verifiers) else None,qwen_completion_tokens=sum(u['completion_tokens'] for u in usages) if valid and len(usages)==len(agents)+len(verifiers) else None,qwen_total_tokens=sum(u['total_tokens'] for u in usages) if valid and len(usages)==len(agents)+len(verifiers) else None,vla_client_call_intents=sum(e.get('name')=='policy_client' for e in starts.values()),raw_result=str(result_file),result_sha256=sha(result_file),os_ledger_sha256=sha(osfile) if osfile.exists() else None,cost_scope='Case host and native time; service startup/shutdown accounted separately in lane STATUS. All technical attempts retained.')
 value.update(native_executed=bool(native_attempts),end_to_end_success=False if agent_failure else physical,agent_failure=agent_failure,failure_kind=(failure_kind or 'AGENT_INVALID_SESSION_BEFORE_DISPATCH') if agent_failure else None,raw_chain_status=row.get('chain_status'),raw_failure_error=row.get('error') if agent_failure else None)
 value.update(verification_protocol_completed=row.get('evidence_layers',{}).get('verification_protocol_completed'))
 # Unmeasured native success remains null; the end-to-end Agent outcome is a measured failure.
 value.update(qwen_token_covered_calls=len(measured_usages),qwen_token_missing_calls=len(agents)+len(verifiers)-len(measured_usages),qwen_observed_token_lower_bound=sum(u['total_tokens'] for u in measured_usages),token_accounting_note='Exact model tool-protocol rejection omitted usage; unknown call tokens retained, observed total is a lower bound.' if known_protocol_rejection else None)
 if agent_failure and not value['qwen_token_usage_complete'] and not known_protocol_rejection:raise RuntimeError('Incomplete token cost for unclassified Agent failure')
 write(case_dir/'METRICS.json',value);write(case_dir/'TERMINAL_CASE.json',{**value,'status':'AGENT_FAILED' if agent_failure else 'RUN_COMPLETED','cost_stream_closed':True,'completed_at':time.time()})
 return value

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--arm',required=True);ap.add_argument('--suite',required=True);ap.add_argument('--variant',required=True);ap.add_argument('--node',required=True,choices=('107','108','125','126','161','187','188'));ap.add_argument('--gpu-base',required=True,type=int);ap.add_argument('--stage',required=True,choices=('smoke','full'));ap.add_argument('--slot-index',type=int,choices=range(16));ap.add_argument('--resource-profile');ap.add_argument('--skip-busy-group',action='store_true');args=ap.parse_args()
 assert args.model in ('pi0','pi05','openvla','xvla') and args.arm in ('VLA','P','R') and args.suite in SUITE_LIMIT
 assert args.gpu_base in (range(8) if args.resource_profile=='whole_gpu_eight_lanes_resource05' or args.arm=='VLA' and args.resource_profile in ('vla_single_gpu_v1','vla_parallel32_v1') else (0,2,4,6))
 lease_name='node'+args.node+('_slot'+str(args.slot_index) if args.slot_index is not None else '')+'_gpu'+str(args.gpu_base)+'.lock'
 lease=ROOT/'leases'/lease_name;lease.parent.mkdir(exist_ok=True);handle=lease.open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
 group=ROOT/'group_status'/('_'.join((args.model,args.arm,args.suite,args.variant))+'.json');group.parent.mkdir(exist_ok=True)
 group_lease=ROOT/'leases'/('_'.join((args.model,args.arm,args.suite,args.variant))+'.lock');group_handle=group_lease.open('a+')
 while True:
  if (ROOT/'HALT.json').exists():raise RuntimeError('Campaign halted; no new group dispatch')
  try:fcntl.flock(group_handle,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:
   if args.skip_busy_group:
    print(json.dumps(dict(status='GROUP_BUSY_SKIPPED',group=group.stem)));return
   time.sleep(2)
 if group.exists():
  history=ROOT/'group_status/history';history.mkdir(exist_ok=True);shutil.copy2(group,history/(group.stem+'_'+str(time.time_ns())+'.json'))
 c,source=campaign_for(args.model,args.arm,args.suite,args.variant)
 index=[r for r in read(ROOT/'ASSET_INDEX_FINAL.json') if r['suite']==args.suite and r['variant']==args.variant];assert len(index)==10
 cases=[dict(id='system_t%d_i%d'%(r['task_id'],i),task_id=r['task_id'],init_state_id=i,task_description=r['language'],suite=args.suite,max_steps=r['max_steps']) for i in range(50) for r in index]
 if args.stage=='smoke':cases=cases[:1]
 done={(d['task_id'],d['init_state_id']) for p in c.glob('node*/lane*/case*/TERMINAL_CASE.json') for d in [read(p)] if d['status'] in ('RUN_COMPLETED','AGENT_FAILED')}
 cases=[x for x in cases if (x['task_id'],x['init_state_id']) not in done]
 for item in cases:
  previous=list(c.glob('node*/lane*/case*_t%d_i%d'%(item['task_id'],item['init_state_id'])))
  assert len(previous)<2,('Technical-attempt bound reached; retain failed attempts',args.model,args.arm,args.suite,args.variant,item['task_id'],item['init_state_id'],len(previous))
 if not cases:write(group,dict(status='COMPLETED',stage=args.stage,completed=len(done)));return
 module=load(source/'FROZEN_CONTROLLER_SOURCE.py','source_lane');adapt(module,source,c,args.model,args.arm,args.suite,args.variant,args.slot_index,args.resource_profile)
 # New private namespace locks and owned IDs; no old campaign controller or services are modified.
 anchor=PROJECT/'runs'/c.name;anchor.mkdir(exist_ok=True)
 (c/('node'+args.node)).mkdir(exist_ok=True)
 attempt=sum(1 for _ in c.glob('node*/lane*'));lane='lane%d_%s_attempt%03d'%(args.gpu_base//2 if args.slot_index is None else args.slot_index,args.stage,attempt)
 placement=dict(host=args.node,lane=lane,arm=args.arm,gpu_base=args.gpu_base,cases=cases)
 if args.slot_index is not None:placement['slot_index']=args.slot_index
 write(group,dict(status='RUNNING',stage=args.stage,model=args.model,arm=args.arm,suite=args.suite,variant=args.variant,node=args.node,gpu_base=args.gpu_base,pid=os.getpid(),lane=lane,planned=len(cases),completed_before=len(done),started_at=time.time()))
 try:
  result=module.run_lane(c,placement);write(c/('LANE_RESULT_'+lane+'.json'),result)
  if result.get('status')!='COMPLETED':raise RuntimeError(result.get('error') or result.get('status'))
  case_dirs=sorted((c/('node'+args.node)/lane).glob('case*'))
  outputs=[]
  for d,item in zip(case_dirs,cases):
   st=read(d/'STOP_RECEIPT.json');assert all(not x['running'] for x in st)
   rr=read(c/('node'+args.node)/lane/'STATUS.json');rec=next(x for x in rr['cases'] if x['task_id']==item['task_id'] and x['init_state_id']==item['init_state_id'])
   outputs.append(read(d/'METRICS.json') if (d/'METRICS.json').exists() else metrics_for(d,args.model,args.arm,args.suite,args.variant,item,rec['host_case_seconds']))
  assert len(outputs)==len(cases)
  write(group,dict(status='COMPLETED',stage=args.stage,model=args.model,arm=args.arm,suite=args.suite,variant=args.variant,node=args.node,gpu_base=args.gpu_base,pid=os.getpid(),lane=lane,completed=len(outputs)+len(done),finished_at=time.time(),lane_result=str(c/('LANE_RESULT_'+lane+'.json'))))
 except BaseException:
  write(group,dict(status='FAILED',stage=args.stage,model=args.model,arm=args.arm,suite=args.suite,variant=args.variant,node=args.node,gpu_base=args.gpu_base,pid=os.getpid(),lane=lane,error=traceback.format_exc(),finished_at=time.time()));raise
 finally:handle.close()
if __name__=='__main__':main()
