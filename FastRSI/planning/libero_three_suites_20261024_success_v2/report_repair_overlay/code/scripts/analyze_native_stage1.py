"""Engineering acceptance and paid bill; does not claim FastRSI efficacy."""
import argparse
import hashlib
import json
from pathlib import Path
from summarize_native_binding import summarize

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='native_stage1_v1_20261002')
parser.add_argument('--out',default='planning/native_stage1_v1/ACCEPTANCE.json');args=parser.parse_args()
run=root/'runs'/args.run_id
result=json.loads((run/'DECODER_RESULT.json').read_text());pairs=[]
for label in [x['label'] for x in result['summary']]:
    group=[x for x in result['rows'] if x['label']==label]
    last={v:[x for x in group if x['variant']==v][-1] for v in ['reference','optimized']}
    secs={v:x['receipt']['generation_seconds'] for v,x in last.items()}
    pairs.append({'label':label,'same_tokens_all_four':all(x['token_ids']==group[0]['token_ids'] for x in group),
        'final_pair_generation_seconds':secs,'relative_reduction':1-secs['optimized']/secs['reference'],
        'final_pair_calls':{v:x['call'] for v,x in last.items()},
        'final_pair_components':{v:x['receipt']['structured_decode'] for v,x in last.items()}})
native=json.loads((run/'stream/RESULT.json').read_text())
recovery=root/'runs/native_workflow_v2_20261002'
memory_path=recovery/'stream/terminal_memory/RESULT.json'
memory=json.loads(memory_path.read_text()) if memory_path.exists() else {'status':'NOT_COMPLETED'}
decoder_pass=all(x['same_tokens_all_four'] and x['relative_reduction']>0 for x in pairs)
source=(run/'MODEL_CONFIG.json').read_text()
workflow_run=root/'runs/native_workflow_v3_20261002'
terminal_run=root/'runs/native_terminal_eval_v2_20261002'
queue_run=root/'runs/native_queue_receipt_v1_20261002'
paid_runs=[run,root/'runs/native_workflow_v1_20261002',root/'runs/native_report_guard_cpu_v1_20261002',recovery,workflow_run,
    root/'runs/native_terminal_eval_v1_20261002',terminal_run,queue_run]
bills=[summarize(p) for p in paid_runs]
cost_keys=['model_requests','incomplete_model_requests','full_wall_seconds','generation_seconds','service_call_seconds','model_calls','model_rejected','prompt_tokens','completion_tokens','hcu_allocation_seconds','cpu_quota_seconds']
failed_memory_requests=sorted((root/'runs/native_workflow_v1_20261002/stream/terminal_memory').glob('memory_call_*/REQUEST.json'))
repaired_memory_requests=sorted((recovery/'stream/terminal_memory').glob('memory_call_*/REQUEST.json'))
same_memory_request=(json.loads(failed_memory_requests[0].read_text())==json.loads(repaired_memory_requests[0].read_text())) if failed_memory_requests and repaired_memory_requests else None
memory_services=[]
for path in [root/'runs/native_workflow_v1_20261002',recovery]:
    request=next(p for p in (path/'stream/service_calls').glob('*/REQUEST.json') if {t.get('function',{}).get('name') for t in json.loads(p.read_text()).get('tools',[])}=={'save_memory'})
    memory_services.append({'prompt_sha256':hashlib.sha256((request.parent/'MODEL_PROMPT.txt').read_bytes()).hexdigest(),
        'receipt':json.loads((request.parent/'RECEIPT.json').read_text())})
workflow_path=workflow_run/'stream/RESULT.json'
workflow=json.loads(workflow_path.read_text()) if workflow_path.exists() else {'status':'NOT_COMPLETED'}
workflow_pairs=[]
for n in range(3):
    matched={r['workflow']:r for r in workflow.get('rows',[]) if r['label'].startswith('context_'+str(n)+'_')}
    if len(matched)!=2:continue
    a,b=matched['serial'],matched['batch_reads']
    if a['status']!='PASSED' or b['status']!='PASSED':
        workflow_pairs.append({'context':n,'comparison_accepted':False,'statuses':{k:r['status'] for k,r in matched.items()},'seconds':{k:r['seconds'] for k,r in matched.items()}})
        continue
    workflow_pairs.append({'context':n,'same_normalized_session':a['normalized_plan']==b['normalized_plan'],
        'fixed_experience_unchanged':a['fixed_experience_unchanged'] and b['fixed_experience_unchanged'],
        'seconds':{k:r['seconds'] for k,r in matched.items()},
        'model_calls':{k:r['model_calls'] for k,r in matched.items()},
        'tool_calls_per_model_call':{k:r['tool_calls_per_model_call'] for k,r in matched.items()},
        'planning_window_reduction':1-b['seconds']/a['seconds']})
terminal=json.loads((terminal_run/'stream/RESULT.json').read_text())
terminal_case=terminal['cases'][0]
queue_replay=json.loads((queue_run/'stream/RESULT.json').read_text())
terminal_pass=terminal['status']=='COMPLETED' and terminal_case['terminal_report']['status']=='PASSED' and terminal_case['experience_before']==terminal_case['experience_after']
value={'status':'COMPONENTS_VERIFIED_WITH_ONE_RECOVERED_NATIVE_EVALUATION' if native['status']=='COMPLETED' and decoder_pass and memory.get('source_unchanged_after') is True and memory['status']=='PASSED' and terminal_pass and queue_replay['status']=='PASSED_ONE_SAVED_INPUT' else 'PARTIAL',
    'scope':'Public decoder and one native terminal-consumption/memory lifecycle acceptance; not FastRSI effectiveness, population success or full-task acceleration.',
    'run_id':args.run_id,'decoder_pairs':pairs,'decoder_optimized_candidate_passed':decoder_pass,
    'decoder_exact_masks':json.loads((run/'DECODER_EQUIVALENCE.json').read_text()),
    'native_cases':native['cases'],'native_memory':memory,'cost_bill':summarize(run),
    'recovery_cost_bill':summarize(recovery),
    'all_paid_engineering_runs':bills,'all_paid_engineering_totals':{k:sum(r.get(k,0) for r in bills) for k in cost_keys},
    'accounting':'Campaign interval sum, not natural elapsed time. Model spans overlap HCU residency. Capacity-time is not utilization. Interrupted request generation is unknown. Local edits, transfers and human time are unmeasured.',
    'memory_retry_same_raw_request':same_memory_request,
    'memory_retry_same_compiled_prompt':memory_services[0]['prompt_sha256']==memory_services[1]['prompt_sha256'],
    'memory_retry_service_evidence':memory_services,
    'fixed_workflow_probe':workflow,'workflow_pairs':workflow_pairs,
    'latest_native_evaluation':terminal,'accepted_queue_replay':queue_replay,
    'native_evaluation_qualification':'Environment, verification, canonical report and evaluation isolation passed. A post-acceptance planning rewrite was rejected at 2048 tokens and the terminal phase recovered. The new queue receipt is verified separately on that saved real-model request; its complete clean-run wall cost has not been remeasured.',
    'workflow_qualification':'Three of six probes produced matching plans, including one maximum-iteration reply. Batch read hints failed to queue in two contexts. Only the empty-context pair completed a matched-plan comparison; batched planning was slower. No robot or learned routing ran in this diagnostic.',
    'terminal_report_qualification':'Original real-model report read correct terminal files but invented a verification artifact path. Preserved as failed reference-quality evidence. New actual-file guard passed CPU regressions only; no real-model report acceptance claimed.',
    'historical_run_status_unchanged':'FAILED; supervisor interrupted to prevent accepted-snapshot mutation',
    'model_config_sha256':hashlib.sha256(source.encode()).hexdigest(),
    'limits':['Three saved request families only; first-use and all paid costs retained.',
        'One development robot episode, no population or efficacy comparison.',
        'Terminal report isolation covered by real-Agent CPU regression, no new robot evaluation.',
        'No automatic token-threshold rollover established by this short conversation.',
        'Native robot lifecycle used reference decoder; optimized real-model evidence is replay-only.']}
out=root/args.out;out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:json.dump(value,f,indent=2)
print(json.dumps({'status':value['status'],'pairs':[{k:x[k] for k in ['label','relative_reduction','same_tokens_all_four']} for x in pairs],
    'memory':memory['status'],'cost':{k:value['cost_bill'][k] for k in ['full_wall_seconds','model_requests','model_rejected','generation_seconds']}}))
