"""Paired public engineering pilot; separate concurrent elapsed from paid work."""
import argparse,hashlib,json
from pathlib import Path
from analyze_representative_batch import analyze,lines,read,ROOT
from analyze_session_flow import normalized_request,normalized_launch
from summarize_native_binding import summarize

def operational(value):
    text=json.dumps(value,sort_keys=True)
    for n in [49500,49502,49510,49520]:text=text.replace(str(n+100),str(n))
    return json.loads(text.replace('/N/','/ARM/').replace('/F/','/ARM/'))

def launch(group,role):
    values=operational(normalized_launch(group,role))
    for i,v in enumerate(values):
        for prefix in ['HIP_VISIBLE_DEVICES=','CUDA_VISIBLE_DEVICES=']:
            if v==prefix+'2':values[i]=prefix+'0'
            elif v==prefix+'3':values[i]=prefix+'1'
    return values

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run_id');ap.add_argument('--out',required=True);a=ap.parse_args()
    run=ROOT/'runs'/a.run_id;status=read(run/'STATUS.json');arms={};pairs=[]
    for arm in ['N','F']:
        group=run/arm
        if not (group/'STATUS.json').exists():continue
        report=analyze(group);report['scope']='Fixed evaluation, public N/F; not learning or development updates'
        for row in report['cases']:
            folder=group/'stream'/row['id'];saved=read(folder/'RESULT.json') if (folder/'RESULT.json').exists() else {}
            events=lines(folder/'AGENT_EVENTS.jsonl');queue=next((e for e in events if e['event']=='queue_accepted'),{})
            row.update(queue_origin=queue.get('origin'),flow_fallbacks=sum(e['event']=='session_flow_handoff' for e in events),
              persistent_updates='Disabled through close/background',
              fixed_and_closed=saved.get('experience_before')==saved.get('experience_after') and (folder/'workspace/.FASTRSI_EVALUATION_CLOSED.json').exists(),
              current_record_read_complete=saved.get('terminal_report',{}).get('read_coverage',{}).get('complete',False),
              planning_calls=sum(c['case']==row['id'] and c['purpose']=='planning' for c in report['model_calls']))
        arms[arm]=report
    if all(x in arms for x in ['N','F']):
        for n,f in zip(arms['N']['cases'],arms['F']['cases']):
            assert n['id']==f['id']
            wn=run/'N/stream'/n['id']/'workspace';wf=run/'F/stream'/f['id']/'workspace'
            pair={'id':n['id'],'same_intent':operational(normalized_request(read(wn/'REQUEST.json')))==operational(normalized_request(read(wf/'REQUEST.json'))),
              'same_initial_images':bool(n['initial_image_hashes']) and n['initial_image_hashes']==f['initial_image_hashes'],
              'same_actual_goal':n['actual_policy_prompts']==f['actual_policy_prompts']==[n['goal']],
              'both_fixed_and_closed':n['fixed_and_closed'] and f['fixed_and_closed'],
              'both_strict_reports':n['current_record_read_complete'] and f['current_record_read_complete'],
              'environment_success':[n['environment_success'],f['environment_success']],
              'semantic_verdicts':[[v['verdict'] for v in row['verification']] for row in [n,f]],
              'planning_calls':[n['planning_calls'],f['planning_calls']],
              'case_seconds':[n['seconds'],f['seconds']]}
            pair['matched_valid']=all(pair[k] for k in ['same_intent','same_initial_images','same_actual_goal','both_fixed_and_closed','both_strict_reports'])
            pairs.append(pair)
    protocol=read(run/'PROTOCOL.json');prior=ROOT/'runs'/Path(protocol['previous_run']).name
    prior_status=read(prior/'STATUS.json');bill=summarize(run)
    receipts=[read(p) for p in run.glob('*/stream/service_calls/*/RECEIPT.json')]
    bill['model_calls']=len(receipts);bill['model_requests']=len(list(run.glob('*/stream/service_calls/*/REQUEST.json')))
    bill['incomplete_model_requests']=bill['model_requests']-len(receipts)
    for key in ['generation_seconds','prompt_tokens','completion_tokens']:bill[key]=sum(r.get(key,0) for r in receipts)
    comparison={}
    if all(x in arms for x in ['N','F']):
        n,f=[arms[x]['bill']['full_wall_seconds'] for x in ['N','F']]
        prelude=min(g['started_at'] for g in status['groups'])-status['started_at']
        postlude=status['finished_at']-max(g['finished_at'] for g in status['groups'])
        pagination=read(ROOT/'runs/session_flow_pagination_cpu_v1_20261002/STATUS.json')['full_wall_seconds']
        branch=read(ROOT/'planning/session_flow_v1/BRANCH_BILL.json')['totals']['full_wall_seconds']
        overhead=prelude+postlude+prior_status['full_wall_seconds']+pagination
        comparison={'launch_configs_equal_after_declared_operational_mapping':{role:launch(run/'N',role)==launch(run/'F',role) for role in ['pi05','qwen','target']},
          'raw_arm_wall_N_minus_F_seconds':n-f,'raw_arm_fraction':(n-f)/n,
          'conservative_candidate_overhead_seconds':overhead,
          'net_N_minus_F_with_transition_pagination_and_current_setup_seconds':n-f-overhead,
          'older_branch_exploration_seconds':branch,
          'net_including_older_branch_exploration_seconds':n-f-overhead-branch,
          'interpretation':'Single concurrent placement; raw difference not stable speedup. Transition and all prior CPU validation charged conservatively to F. Common baseline acquisition/repairs remain separate paid prerequisites.'}
    report={'status':status['status'],'scope':'Shared static engineering diagnostic; no FastRSI learning claim','arms':arms,'pairs':pairs,'comparison':comparison,
      'bill':bill,'interrupted_original_full_wall_seconds':prior_status['full_wall_seconds'],
      'total_active_study_wall_seconds':prior_status['full_wall_seconds']+status['full_wall_seconds'],
      'continuous_study_elapsed_seconds':status['finished_at']-prior_status['started_at'],
      'accounting':'Parallel arm intervals overlap. Do not add them to campaign wall. GPU capacity-time sums across cards; no utilization claim. Local development, transfer and human time unmeasured.',
      'no_claim':['learned selection benefit','30% complete cost savings','2pp noninferiority','generalization','default promotion']}
    out=ROOT/a.out;out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({'status':report['status'],'pairs':pairs,'comparison':comparison,'active_wall':report['total_active_study_wall_seconds']},indent=2))

if __name__=='__main__':main()
