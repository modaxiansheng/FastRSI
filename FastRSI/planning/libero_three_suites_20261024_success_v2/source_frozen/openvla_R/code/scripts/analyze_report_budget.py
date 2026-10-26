"""Keep candidate execution effects, policy allocations and research cost separate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from analyze_representative_batch import analyze
from fastrsi.validation_policy import information_value

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run_id');ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args();run=ROOT/'runs'/a.run_id;status=read(run/'STATUS.json')
    assert status['status'] in ('COMPLETED','FAILED')
    pairs=[];bills=[];attempts=[]
    for folder in sorted(run.glob('pair*')):
        attempt={'pair':folder.name,'arms':{}}
        for arm in ('N','F'):
            path=folder/arm/'STATUS.json'
            if path.exists():attempt['arms'][arm]=read(path)
        attempts.append(attempt)
        if not (folder/'FEEDBACK.json').exists():continue
        feedback=read(folder/'FEEDBACK.json');arms={}
        for arm in ('N','F'):
            report=analyze(folder/arm);bills.append(report['bill'])
            calls=report['purpose_summary'];cases=report['cases']
            saved=read(folder/arm/'stream/RESULT.json')['cases'][0]
            workspace=folder/arm/'stream'/saved['case']['id']/'workspace'
            report_path=workspace/saved['terminal_report']['report']
            arms[arm]={'wall':report['bill']['full_wall_seconds'],
                'controller':read(folder/arm/'stream/RESULT.json')['seconds'],
                'stages':report['stage_seconds'],'model_purposes':calls,
                'cases':[{k:c.get(k) for k in ('id','task','init','environment_success','verification','chain_status','actual_policy_prompts')} for c in cases],
                'report_path':str(report_path.relative_to(ROOT)),'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
                'report_text':report_path.read_text(),
                'prefetch':saved['terminal_report']['prefetch_enabled'],
                'fixed_experience_unchanged':saved['experience_before']==saved['experience_after'],
                'evaluation_closed':(workspace/'.FASTRSI_EVALUATION_CLOSED.json').exists(),
                'bill':report['bill']}
        pairs.append({'pair':folder.name,'feedback':feedback,'arms':arms,
                      'wall_saving':arms['N']['wall']-arms['F']['wall'],
                      'controller_saving':arms['N']['controller']-arms['F']['controller']})
    states=read(run/'FROZEN_POLICY_STATES.json') if (run/'FROZEN_POLICY_STATES.json').exists() else {}
    policies={};independent=sum(p['feedback']['paid_cost'] for p in pairs if p['feedback']['mode']=='evaluation')
    for mode,state in states.items():
        rows=state['records'];decisions=[]
        for f in sorted(run.glob(mode+'_STEP_*.json')):
            d=read(f)['receipt']['decision']
            decisions.append({k:d[k] for k in ('action','reason','quality_pairs','mean_saving','modeled_next_information_value','modeled_next_net_value','selection_record_ids')})
        policies[mode]={'paid_development_pairs':[r['pair_id'] for r in rows],
                        'attributed_development_capacity_seconds':sum(r['paid_cost'] for r in rows),
                        'common_independent_capacity_seconds':independent,
                        'decisions':decisions,'independent_cold_start_net_benefit':None}
    fixed_two={'scope':'Additional simple reference from the identical paid bootstrap; not a separately run method',
               'attributed_development_capacity_seconds':sum(p['feedback']['paid_cost'] for p in pairs if p['pair'] in ('pair0','pair1')),
               'common_independent_capacity_seconds':independent,
               'interpretation':'If feedback also stops at two, it has not beaten this fixed-two rule. Do not count fixed-four savings as evidence against every reasonable fixed allocation.'}
    sensitivity=[]
    for f in sorted(run.glob('DECISIONS_*.json')):
        for mode,d in read(f).items():
            for uses in (2,20,100,200,500):
                cfg=read(run/'PROTOCOL.json')['config']
                info=information_value(d['mean_saving'],d['posterior_variance'],cfg['observation_sd']**2,uses)
                sensitivity.append({'decision_file':f.name,'mode':mode,'assumed_uses':uses,
                                    'cost_information_value':info,'net_value':info-cfg['next_pair_estimate']})
    output={'status':status,'scope':'Prospective common-prefix mechanism diagnostic, not independent full RSI cold-start comparison',
            'pairs':pairs,'attempts_including_failures':attempts,'policies':policies,'fixed_two_reference':fixed_two,'sensitivity':sensitivity,
            'project_bill':{'campaign_wall_seconds':status['full_wall_seconds'],
                'actual_pair_capacity_seconds':sum(s['full_wall_seconds'] for p in attempts for s in p['arms'].values() if 'full_wall_seconds' in s),
                'unclosed_arms':[p['pair']+'/'+arm for p in attempts for arm,s in p['arms'].items() if 'full_wall_seconds' not in s],
                'prior_failed_cpu_attempt_seconds':read(run/'PRIOR_ATTEMPT.json')['full_wall_seconds'] if (run/'PRIOR_ATTEMPT.json').exists() else 0,
                'model_calls':sum(b.get('model_calls',0) for b in bills),
                'hcu_allocation_seconds':sum(b.get('hcu_allocation_seconds',0) for b in bills),
                'cpu_gate_and_failures':'Current CPU gate included in campaign wall; prior attempt separate. Pair costs include failed arms. Model/device details below cover analyzed completed feedback pairs only if this campaign failed.',
                'manual_development_and_transfer_seconds':None},
            'accounting':'Do not add overlapping stage durations or sum per-policy attributed costs as actual project cost. Shared measured branch outcomes are common random numbers; not three separately executed method cold starts.',
            'claims_excluded':['2pp noninferiority','30% lifetime net return','cross-candidate policy learning']}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(output,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':status['status'],'pairs':len(pairs),'policies':policies,'bill':output['project_bill']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
