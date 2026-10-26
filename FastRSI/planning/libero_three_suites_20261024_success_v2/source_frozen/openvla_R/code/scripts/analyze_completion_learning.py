"""Summarize saved model/decision results without selecting new thresholds."""
import argparse,hashlib,json,statistics
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    result=json.loads((a.run/'experiment/RESULT.json').read_text())
    status=json.loads((a.run/'STATUS.json').read_text())
    assert result['status']==status['status']=='COMPLETED'
    labels=json.loads((a.run/'inputs/SUPERVISION.json').read_text());truth={r['id']:r for r in labels}
    predictions=json.loads((a.run/'experiment/PREDICTIONS.json').read_text())
    bindings=json.loads((a.run/'experiment/TRAINING_BINDINGS.json').read_text())
    test_groups={truth[r['id']]['group'] for r in predictions}
    for name,b in bindings.items():
        assert not (set(b['groups']) & test_groups)
        assert all(truth[i]['role']!='test' for i in b['ids'])
        assert hashlib.sha256((a.run/'experiment'/f'{name}_head.pt').read_bytes()).hexdigest()==b['checkpoint_sha256']
    changes=[]
    for p in predictions:
        if p['decisions']['frozen']!=p['decisions']['updated']:
            y=truth[p['id']]
            changes.append(dict(id=p['id'],goal=y['goal'],source=y['source_trace'],step=y['source_step'],
                terminal=y['is_terminal'],native_complete=y['native_complete'],
                before=p['decisions']['frozen'],after=p['decisions']['updated'],
                p_before=p['estimates']['frozen'],p_after=p['estimates']['updated']))
    calls=result['calls'];timings=[r['seconds'] for r in calls]
    summary=dict(status='ANALYZED',scope='Retrospective grouped development; no OS verifier replacement or realized runtime saving',
        samples=len(calls),test_samples=len(predictions),test_initial_groups=len(test_groups),
        metrics={k:{n:v for n,v in row.items() if n!='rows'} for k,row in result['metrics'].items()},
        actual_changed_shadow_decisions=changes,
        cost=dict(campaign_seconds=status['seconds'],model_load_seconds=result['load_seconds'],
            image_inference_seconds=sum(timings),first_request_seconds=timings[0],
            subsequent_request_median_seconds=statistics.median(timings[1:]),
            fit_seconds=result['fit_seconds'],cost_note='Nested durations; do not add to campaign wall. Paid historical robot acquisition is separate.'),
        terminal_failure_holdout_present=any(r['role']=='test' and r['is_terminal'] and not r['native_complete'] for r in labels),
        learned_parameters_each=bindings['updated']['parameters'],new_robot_episodes=0,
        deployment_approved=False,learning_net_savings_established=False,
        source_sha256={str(p.relative_to(a.run)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (a.run/'STATUS.json',a.run/'experiment/RESULT.json',a.run/'experiment/PREDICTIONS.json',a.run/'inputs/SUPERVISION.json')})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
