"""Join paid observations to offline labels only after model predictions exist."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.drawer_facts import proposed_operation,DESCRIPTIONS


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    args=p.parse_args();run=args.run
    native=json.loads((run/'NATIVE_LABELS.json').read_text());assert native['status']=='EXACT_IMAGE_REPLAY'
    labels={r['step']:r for r in native['rows']}
    manifest=json.loads((args.inputs/'MANIFEST.json').read_text());records={}
    for entry in manifest['records']:
        raw=(args.inputs/entry['file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==entry['sha256']
        records[entry['file']]=json.loads(raw)
    qwen=json.loads((run/'qwen/RESULT.json').read_text());clef=json.loads((run/'replay/RESULT.json').read_text())
    assert qwen['status']==clef['status']=='COMPLETED'
    models={
        'qwen':{r['file']:dict(facts=r['facts'],seconds=r['seconds']) for r in qwen['rows']},
        'clef':{r['file']:dict(facts={k:v['choice'] for k,v in r['response']['answers'].items()},seconds=r['total_seconds']) for r in clef['calls']}}
    summaries={};details=[]
    coverage={k:{'positive':0,'negative':0} for k in DESCRIPTIONS}
    for record in records.values():
        step=int(record['request']['state']['observation_id'].split('_')[-1])
        for k,v in labels[step]['facts'].items():coverage[k]['positive' if v else 'negative']+=1
    for model,predictions in models.items():
        assert set(predictions)==set(records)
        stats=dict(predictions=0,correct=0,false_positive=0,false_negative=0,unknown=0,contradictions=0,
                   close_candidates=0,unsupported_close_candidates=0,seconds=0)
        for name,record in records.items():
            step=int(record['request']['state']['observation_id'].split('_')[-1]);truth=labels[step]['facts'];pred=predictions[name]
            for image in record['images']:
                filename=Path(image['path']).name
                assert labels[step]['images'][filename]==image['sha256'],'Label does not bind to model image'
            mistakes=[]
            for key in DESCRIPTIONS:
                value=pred['facts'][key];stats['predictions']+=1
                if value=='unknown':stats['unknown']+=1
                elif (value=='true')==truth[key]:stats['correct']+=1
                else:
                    kind='false_positive' if value=='true' else 'false_negative';stats[kind]+=1;mistakes.append(key+':'+kind)
            operation=proposed_operation(pred['facts'])
            stats['contradictions']+=int(operation=='native_unresolved')
            candidate=operation=='candidate_close_drawer';supported=truth['bowl_inside'] and not truth['drawer_closed']
            stats['close_candidates']+=int(candidate);stats['unsupported_close_candidates']+=int(candidate and not supported)
            stats['seconds']+=pred['seconds']
            details.append(dict(model=model,file=name,step=step,truth=truth,predicted=pred['facts'],mistakes=mistakes,
                                diagnostic_operation=operation,necessary_close_conditions_met=supported))
        summaries[model]=stats
    report=dict(status='ANALYZED',models=summaries,rows=details,observations=len(records),independent_episodes=1,
        scope='Seen development trajectory; native-predicate agreement and abstention, not calibrated safety or unseen generalization.',
        class_coverage=coverage,
        label_limit='Native in is geometric containment; it does not independently establish release, stability or clearance. Question and native-predicate agreement is diagnostic, not a ready-made training target.',
        skill_admission='NOT_ACCEPTED',
        reason='Necessary visual-condition diagnostics only. No real skill intervention; release/clearance conditions also required before closing.',
        new_robot_rollouts=0,trained_models=0,learning_gain_proven=False)
    (run/'ANALYSIS.json').write_text(json.dumps(report,indent=2));print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
