"""Repair dispatch-budget compatibility and prepare balanced whole-device lanes."""
import ast
import hashlib
import importlib.util
import inspect
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, v):
    with p.open('x') as f:
        json.dump(v, f, indent=2)


def main():
    assert (ROOT/'control03/OWNER_RESULT.json').exists(), 'Drain all paid cases first'
    (ROOT/'control05').mkdir(exist_ok=False)
    overlays = {}
    for model in ['pi05','pi0','xvla','openvla']:
        source = ROOT/'source_frozen'/(model+'_R')/'code/scripts/run_native_baseline.py'
        s = source.read_text()
        for a,b in [('for attempt in range(3):','for attempt in range(6):'),
                    ('max_replans=2,','max_replans=5,'),
                    ("'max_replans_per_episode':2", "'max_replans_per_episode':5")]:
            assert s.count(a)==1,(model,a,s.count(a))
            s=s.replace(a,b)
        ast.parse(s)
        out=ROOT/'resource05_overlay'/model/'run_native_baseline.py'
        out.parent.mkdir(parents=True)
        out.write_text(s)
        overlays[model]=dict(original=str(source),original_sha256=sha(source),overlay=str(out),overlay_sha256=sha(out))
    # Preserve a closed failed run conservatively; do not rerun its root for quality.
    case=ROOT/'campaigns/libero_original_best_20261024_xvla_R_libero_spatial_original/node107/lane1_full_attempt002/case005_t4_i2'
    raw=case/'stream/system_t4_i2/RESULT.json'
    result=json.loads(raw.read_text())
    assert result['chain_status']=='FAILED' and result['environment_success'] is False
    assert result['error'].rstrip().endswith("ValueError: Current terminal records are incomplete")
    sessions=json.loads((case/'stream/system_t4_i2/CANONICAL_TERMINAL_SNAPSHOT.json').read_text())['sessions']
    executed=[s for s in sessions if s['result'].get('artifact_dir')]
    assert len(executed)==3 and len(sessions)==4 and sessions[-1]['status']=='pending'
    assert all(s['result']['success'] is False for s in executed)
    assert all(o['code']=='verified' for o in result['outcomes'])
    spec=importlib.util.spec_from_file_location('retained',ROOT/'run_lane.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    code=inspect.getsource(m.metrics_for)
    code=code.replace("after_execution_failure=agent_failure and failure_kind in (", "after_execution_failure=agent_failure and failure_kind in ('RUNTIME_RECOVERY_DISPATCH_LIMIT',")
    needle="  p=case_dir/'stream'/row['case']['id']/'workspace'/s['result']['artifact_dir']/'episode.json'"
    assert code.count(needle)==1
    code=code.replace(needle,"  if not s['result'].get('artifact_dir'):continue\n"+needle)
    needle=" write(case_dir/'METRICS.json',value);"
    assert code.count(needle)==1
    code=code.replace(needle," value.update(agent_failure=False,system_runtime_failure=True,recovery_budget_completed=False,registered_attempt_ceiling=6,pending_unexecuted_children=1,sealed_failure_no_quality_rerun=True,status_schema_note='AGENT_FAILED is the retained closed-failure status key; this specific cause is runtime dispatch-budget compatibility, not Agent reasoning.')\n"+needle)
    exec(compile(code,'resource05_failed_case_sealer','exec'),m.__dict__)
    status=json.loads((case.parent/'STATUS.json').read_text())
    row=next(x for x in status['cases'] if x['task_id']==4 and x['init_state_id']==2)
    value=m.metrics_for(case,'xvla','R','libero_spatial','original',dict(task_id=4,init_state_id=2),row['host_case_seconds'],True,'RUNTIME_RECOVERY_DISPATCH_LIMIT')
    save(ROOT/'control05/CLOSED_RUNTIME_FAILURE_RECEIPT.json',dict(time=time.time(),raw_result=str(raw),raw_sha256=sha(raw),physical_success=False,end_to_end_success=False,executed_attempts=3,action_steps=value['action_steps'],qwen_tokens=value['qwen_total_tokens'],next_child_pending_and_unexecuted=True,no_quality_rerun=True,scope='One runtime failure retained in complete500 denominator; planned recovery did not fully execute. No success imputed. Raw artifacts unmodified.'))
    lane=(ROOT/'run_lane.py').read_text()
    lane=lane.replace('base=32100+100*','base=21100+100*')
    lane=lane.replace("assert args.gpu_base in (range(8) if args.arm=='VLA'", "assert args.gpu_base in (range(8) if args.resource_profile=='whole_gpu_eight_lanes_resource05' or args.arm=='VLA'")
    needle="  if role=='runner':\n"
    assert lane.count(needle)==1
    mapping="{0:2,1:3,2:2,3:3,4:6,5:7,6:6,7:7}"
    lane=lane.replace(needle,"  if role=='qwen':\n   assigned="+mapping+"[slot_index]\n   argv=[('HIP_VISIBLE_DEVICES='+str(assigned)) if v.startswith('HIP_VISIBLE_DEVICES=') else ('CUDA_VISIBLE_DEVICES='+str(assigned)) if v.startswith('CUDA_VISIBLE_DEVICES=') else v for v in argv]\n"+needle+"   argv=[('FASTRSI_RESIDENT_MAX_INFERENCES=2000' if v.startswith('FASTRSI_RESIDENT_MAX_INFERENCES=') else v) for v in argv]\n   pos=argv.index(module.IMAGE)\n   argv[pos:pos]=['-v',str(ROOT/'resource05_overlay'/model/'run_native_baseline.py')+':'+cbase+'/scripts/run_native_baseline.py:ro']\n")
    lane=lane.replace("sha(ROOT/'run_lane.py')", "sha(ROOT/'run_lane_resource05.py')")
    ast.parse(lane)
    (ROOT/'run_lane_resource05.py').write_text(lane)
    bindings=json.loads((ROOT/'SOURCE_BINDINGS.json').read_text())
    bindings.update(lane_sha256=sha(ROOT/'run_lane_resource05.py'),resource05_overlays=overlays)
    save(ROOT/'RESOURCE05_BINDINGS.json',bindings)
    save(ROOT/'control05/PREPARATION_RECEIPT.json',dict(time=time.time(),overlays=overlays,retained_failed_case=str(case),change='Dispatcher six total attempts and supervisor five child replans now match already-authorized verifier budget; inference safety ceiling2000 covers6x300 steps. GPU roles share one physical GPU per lane. No model generation or success scoring change.',eight_lane_layout={s:gpu for s,gpu in enumerate([2,3,2,3,6,7,6,7])}))
    print('RESOURCE05_PREPARED_COMPATIBILITY_REPAIRED_AND_ONE_FAILURE_RETAINED')


if __name__=='__main__':
    main()
