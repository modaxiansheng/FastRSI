"""Append two audited default-continuation pairs without modifying older evidence."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np

def main(project,output):
    base=project/'planning/native_quality_cost_pairs_v1_20261010'
    pairs=json.loads((base/'PAIRS.json').read_text());arrays=dict(np.load(base/'INPUTS.npz',allow_pickle=False))
    audit_path=project/'planning/native_last_choice_audit_v1_20261010/RESULT.json'
    audit=json.loads(audit_path.read_text());assert audit['status']=='AUDITED_REAL_LAST_CHOICE_PAIR_SUPERVISION'
    codec_path=project/'third_party/PhyAgentOS-core/PhyAgentOS/runtime/policy/msgpack_numpy.py'
    spec=importlib.util.spec_from_file_location('merge_codec',codec_path);codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    for slot,row in enumerate(audit['cases']):
        result_path=project/f'planning/native_last_choice_gpu{slot}_v2_20261010/RESULT.json'
        assert sha(result_path)==row['result_sha256']
        result=json.loads(result_path.read_text());plan=result['plan'];index=len(pairs)
        req_path=project/plan['source_request'];resp_path=project/plan['source_response']
        assert sha(req_path)==plan['source_request_sha256'] and sha(resp_path)==plan['source_response_sha256']
        req=codec.unpackb(req_path.read_bytes());resp=codec.unpackb(resp_path.read_bytes())
        arrays[f'feature_{index}']=np.asarray(resp['prefix_feature'],np.float32)
        arrays[f'robot_{index}']=np.asarray(req['observation/state'],np.float32)
        arrays[f'candidates_{index}']=np.asarray(resp['candidates'],np.float32)
        arrays[f'budget_{index}']=np.array([400-plan['fork_step']],np.float32)
        successes=[int(v) for v in row['native_success']];remaining=row['remaining_actions']
        pairs.append(dict(pair=index,development_group=row['initial'],step=row['fork_step'],
            continuation='unchanged default pi, explicitly matched future noise',
            request=plan['source_request'],request_sha256=plan['source_request_sha256'],
            response=plan['source_response'],response_sha256=plan['source_response_sha256'],
            observed_native_success=successes,observed_remaining_actions=remaining,
            quality_preference=None if successes[0]==successes[1] else int(successes[1]>successes[0]),
            cost_preference_given_both_success=(int(remaining[1]<remaining[0]) if all(successes) and remaining[0]!=remaining[1] else None),
            both_failed_does_not_create_preferred_action=not any(successes),
            action_count_is_cost_proxy_not_wall_time=True,
            success_labels_not_calibrated_probabilities=True,not_labels_for_arbitrary_router_continuation=True,
            evidence_bindings=dict(last_choice_audit_sha256=sha(audit_path),actual_candidate0_result_sha256=sha(result_path),
                actual_candidate1_reference_sha256=plan['reference_outcome_sha256'])))
    assert len(pairs)==10
    output.mkdir(exist_ok=False);np.savez_compressed(output/'INPUTS.npz',**arrays)
    (output/'PAIRS.json').write_text(json.dumps(pairs,indent=2)+'\n')
    result=dict(status='MERGED_TEN_FACTUAL_QUALITY_COST_PAIRS',pairs=len(pairs),
        context_groups=sorted({r['development_group'] for r in pairs}),
        informative_preference_groups=sorted({r['development_group'] for r in pairs
            if r['quality_preference'] is not None or r['cost_preference_given_both_success'] is not None}),
        quality_preference_pairs=sum(r['quality_preference'] is not None for r in pairs),
        conditional_action_cost_preference_pairs=sum(r['cost_preference_given_both_success'] is not None for r in pairs),
        unresolved_or_tied_pairs=sum(r['quality_preference'] is None and r['cost_preference_given_both_success'] is None for r in pairs),
        online_input_keys=['existing_policy_prefix_feature','robot8','candidate_action_blocks','remaining_budget'],
        group_ids_are_metadata_only=True,robot_executions=0,model_updates=0,goal_complete=False,
        limitation='Small development pair set; no new model or independent gain. Complement factual default-pi return data, do not infer arbitrary router futures.')
    (output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.project,args.output)
