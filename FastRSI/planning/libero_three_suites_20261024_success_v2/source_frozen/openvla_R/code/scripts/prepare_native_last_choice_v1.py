"""Bind the last real learned change to its actual default-pi suffix.

The recorded successful learned run supplies candidate1 + default-pi outcome.
Only candidate0 + default-pi still requires a new execution. No unexecuted
candidate is labeled, and this is development data rather than a new test score.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np


def main(project, output):
    codec_path = project/'third_party/PhyAgentOS-core/PhyAgentOS/runtime/policy/msgpack_numpy.py'
    spec = importlib.util.spec_from_file_location('last_choice_codec', codec_path)
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    output.mkdir(exist_ok=False)
    cases = []
    for version, initial in ((14,34),(15,35)):
        root = project/f'planning/native_value_phase_v{version}_20261010'
        service = root/'R/stream/value_service'
        calls = [json.loads(x) for x in (service/'VALUE_POLICY_CALLS.jsonl').read_text().splitlines()]
        changes = [i for i,r in enumerate(calls) if r['selected_index'] == 1]
        assert changes
        fork_index = changes[-1]
        fork = calls[fork_index]['step']
        assert fork == fork_index*10
        prefix = []
        for i,r in enumerate(calls):
            assert r['index'] == i and r['step'] == 10*i
            reply = codec.unpackb((service/(r['evidence_name']+'_response.msgpack')).read_bytes())
            assert np.array_equal(reply['actions'], reply['candidates'][r['selected_index']])
            if i < fork_index:
                prefix.append(reply['actions'].tolist())
            elif i > fork_index:
                assert r['selected_index'] == 0
        outcome_path = root/'R/stream/RESULT.json'
        outcome = json.loads(outcome_path.read_text())['cases'][0]['sessions'][0]['result']
        assert outcome['success'] is True
        row = calls[fork_index]
        request = service/(row['evidence_name']+'_request.msgpack')
        response = service/(row['evidence_name']+'_response.msgpack')
        model = project/'planning/native_paired_q_update_v1_20261010/training/paired_router.npz'
        assert sha(model) == row['model_sha256']
        d = output/f'init{initial}'
        d.mkdir()
        candidates = codec.unpackb(response.read_bytes())['candidates']
        assert candidates.shape == (2,10,7)
        np.savez_compressed(d/'CANDIDATES.npz',candidates=candidates)
        paths = dict(source_request=request, source_response=response,
                     reference_outcome=outcome_path, model=model)
        plan = dict(status='REGISTERED_LAST_LEARNED_CHANGE_BRANCH', initial=initial,
            fork_step=fork, noise_base=row['noise_base'], prefix=prefix,
            **{k:str(v.relative_to(project)) for k,v in paths.items()},
            **{k+'_sha256':sha(v) for k,v in paths.items()},
            reference_selected_index=1, intervention_selected_index=0,
            reference_native_success=outcome['success'],
            reference_remaining_actions=outcome['num_steps']-fork,
            source_final_control_step=calls[-1]['step'],
            reference_suffix_default_pi_verified=True,
            max_steps=400, post_choice_policy='unchanged pi, candidate0, same explicit future noise',
            reference_inference_and_prefix_cost_not_free=True,
            held_out_result=False, operator_learning_or_speedup_claimed=False)
        (d/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
        cases.append(dict(initial=initial,fork_step=fork,
            reference_remaining_actions=plan['reference_remaining_actions']))
    result = dict(status='PREPARED_REAL_LAST_CHOICE_BRANCHES',cases=cases,
        robot_executions=0,model_updates=0,reference_reuse_requires_new_fork_equality=True,
        purpose='Obtain same-state candidate0 versus factual candidate1 default-pi continuation labels',
        learned_gain_claimed=False,goal_complete=False)
    (output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    main(args.project,args.output)
