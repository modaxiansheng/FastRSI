"""Audit executed first-choice diagnostics; retain factual continuation labels only."""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from phase_iql_value_v1 import Predictor


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(project, output):
    started = time.perf_counter()
    # Loading the codec directly avoids importing the whole Agent package for an audit.
    codec_path = project / 'third_party/PhyAgentOS-core/PhyAgentOS/runtime/policy/msgpack_numpy.py'
    if not codec_path.exists():
        codec_path = Path('/mnt/data/task/yuchen/FastPhyagentos/frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1/dependencies/PhyAgentOS-core/PhyAgentOS/runtime/policy/msgpack_numpy.py')
    spec = importlib.util.spec_from_file_location('audit_codec', codec_path)
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    output.mkdir(exist_ok=False)
    cases = []
    for slot, initial in enumerate((33, 34)):
        root = project / f'planning/native_first_choice_gpu{slot}_v1_20261010'
        result = json.loads((root / 'RESULT.json').read_text())
        assert result['status'] == 'COMPLETED'
        plan = result['plan']
        fork = plan['fork_step']
        assert initial == plan['initial'] == result['initial']
        for key in ('source_request', 'source_response', 'source_default_response',
                    'default_outcome', 'full_learned_outcome', 'model'):
            assert sha(project / plan[key]) == plan[key + '_sha256']
        predictor = Predictor(project / plan['model'])
        expected = codec.unpackb((project / plan['source_request']).read_bytes())
        original = codec.unpackb((project / plan['source_response']).read_bytes())
        default_response = codec.unpackb((project / plan['source_default_response']).read_bytes())
        assert np.array_equal(original['candidates'][0], default_response['actions'])
        changes = []
        total_candidates = 0
        records = []
        for row_index, row in enumerate(result['rows']):
            before = fork + 10 * row_index
            assert row['before'] == before and row['after'] == before + 10
            index = before // 10
            request_path = root / f'{index:03d}_request.msgpack'
            response_path = root / f'{index:03d}_response.msgpack'
            request = codec.unpackb(request_path.read_bytes())
            response = codec.unpackb(response_path.read_bytes())
            tag = request['_branch_baseline_v2']
            assert tag['step'] == before and tag['fork_step'] == fork
            assert tag['noise_base'] == plan['noise_base']
            selected = int(before == fork)
            count = 2 if selected else 1
            candidates = np.asarray(response['candidates'])
            assert candidates.shape == (count, 10, 7)
            assert row['selected_index'] == selected and row['candidates'] == count
            commands = np.load(root / f'{index:03d}_command.npz', allow_pickle=False)['actions']
            assert np.array_equal(commands, candidates[selected])
            for j in range(count):
                noise = np.random.default_rng(np.random.SeedSequence(
                    [plan['noise_base'], index, j])).standard_normal((10, 32)).astype(np.float32)
                assert hashlib.sha256(noise.tobytes()).hexdigest() == response['branch_baseline']['noise_sha256'][j]
            if selected:
                for key in ('observation/image', 'observation/wrist_image', 'observation/state'):
                    assert np.array_equal(request[key], expected[key])
                assert request['prompt'] == expected['prompt']
                assert np.array_equal(candidates, original['candidates'])
                choice, score = predictor.score(response['prefix_feature'],
                    request['observation/state'], candidates, 400 - before)
                assert choice == 1
                assert np.allclose(score['q_scores'], result['first_choice_model_scores']['q_scores'], atol=1e-6)
                changes.append(before)
            total_candidates += count
            records.append(dict(before=before, request_sha256=sha(request_path),
                response_sha256=sha(response_path), command_sha256=sha(root / f'{index:03d}_command.npz')))
        assert changes == [fork]
        assert total_candidates == result['actor_inferences']
        assert len(result['rows']) == result['remaining_control_requests']
        assert result['physical_steps'] == fork + 10 * len(result['rows']) == 400
        assert result['parameter_updates'] == result['persistent_updates'] == 0
        assert not result['full_agent_chain']
        for stop in json.loads((root / 'STOP_RECEIPT.json').read_text()):
            assert stop['state']['Running'] is False and stop['state']['Pid'] == 0
        def native(path):
            value = json.loads((project / path).read_text())
            sessions = value['cases'][0]['sessions']
            assert len(value['cases']) == len(sessions) == 1
            return sessions[0]['result']
        default = native(plan['default_outcome'])
        full = native(plan['full_learned_outcome'])
        cases.append(dict(initial=initial, fork_step=fork,
            default_success=default['success'], default_actions=default['num_steps'],
            first_change_then_default_success=result['native_success'],
            first_change_then_default_actions=result['physical_steps'],
            full_learned_success=full['success'], full_learned_actions=full['num_steps'],
            checked_requests=len(records), checked_candidate_inferences=total_candidates,
            diagnostic_case_seconds=result['seconds_including_close'],
            full_host_seconds=json.loads((root / 'HOST.json').read_text())['seconds'],
            result_sha256=sha(root / 'RESULT.json'), records=records,
            paired_label_continuation='candidate at fork then unchanged default pi',
            local_preference=0 if default['success'] and not result['native_success'] else None,
            interpretation=('Observed one-change intervention loses native success relative to default'
                if initial == 33 else 'First change alone fails; later learned choices are necessary for the observed full-run success')))
    report = dict(status='AUDITED_FIRST_LEARNED_CHANGE_DIAGNOSTICS', cases=cases,
        robot_executions=2, model_updates=0, full_agent_comparison=False,
        same_fork_inputs_and_cross_host_candidates_exact=True,
        commands_and_future_noise_checked=True,
        no_general_limit_one_intervention_supported=True,
        now_development_contexts=[33, 34], independent_new_performance_claim=False,
        goal_complete=False, seconds=time.perf_counter()-started)
    (output / 'RESULT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({**report, 'cases': [{k:v for k,v in c.items() if k != 'records'} for c in cases]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.project, args.output)
