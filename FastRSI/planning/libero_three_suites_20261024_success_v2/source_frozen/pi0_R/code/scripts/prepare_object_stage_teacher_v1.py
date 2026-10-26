"""Freeze four saved developmental observations before sparse teacher diagnosis."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
NAME = 'object_stage_teacher_v1_20261006'
IDS = ('i26-s120-white_mug', 'i31-s120-white_mug',
       'i31-s240-yellow_white_mug', 'i32-s215-yellow_white_mug')


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def save(p, x):
    with p.open('x') as f: json.dump(x, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', choices=(NAME, 'object_stage_teacher_v2_20261006', 'object_stage_teacher_v3_20261006'), default=NAME)
    args = parser.parse_args(); name = args.name
    prior = ROOT / 'runs/object_stage_memory_v2_20261006'
    source = prior / 'inputs/INPUTS.json'
    if sha(source) != read(prior / 'model_job/learning/RESULT.json')['input_artifact_sha256']['INPUTS.json']:
        raise ValueError('Saved complete model input changed')
    run = ROOT / 'runs' / name
    run.mkdir(exist_ok=False)
    inputs = run / 'inputs'; inputs.mkdir()
    by_id = {x['id']: x for x in read(source)}
    bindings, private, rows = {}, [], []
    for identity in IDS:
        row = by_id[identity]
        # IDs, roles and outcome supervision are private audit metadata only.
        for image in row['images']:
            if sha(ROOT / image['path']) != image['sha256']:
                raise ValueError('Saved legal image changed')
            bindings[image['path']] = image['sha256']
        rows.append(row)
        init = int(identity.split('-')[0][1:]); step = int(identity.split('-')[1][1:])
        replay_path = prior / 'replays' / ('init' + str(init)) / 'REPLAY.json'
        replay = read(replay_path)
        if replay['status'] != 'EXACT_IMAGE_REPLAY': raise ValueError('Unverified audit facts')
        frame = next(f for f in replay['rows'] if f['step'] == step)
        fact = frame['offline_supervision'][row['subject']]
        private.append(dict(id=identity, source_replay=str(replay_path.relative_to(ROOT)),
            source_replay_sha256=sha(replay_path), native_subgoal=fact['native_subgoal'],
            native_grasp=fact['native_grasp'], native_gripper_contact=fact['native_gripper_contact'],
            current_complete_proxy=fact['released_support_proxy']))
    save(inputs / 'INPUTS.json', rows)
    save(inputs / 'OFFLINE_AUDIT_ONLY.json', private)
    save(inputs / 'SOURCE_MANIFEST.json', bindings)
    cfg_source = ROOT / 'runs/task4_informative_pair_dev_187_v8_20261006/MODEL_CONFIG.json'
    shutil.copyfile(cfg_source, run / 'MODEL_CONFIG.json')
    rels = ['scripts/prepare_object_stage_teacher_v1.py', 'scripts/run_object_stage_teacher_v1.py',
        'scripts/orchestrate_object_stage_teacher_v1.py', 'scripts/orchestrate_full_os.py',
        'scripts/native_service_config.py', 'scripts/qwen_agent_server.py',
        'src/fastrsi/__init__.py', 'src/fastrsi/tool_protocol.py',
        'src/fastrsi/constrained_tools.py', 'src/fastrsi/decoding_policy.py',
        'src/fastrsi/object_stage_memory_v1.py', 'src/fastrsi/decision_corpus.py', 'scripts/stage_trace.py']
    for rel in rels:
        dst = run / 'code' / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dst)
    save(run / 'CODE_MANIFEST.json', {rel: sha(run / 'code' / rel) for rel in rels})
    save(run / 'REGISTRATION.json', dict(name=name, registered_at=time.time(), identities=list(IDS),
        input_sha256=sha(inputs / 'INPUTS.json'), model_config_sha256=sha(run / 'MODEL_CONFIG.json'),
        source_inputs_sha256=sha(source), offline_audit_sha256=sha(inputs / 'OFFLINE_AUDIT_ONLY.json'),
        max_model_calls=4, max_tokens=384, resources='one Qwen service on HCU 1; no robot',
        authority='UNVERIFIED_TEACHER_DIAGNOSIS_ONLY', persistent_updates=False,
        view_mode=('human_upright_current' if name.endswith('v3_20261006') else
            'current_only' if name != NAME else 'previous_and_current'),
        image_transform='horizontal flip of existing policy RGB only' if name.endswith('v3_20261006') else 'identity',
        comparison='Same four saved boundaries. v2 removes history; v3 differs from v2 only by the native human-video horizontal transform. No new sample or robot experiment.',
        default_changed=False, scope='Seen developmental observations; no generalization or speedup claim.'))
    print(run)


if __name__ == '__main__': main()
