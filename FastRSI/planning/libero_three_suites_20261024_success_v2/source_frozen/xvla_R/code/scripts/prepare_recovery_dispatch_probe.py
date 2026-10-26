"""Bind six existing requests before running the prompt-only diagnostic."""
import ast
import hashlib
import json
from pathlib import Path
from fastrsi.recovery_dispatch_probe import bundle, summarize_verdict

ROOT = Path(__file__).resolve().parents[1]
NAME = 'recovery_dispatch_v1_20261004'


def main():
    out = ROOT / 'planning' / NAME
    out.mkdir(exist_ok=False)
    source = ROOT / 'third_party/PhyAgentOS-core/PhyAgentOS/verification/service.py'
    tree = ast.parse(source.read_text())
    prompts = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
               if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
               and n.targets[0].id in ('SESSION_PROMPT', 'EPISODE_PROMPT')}
    (out / 'PROMPTS.json').write_text(json.dumps(prompts, indent=2))
    prefix = 'runs/structural_window_v1_20261003/host187/'
    sources = [
        ('dual_failure', 'runs/stall_recovery_v1_20261004/N/stream/service_calls/call_cd6ded303d2d47ce9ceea2b0eeb5e80f', False),
        ('moka_failure', prefix+'F/pool/qwen/service_calls/call_54bd632ca92549c7af9afc6202031f44', False),
        ('drawer90_failure', 'runs/prerequisite_native_v2_20261004/N/stream/service_calls/call_19884a2506c54fed8442ee896a328ef2', False),
        ('moka_success', prefix+'N/pool/qwen/service_calls/call_6fcf3aad43584172a40ec0d6b732eba9', True),
        ('drawer_success', 'runs/learning_transfer_v2_20261004/resident/F/pool/qwen/service_calls/call_553b4455bc374bcca794251871462844', True),
        ('book_conflict', prefix+'F/pool/qwen/service_calls/call_6a8910b555364cb88a58d17e73e9bda3', True),
    ]
    rows=[];manifest={}
    for i,(name,path,success) in enumerate(sources):
        p=ROOT/path/'REQUEST.json'; data=p.read_bytes(); body=json.loads(data); b=bundle(body)
        assert b['runtime_result']['success'] is success
        response=ROOT/path/'RESPONSE.json'
        archived=summarize_verdict(json.loads(response.read_text()))
        (out/(name+'.json')).write_bytes(data)
        for f in (p,response,source):manifest[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
        rows.append(dict(name=name,source=str(p.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest(),
            environment_success=success,archived_verdict=archived['verdict'],
            suite=b['session']['benchmark']['suite_id'],session_id=b['session']['session_id'],
            order=['session','episode_prompt'] if i%2==0 else ['episode_prompt','session']))
    (out/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    protocol=dict(version=1,cases=rows,max_model_calls=12,max_campaign_seconds=900,
        scope='Prompt contrast on saved Session inputs, NOT target_native reproduction or recovery execution',
        endpoints_unchanged=True,robot_episodes=0,persistent_updates=0,
        primary_question='Does the native episode recovery criterion change failure/replan choices?',
        nonclaims=['replan is not proven recoverable','no success gain','no learned efficiency','known development cases only'],
        source_selection='Three native failures from distinct task families; two native/semantic successes; one native-success semantic conflict. No selection based on new outputs.',
        continuation_rule='No automatic real retry or experience admission from these proposals')
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps(dict(cases=len(rows),model_calls=12,output=str(out))))


if __name__=='__main__':main()
