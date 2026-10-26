"""Actual saved planned commands equal the declared single/equal-mean rule."""
from pathlib import Path
import hashlib,json
import numpy as np
from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb
P=Path('/project');checked=0;cases=[]
for slot in (0,1,2,3):
    root=P/'runs'/('bowl_mean_validation_gpu%d_v1_20261010'%slot)
    for case in json.loads((root/'AGGREGATE.json').read_text())['cases']:
        folder=root/case['case'];result=json.loads((folder/'RESULT.json').read_text())
        assert result['status']=='COMPLETED'
        for row in result['rows']:
            i=row['chunk'];command=folder/('%03d_command.npz'%i)
            proposed=np.load(command,allow_pickle=False)['planned_actions']
            candidates=np.asarray(unpackb((folder/('%03d_response.msgpack'%i)).read_bytes())['candidates'])
            expected=candidates[0] if case['arm']=='single' else np.mean(candidates,axis=0)
            assert np.array_equal(proposed,expected) and proposed.shape==(10,7) and np.isfinite(proposed).all()
            initial=case['initial'];noise_base=20261010+(initial-32)//2
            for j,digest in enumerate(row['candidate_noise_sha256']):
                noise=np.random.default_rng(np.random.SeedSequence([noise_base,i,j])).standard_normal((10,32)).astype(np.float32)
                assert hashlib.sha256(noise.tobytes()).hexdigest()==digest
            assert row['combination']==('single_native' if case['arm']=='single' else 'equal_mean_of_two_native_candidates')
            checked+=1
        cases.append(dict(case=case['case'],commands=len(result['rows']),all_planned_commands_exact=True))
report=dict(status='VERIFIED_ACTUAL_SAVED_COMBINATIONS',cases=cases,commands=checked,training_steps=0)
Path('/check/COMMAND_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
