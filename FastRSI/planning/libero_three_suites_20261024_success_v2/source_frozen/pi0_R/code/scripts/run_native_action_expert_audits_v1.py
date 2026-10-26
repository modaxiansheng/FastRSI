"""CPU audit of TERMINAL registered host half, including actual submitted actions."""
import hashlib,json,subprocess,time,sys
from pathlib import Path
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
slot=int(sys.argv[1]);assert slot in (0,1)
B=P/f"planning/native_action_expert_batch_host{slot}_v{1 if slot==0 else 2}_20261010"
assert json.loads((B/'STATUS.json').read_text())['status']=='COMPLETED'
template=json.loads((P/'planning/native_value_phase_v12_20261010_audit/LAUNCH.json').read_text())
results=[]
for version in range(16+slot*2,18+slot*2):
    name=f'native_value_phase_v{version}_20261010';root=P/'runs'/name
    assert json.loads((root/'STATUS.json').read_text())['status']=='COMPLETED'
    O=P/'planning'/(name+'_expert_audit');O.mkdir(exist_ok=False)
    container=f'yuchen-fastrsi-native-expert-v{version}-audit-v1-20261010'
    argv=[x.replace('yuchen-fastrsi-native-paired-v12-audit-v1-20261010',container)
        .replace(str(P/'planning/native_value_phase_v12_20261010_audit'),str(O)) for x in template]
    argv[-1]=('source /opt/dtk/env.sh && /project/envs/os-v1/bin/python '
        '/project/scripts/audit_native_action_expert_v1.py /project/runs/'+name+' /check/report_v1')
    (O/'LAUNCH.json').write_text(json.dumps(argv,indent=2)+'\n');tick=time.perf_counter()
    try:
        with (O/'LOG.txt').open('x') as log:
            result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        assert result.returncode==0,result.returncode
        audited=json.loads((O/'report_v1/AUDIT.json').read_text())
        assert audited['status']=='PASSED_NATIVE_BINDING_AUDIT'
        results.append(dict(campaign=name,**audited))
    finally:
        record=json.loads(subprocess.check_output(['docker','inspect',container]))[0]
        if record['State']['Running']:subprocess.run(['docker','stop','-t','10',container],check=True,capture_output=True)
        record=json.loads(subprocess.check_output(['docker','inspect',container]))[0]
        assert not record['State']['Running'] and record['State']['Pid']==0
        (O/'STOP_RECEIPT.json').write_text(json.dumps(dict(name=record['Name'],image=record['Image'],state=record['State'],seconds=time.perf_counter()-tick),indent=2)+'\n')
(B/'AUDITS.json').write_text(json.dumps(dict(status='PASSED',host_slot=slot,cases=results),indent=2)+'\n')
print(json.dumps(dict(status='PASSED',host_slot=slot,commands=sum(r['checked_commands'] for r in results),cases=[r['cases'] for r in results])),flush=True)
