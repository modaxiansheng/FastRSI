"""Owned four-device offline replay; one lock, immutable outputs, bounded exit."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fcntl,hashlib,json,re,shutil,subprocess,time
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
A=P.parent/'frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1'
R=P/'planning/phase_value_features_v1_20261010'
def cap(argv):
    p=subprocess.run(argv,capture_output=True,text=True,timeout=45)
    return dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
def save(name,obj):(R/name).write_text(json.dumps(obj,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    lock=(P/'runs/full_os_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    R.mkdir(exist_ok=False);before=sha(P/'CURRENT_NATIVE_ENTRY.json')
    pre=dict(hcu=cap(['bash','-lc','source /opt/dtk/env.sh && source /opt/hyhal/env.sh && hy-smi --showuse --showmemuse']),
        containers=cap(['docker','ps','--format','{{.Names}}']),holders=cap(['bash','-lc','fuser -v /dev/kfd /dev/mkfd /dev/dri/render*']))
    save('PREFLIGHT.json',pre)
    mem=re.findall(r'HCU\[(\d+)\].*?HCU memory use \(%\):\s*([0-9.]+)',pre['hcu']['stdout'])
    assert len(mem)==8 and all(float(v)==0 for _,v in mem)
    assert not pre['containers']['stdout'].strip() and pre['containers']['returncode']==0
    identities=[]
    for pid in set(map(int,pre['holders']['stdout'].split())):
        exe=str((Path('/proc')/str(pid)/'exe').resolve());comm=(Path('/proc')/str(pid)/'comm').read_text().strip()
        identities.append(dict(pid=pid,exe=exe,comm=comm));assert comm=='hymgr' and exe=='/usr/local/hyhal/bin/hymgr'
    save('HOLDER_IDENTITIES.json',identities)
    assert json.loads((P/'planning/prefix_feature_check_v1_20261010/RESULT.json').read_text())['status']=='PASSED'
    (R/'code').mkdir()
    for n in ('phase_value_export_v1.py','phase_value_features_v1.py','prefix_feature_capture_v1.py','run_phase_value_features_v1.py'):
        shutil.copy2(P/'scripts'/n,R/'code'/n)
    save('PROTOCOL.json',dict(robots=0,pi_updates=0,devices=[0,1,2,3],shards=4,
        dataset='phase_value_dataset_v1_20261010',goal='Prepare actual transitions and unchanged frozen representation, not a benefit claim',
        default_sha256=before,code_sha256={f.name:sha(f) for f in (R/'code').iterdir()}))
    common=['docker','run','--network','none','--pids-limit','1024','-e','PYTHONDONTWRITEBYTECODE=1',
        '-v',str(P)+':/project:ro','-v',str(A/'dependencies/PhyAgentOS-core')+':/core:ro']
    exportname='yuchen-fastrsi-phase-value-export-v1-20261010'
    cmd=common+['--name',exportname,'--cpus','2','--memory','4g','--entrypoint','python','-e','PYTHONPATH=/core',
        '-v',str(P/'planning')+':/planning',
        'sha256:3171fda2362d773433482f79c4fd53f6f428413efbf41c3fdfb43d8c799d6c95',
        '/project/scripts/phase_value_export_v1.py','--project','/project','--output','/planning/phase_value_dataset_v1_20261010']
    save('EXPORT_LAUNCH.json',cmd);tick=time.perf_counter()
    with (R/'EXPORT.log').open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    save('EXPORT_HOST.json',dict(exit_code=p.returncode,seconds=time.perf_counter()-tick));assert p.returncode==0
    def worker(slot):
        root=R/f'gpu{slot}';root.mkdir();shutil.copytree(P/'models/openpi_data',root/'tokenizer_cache')
        name=f'yuchen-fastrsi-phase-value-features-gpu{slot}-v1-20261010'
        cmd=common+['--name',name,'--cpus','8','--memory','64g','--shm-size','2g',
            '--device','/dev/kfd','--device','/dev/mkfd','--device','/dev/dri','--group-add','video','--group-add','render',
            '-e',f'HIP_VISIBLE_DEVICES={slot}','-e',f'CUDA_VISIBLE_DEVICES={slot}','-e',f'FEATURE_SLOT={slot}',
            '-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4','-e','OPENPI_DATA_HOME=/workspace/openpi_data',
            '-e','PYTHONPATH=/code:/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src:/core',
            '-v',str(root)+':/check','-v',str(R/'code')+':/code:ro',
            '-v',str(root/'tokenizer_cache')+':/workspace/openpi_data',
            '-v',str(A/'dependencies/openpi')+':/workspace/openpi:ro',
            '-v',str(A/'weights/pi05_libero_pytorch_fp32')+':/model:ro',
            '-v','/opt/hyhal:/opt/hyhal:ro','-v','/etc/hfm:/etc/hfm:ro',
            'sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81',
            'timeout','1200','bash','-lc','set -e; source /opt/dtk/env.sh; source /opt/hyhal/env.sh; exec python /code/phase_value_features_v1.py']
        (root/'LAUNCH.json').write_text(json.dumps(cmd,indent=2));tick=time.perf_counter();code=-1
        try:
            with (root/'FEATURE.log').open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1250);code=p.returncode
        finally:
            info=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
            if info['State']['Running']:cap(['docker','stop','-t','10',name]);info=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
            (root/'CONTAINER.json').write_text(json.dumps(info,indent=2))
            report=dict(slot=slot,exit_code=code,seconds=time.perf_counter()-tick,state=info['State'])
            (root/'HOST.json').write_text(json.dumps(report,indent=2))
        return report
    tick=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:reports=list(pool.map(worker,range(4)))
    save('HOST_RESULT.json',dict(shards=reports,parallel_seconds=time.perf_counter()-tick,default_unchanged=before==sha(P/'CURRENT_NATIVE_ENTRY.json')))
    assert all(r['exit_code']==0 and not r['state']['Running'] for r in reports)
    results=[json.loads((R/f'gpu{i}/RESULT.json').read_text()) for i in range(4)]
    save('RESULT.json',dict(status='COMPLETED',requests=sum(r['requests'] for r in results),
        inference_seconds=sum(r['wall_seconds'] for r in results),robots=0,training_steps=0,default_unchanged=before==sha(P/'CURRENT_NATIVE_ENTRY.json')))
    print((R/'RESULT.json').read_text(),flush=True)
if __name__=='__main__':main()
