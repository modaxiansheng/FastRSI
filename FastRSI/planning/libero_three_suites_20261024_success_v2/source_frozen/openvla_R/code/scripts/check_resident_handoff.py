"""Real action continuity across explicit connection handoff; no robot writes."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from replay_resident_policy import Wire
from fastrsi.resident_client import ResidentClient

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    source=Path('/project/runs/resident_services_v1_20261003')
    protocol=json.loads((source/'PROTOCOL.json').read_text())['vla_replay']
    client=ResidentClient(args.port,'handoff-continuity-check')
    receipts=[];outputs=[];started=time.time()
    result={'status':'RUNNING','scope':'Real handoff continuity; no robot or learning writes'}
    try:
        receipts.append(client.begin('handoff-check-begin',max_inferences=2))
        for i in range(2):
            if i:receipts.append(client.handoff('handoff-check-transfer'))
            spec=protocol['inputs'][i];path=Path(spec['path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
            with np.load(path,allow_pickle=False) as data:
                obs={k:data[k].copy() for k in data.files if k!='prompt'};obs['prompt']=str(data['prompt'].item())
            wire=Wire(args.port)
            try:arr=np.asarray(wire.call(obs)['actions'])
            finally:wire.close()
            np.save(args.output/f'actions_{i}.npy',arr,allow_pickle=False)
            reference=np.load(source/f'vla/replay/fresh_A_{i}.npy',allow_pickle=False)
            outputs.append({'index':i,'exact':bool(np.array_equal(arr,reference)),'max_abs':float(np.max(np.abs(arr-reference)))})
        receipts.append(client.end('handoff-check-end'))
        assert all(row['exact'] for row in outputs),'Handoff action continuity differs from native fresh sequence'
        result['status']='PASSED'
    except BaseException:
        import traceback
        result.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        result.update(receipts=receipts,comparisons=outputs,seconds=time.time()-started)
        (args.output/'RESULT.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
