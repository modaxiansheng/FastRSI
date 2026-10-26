"""Saved legal inputs, native fresh policy versus resident begin/reset policy."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import websocket
from PhyAgentOS.runtime.policy.msgpack_numpy import packb, unpackb


class Wire:
    def __init__(self, port):
        self.ws = websocket.create_connection(f'ws://127.0.0.1:{port}', timeout=120,
                                               enable_multithread=True)
        self.metadata = unpackb(self.ws.recv())

    def call(self, request):
        self.ws.send_binary(packb(request))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(response)
        return unpackb(response)

    def close(self):
        self.ws.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    protocol = json.loads(args.protocol.read_text())['vla_replay']
    out = args.output; out.mkdir(exist_ok=False, parents=True)
    inputs = []
    for spec in protocol['inputs']:
        path = Path(spec['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec['sha256']
        with np.load(path, allow_pickle=False) as values:
            observation = {key: values[key].copy() for key in values.files if key != 'prompt'}
            observation['prompt'] = str(values['prompt'].item())
        inputs.append(observation)
    assert len(inputs) == 3
    started = time.time(); calls = []; admin_receipts = []
    result = {'status': 'RUNNING', 'started_at': started, 'scope': 'Real model replay only; no robot or learning updates'}
    def save():
        result.update(calls=calls, admin_receipts=admin_receipts,
                      elapsed_seconds=time.time()-started)
        (out/'RESULT.json').write_text(json.dumps(result, indent=2))
    def invoke(port, items, label):
        wire = Wire(port)
        outputs = []
        try:
            for index in items:
                begin = time.perf_counter()
                response = wire.call(inputs[index])
                arr = np.asarray(response['actions'])
                assert arr.shape == (10,7) and np.isfinite(arr).all()
                filename = f'{label}_{len(outputs)}.npy'
                np.save(out/filename, arr, allow_pickle=False)
                calls.append({'label':label,'input_index':index,'actions_file':filename,
                              'seconds':time.perf_counter()-begin,'shape':list(arr.shape),'dtype':str(arr.dtype),
                              'actions_sha256':hashlib.sha256(arr.tobytes()).hexdigest(),
                              'metadata':wire.metadata})
                outputs.append(arr.copy()); save()
        finally:
            wire.close()
        return outputs
    def admin(op, session, request_id, **kw):
        wire = Wire(protocol['resident_port'])
        try:
            response = wire.call({'__fastrsi_resident__': {'op':op,'session_id':session,
                                 'request_id':request_id,**kw}})
            assert response['version'] == 'resident_openpi_postload_reset_v1'
            if op != 'status':
                assert response['op'] == op and response['session_id'] == session
                assert response['request_id'] == request_id
            admin_receipts.append({'request':{'op':op,'session_id':session,'request_id':request_id,**kw},
                                   'response':response}); save()
            return response
        finally:
            wire.close()
    def end(session, request_id):
        until = time.monotonic()+5
        while True:
            state = admin('status',session,request_id+'-status')
            assert not state['broken']
            if not state['owner_connected']:
                return admin('end',session,request_id)
            assert time.monotonic()<until, 'Inference connection did not close'
            time.sleep(.05)
    try:
        fresh = invoke(protocol['fresh_port'], [0,1], 'fresh_A')
        first = admin('begin','resident-before','begin-before',max_inferences=3)
        retry = admin('begin','resident-before','begin-before',max_inferences=3)
        assert first == retry, 'Begin retry must not reset or replace its receipt'
        before = invoke(protocol['resident_port'], [0,1,2], 'resident_B_before_and_exposure')
        end('resident-before','end-before')
        admin('begin','resident-after','begin-after',max_inferences=2)
        after = invoke(protocol['resident_port'], [0,1], 'resident_B_after')
        end('resident-after','end-after')
        comparisons = []
        for i in range(2):
            comparisons.append({'input_index':i,
                'fresh_equals_resident_before':np.array_equal(fresh[i],before[i]),
                'fresh_equals_resident_after':np.array_equal(fresh[i],after[i]),
                'max_abs_before':float(np.max(np.abs(fresh[i]-before[i]))),
                'max_abs_after':float(np.max(np.abs(fresh[i]-after[i])))})
        result['comparisons'] = comparisons
        assert all(r['fresh_equals_resident_before'] and r['fresh_equals_resident_after'] for r in comparisons), 'Actions differ; resident equivalence not established'
        result['status'] = 'PASSED'
    except BaseException:
        import traceback
        result.update(status='FAILED',error=traceback.format_exc())
        raise
    finally:
        result['finished_at'] = time.time(); save()


if __name__ == '__main__':
    main()
