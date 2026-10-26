"""Authenticate a live holder of the shared full-OS flock; fail closed."""
import hashlib
import hmac
import json
import secrets
import time
from pathlib import Path
from urllib.request import Request, ProxyHandler, build_opener


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(manifest_path, token, node, name, model, controller_path):
    manifest = json.loads(Path(manifest_path).read_text())
    secret = Path(token).read_text().strip()
    assert hashlib.sha256(secret.encode()).hexdigest() == manifest['token_sha256']
    assert manifest['status'] == 'REGISTERED_BEFORE_OUTCOMES'
    assert manifest['coordinator_host'] == '10.232.1.187'
    assert manifest['endpoint'] == 'http://10.232.1.187:26370/status'
    rows = [r for r in manifest['placements'] if r['node'] == node]
    assert len(rows) == 1
    row = rows[0]
    assert (row['name'], row['model'], row['gpu_mask']) == (name, model, [0,1,2,3,4,5])
    assert sha(controller_path) == manifest['controller_sha256']
    assert sha(__file__) == manifest['authorization_source_sha256']
    nonce = secrets.token_hex(16)
    request = Request(manifest['endpoint']+'?nonce='+nonce,
                      headers={'Authorization':'Bearer '+secret})
    with build_opener(ProxyHandler({})).open(request, timeout=3) as response:
        assert response.status == 200
        payload = json.load(response)
    signature = payload.pop('signature')
    message = json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
    assert hmac.compare_digest(signature,hmac.new(secret.encode(),message,hashlib.sha256).hexdigest())
    assert payload['nonce'] == nonce
    assert payload['manifest_sha256'] == sha(manifest_path)
    assert payload['owner_pid'] == manifest['owner_pid']
    assert payload['owner_boot_id'] == manifest['owner_boot_id']
    assert payload['coordinator_source_sha256'] == manifest['coordinator_source_sha256']
    assert payload['global_flock_held'] is True
    assert abs(time.time()-payload['time']) < 15
    assert payload['status'] in ('READY','RUNNING')
    return payload
