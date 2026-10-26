"""Strict, lossless JSON for receipts and process-shared append-only evidence."""
import fcntl
import json
import math
import os
import tempfile
from pathlib import Path


class EvidenceError(ValueError):
    pass


def plain(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise EvidenceError('non-finite numeric evidence')
        return value
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        if any(type(k) is not str for k in value):
            raise EvidenceError('JSON object keys must be strings')
        return {k: plain(v) for k, v in value.items()}
    # Import numpy only in environments that actually use it. Never stringify
    # unsupported objects: a stringified array is not quantitative evidence.
    if type(value).__module__.split('.')[0] == 'numpy':
        import numpy as np
        if isinstance(value, np.ndarray):
            return plain(value.tolist())
        if isinstance(value, np.generic):
            return plain(value.item())
    raise EvidenceError('unsupported evidence type: ' + type(value).__name__)


def encode(value):
    return (json.dumps(plain(value), ensure_ascii=False, allow_nan=False,
                       separators=(',', ':')) + '\n').encode('utf-8')


def strict_loads(data):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise EvidenceError('duplicate JSON key: ' + k)
            out[k] = v
        return out
    def invalid(value):
        raise EvidenceError('non-finite JSON number: ' + value)
    return plain(json.loads(data, object_pairs_hook=pairs, parse_constant=invalid))


def atomic_receipt(path, value):
    """Create exactly once. A failure never replaces an existing receipt."""
    path = Path(path)
    payload = encode(value)  # Validate before filesystem mutation.
    fd, temp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        os.link(temp, str(path))  # Atomic no-clobber publication.
    finally:
        os.unlink(temp)


class Ledger:
    """Cooperative lock per append; damaged tails stop further writes.

    Existing history is never truncated/repaired automatically. This writer
    must be used by every producer of a NEW ledger; do not attach it to an old
    actively written ledger whose producers do not take the same lock.
    """
    def __init__(self, path, sync=False):
        self.path = Path(path)
        self.sync = sync
        self.checked_size = None

    def emit(self, row):
        if not isinstance(row, dict) or not isinstance(row.get('event'), str) or not row['event']:
            raise EvidenceError('event name required')
        payload = encode(row)
        with self.path.open('a+b', buffering=0) as out:
            fcntl.flock(out.fileno(), fcntl.LOCK_EX)
            out.seek(0, os.SEEK_END)
            size = out.tell()
            if self.checked_size is None or size < self.checked_size:
                # Full validation once per writer, before extending any history.
                out.seek(0)
                for line in out:
                    if not line.endswith(b'\n'):
                        raise EvidenceError('incomplete ledger; preserve and stop')
                    item = strict_loads(line)
                    if not isinstance(item, dict) or not item.get('event'):
                        raise EvidenceError('invalid ledger event')
            elif size:
                out.seek(-1, os.SEEK_END)
                if out.read(1) != b'\n':
                    raise EvidenceError('incomplete ledger; preserve and stop')
            out.seek(0, os.SEEK_END)
            offset = 0
            while offset < len(payload):
                count = os.write(out.fileno(), payload[offset:])
                if count <= 0:
                    raise OSError('incomplete ledger write')
                offset += count
            if self.sync:
                os.fsync(out.fileno())
            self.checked_size = size + len(payload)


def measured_cost(*, actions=None, policy_inferences=None, forward_calls=None,
                  prompt_tokens=None, completion_tokens=None, seconds=None,
                  money=None):
    """Unknown stays null. Counts never infer money or time from GPU usage."""
    counts = dict(actions=actions, policy_inferences=policy_inferences,
                  forward_calls=forward_calls, prompt_tokens=prompt_tokens,
                  completion_tokens=completion_tokens)
    for key, value in counts.items():
        if value is not None and (type(value) is not int or value < 0):
            raise EvidenceError('invalid count: ' + key)
    for key, value in [('seconds', seconds), ('money', money)]:
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise EvidenceError('invalid amount: ' + key)
    return dict(counts, seconds=seconds, money=money)
