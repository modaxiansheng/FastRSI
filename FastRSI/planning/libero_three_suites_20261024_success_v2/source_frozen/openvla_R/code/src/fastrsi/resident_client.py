"""Opt-in local administrative client; never sends observations or infers.

The caller owns begin/end and logs returned receipts. Handoff preserves one
case's RNG progression while transferring its disconnected recovery client.
Only status polling is repeated; ambiguous mutations and all errors propagate.
"""
import copy
import math
import os
import time

from .resident_policy import ADMIN_KEY, VERSION


class ResidentClientError(RuntimeError):
    pass


class ResidentClient:
    def __init__(self, port, session_id, *, timeout_seconds=30, expected_action_horizon=10):
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('Resident port must be an integer in 1..65535')
        if not isinstance(session_id, str) or not 0 < len(session_id) <= 160:
            raise ValueError('Resident session_id must be a nonempty string of at most 160 characters')
        if not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('Positive finite resident timeout required')
        if type(expected_action_horizon) is not int or expected_action_horizon not in (1, 10, 20, 30, 50):
            raise ValueError('Expected resident action horizon must be explicitly 10 or 20')
        self.expected_action_horizon = expected_action_horizon
        self.port, self.session_id, self.timeout_seconds = port, session_id, float(timeout_seconds)
        self.metadata = None

    def _remaining(self, deadline):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Resident administration deadline exceeded')
        return remaining

    def _check_metadata(self, metadata):
        model = os.environ['FASTRSI_VLA_MODEL']
        expected = {'system_vla_model': model, 'wire_protocol': 'openpi_msgpack_numpy',
                    'action_dim': 7, 'action_horizon': self.expected_action_horizon}
        expected['backend'] = {'pi0':'openpi_native','xvla':'xvla_lerobot','openvla':'openvla'}[model]
        if model == 'pi0':
            expected.update(policy_config='pi0_libero', torch_compile_disabled=True)
        if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in expected.items()):
            raise ResidentClientError('Wrong resident policy identity')
        resident = metadata.get('resident', {})
        if (resident.get('version') != VERSION or resident.get('begin_required') is not True
                or resident.get('exclusive_session') is not True or not resident.get('rng_sha256')):
            raise ResidentClientError('Wrong resident control protocol')
        if self.metadata is not None and metadata != self.metadata:
            raise ResidentClientError('Resident service identity changed during operation')
        self.metadata = copy.deepcopy(metadata)

    def _request(self, command, deadline):
        # Lazy imports keep the ordinary nonresident path dependency-free.
        import websocket
        from PhyAgentOS.runtime.policy.msgpack_numpy import packb, unpackb
        wire = websocket.create_connection(f'ws://127.0.0.1:{self.port}',
            timeout=self._remaining(deadline), enable_multithread=True)
        try:
            raw = wire.recv()
            if isinstance(raw, str):
                raise ResidentClientError(raw)
            self._check_metadata(unpackb(raw))
            wire.settimeout(self._remaining(deadline))
            wire.send_binary(packb({ADMIN_KEY: command}))
            raw = wire.recv()
            if isinstance(raw, str):
                raise ResidentClientError(raw)
            result = unpackb(raw)
            if not isinstance(result, dict):
                raise ResidentClientError('Resident response must be a map')
            return result
        finally:
            wire.close()

    def _status(self, deadline):
        result = self._request({'op': 'status'}, deadline)
        if result.get('version') != VERSION or result.get('broken') is not False:
            raise ResidentClientError('Resident service is unhealthy or has wrong version')
        if type(result.get('generation')) is not int or type(result.get('owner_connected')) is not bool:
            raise ResidentClientError('Incomplete resident status')
        if result.get('rng_sha256') != self.metadata['resident']['rng_sha256']:
            raise ResidentClientError('Resident RNG identity changed')
        active = result.get('active')
        if active is not None:
            if not isinstance(active, dict) or active.get('session_id') != self.session_id:
                raise ResidentClientError('Resident service belongs to another session')
            if (active.get('generation') != result['generation']
                    or type(active.get('inferences')) is not int
                    or type(active.get('max_inferences')) is not int):
                raise ResidentClientError('Malformed active resident session')
        elif result['owner_connected']:
            raise ResidentClientError('Idle resident reports a connected inference owner')
        return result

    def status(self):
        return self._status(time.monotonic()+self.timeout_seconds)

    def _disconnected(self, deadline):
        while True:
            status = self._status(deadline)
            if status.get('active') is None:
                raise ResidentClientError('Expected active resident session')
            if not status['owner_connected']:
                return status
            time.sleep(min(0.05, self._remaining(deadline)))

    def _receipt(self, command, deadline):
        receipt = self._request(command, deadline)
        for key, expected in {'version': VERSION, 'op': command['op'],
                              'session_id': self.session_id, 'request_id': command['request_id']}.items():
            if receipt.get(key) != expected:
                raise ResidentClientError('Resident receipt identity mismatch: '+key)
        return receipt

    def _command(self, op, request_id):
        if not isinstance(request_id, str) or not 0 < len(request_id) <= 160:
            raise ValueError('Nonempty administrative request_id of at most 160 characters required')
        return {'op': op, 'session_id': self.session_id, 'request_id': request_id}

    def begin(self, request_id, max_inferences):
        if type(max_inferences) is not int or max_inferences <= 0:
            raise ValueError('Positive bounded inference count required')
        started = time.monotonic(); deadline = started+self.timeout_seconds
        command = self._command('begin', request_id); command['max_inferences'] = max_inferences
        before = self._status(deadline)
        if before.get('active') is not None:
            raise ResidentClientError('Begin requires an idle resident service')
        receipt = self._receipt(command, deadline)
        after = self._status(deadline); active = after.get('active') or {}
        if (receipt.get('generation') != before['generation']+1
                or after['generation'] != receipt['generation'] or active.get('owner') is not None
                or active.get('inferences') != 0 or active.get('max_inferences') != max_inferences):
            raise ResidentClientError('Begin did not establish a fresh bounded session')
        if receipt.get('rng_sha256') != after['rng_sha256'] or not receipt.get('rng_hashes'):
            raise ResidentClientError('Begin RNG reset receipt missing or inconsistent')
        return self._result(command, receipt, before, after, started)

    def handoff(self, request_id):
        started = time.monotonic(); deadline = started+self.timeout_seconds
        before = self._disconnected(deadline); old = before['active']
        if not old.get('owner'):
            raise ResidentClientError('Handoff requires a prior inference owner')
        command = self._command('handoff', request_id)
        command['expected_generation'] = old['generation']
        receipt = self._receipt(command, deadline)
        if (receipt.get('rng_reset') is not False or not receipt.get('rng_before_hashes')
                or receipt.get('rng_before_hashes') != receipt.get('rng_after_hashes')
                or receipt.get('previous_generation') != old['generation']
                or receipt.get('generation') != old['generation']+1
                or receipt.get('previous_owner') != old['owner']
                or receipt.get('inferences') != old['inferences']
                or receipt.get('max_inferences') != old['max_inferences']):
            raise ResidentClientError('Handoff changed RNG/budget or has an invalid generation receipt')
        after = self._status(deadline); active = after.get('active') or {}
        if (after['generation'] != receipt['generation'] or active.get('owner') is not None
                or after['owner_connected'] or active.get('inferences') != old['inferences']
                or active.get('max_inferences') != old['max_inferences']):
            raise ResidentClientError('Handoff did not release the next bounded generation')
        return self._result(command, receipt, before, after, started)

    def end(self, request_id):
        started = time.monotonic(); deadline = started+self.timeout_seconds
        before = self._disconnected(deadline)
        command = self._command('end', request_id)
        receipt = self._receipt(command, deadline)
        after = self._status(deadline)
        if (after.get('active') is not None or after['owner_connected']
                or after['generation'] != before['generation']
                or receipt.get('generation') != before['generation']
                or receipt.get('inferences') != before['active']['inferences']):
            raise ResidentClientError('End did not release this resident session')
        return self._result(command, receipt, before, after, started)

    def _result(self, command, receipt, before, after, started):
        return {'operation': command['op'], 'request': command, 'receipt': receipt,
                'status_before': before, 'status_after': after,
                'service_metadata': copy.deepcopy(self.metadata), 'seconds': time.monotonic()-started}


def from_environment(environ=None):
    environ = os.environ if environ is None else environ
    port = environ.get('FASTRSI_RESIDENT_POLICY_PORT')
    session = environ.get('FASTRSI_RESIDENT_SESSION_ID')
    horizon = environ.get('FASTRSI_RESIDENT_INFERENCE_HORIZON')
    if port is None and session is None and horizon is None:
        return None
    if not port or not session or not str(port).isdigit():
        raise ValueError('Both valid resident policy port and session ID must be supplied')
    if horizon is not None and horizon not in ('1', '10', '20', '30', '50'):
        raise ValueError('Resident horizon environment must be 10 or 20')
    return ResidentClient(int(port), session, expected_action_horizon=int(horizon or '10'))
