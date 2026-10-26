"""One native Agent case inside an explicitly owned resident-policy lifecycle.

Native AgentLoop, task workspace, end hooks, admission and evaluation guards are
unchanged. The wrapper only establishes and releases the external model session.
"""
import asyncio
import hashlib
import signal
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
import traceback


def valid_roster(roster):
    if not isinstance(roster, list) or len(roster) != 1 or not isinstance(roster[0], dict):
        raise ValueError('Resident native runner requires exactly one ROSTER case')
    if not isinstance(roster[0].get('id'), str) or not roster[0]['id']:
        raise ValueError('Resident native runner needs a bound case identity')
    return roster[0]


def validate_resident_case(client, case):
    model = os.environ['FASTRSI_VLA_MODEL']
    expected = 50 if model == 'pi0' else 10
    if client.expected_action_horizon != expected:
        raise ValueError('Explicit model-specific action horizon mismatch')
    return None


async def run_owned_session(native_main, client, roster, root, max_inferences, *, span_factory=nullcontext):
    """Release after every success/failure path, including uncertain begin replies."""
    case = valid_roster(roster)
    horizon_binding = validate_resident_case(client, case)
    if type(max_inferences) is not int or max_inferences < 1:
        raise ValueError('Invalid resident inference bound')
    root = Path(root)
    path = root / 'RESIDENT_SESSION.json'
    if path.exists():
        raise FileExistsError('Refusing to replace a resident Session receipt')
    started = time.time()
    session_id = client.session_id
    receipt = {'status': 'RUNNING', 'case_id': case['id'], 'resident_session_id': session_id,
               'started_at': started, 'max_inferences': max_inferences,
               'release_confirmed': False, 'native_main_called': False,
               'scope': 'Shared service lifecycle only; native experience and evaluation rules unchanged.'}
    if horizon_binding is not None:
        receipt['horizon_commitment_binding']=horizon_binding
    def save():
        path.write_text(json.dumps(receipt, indent=2) + '\n')
    def scope(name):
        return nullcontext() if span_factory is nullcontext else span_factory(name)
    save()
    failure = None
    failure_traceback = None
    cleanup_failure = None
    try:
        receipt['begin_request_id'] = 'native-begin-' + hashlib.sha256(session_id.encode()).hexdigest()
        save()
        with scope('resident_policy_begin'):
            receipt['begin'] = await asyncio.to_thread(client.begin, receipt['begin_request_id'], max_inferences)
        save()
        receipt['native_main_called'] = True
        await native_main()
        receipt['native_main_completed'] = True
    except BaseException as exc:
        failure, failure_traceback = exc, exc.__traceback__
        receipt['error'] = traceback.format_exc()
    finally:
        try:
            with scope('resident_policy_release'):
                # A timed-out begin might have succeeded. Never touch a foreign owner.
                state = await asyncio.to_thread(client.status)
                receipt['status_before_release'] = state
                active = state.get('active')
                if active is not None:
                    if not isinstance(active, dict) or active.get('session_id') != session_id:
                        raise RuntimeError('Resident service belongs to another Session; refusing cleanup')
                    receipt['end_request_id'] = 'native-end-' + hashlib.sha256(session_id.encode()).hexdigest()
                    save()
                    receipt['end'] = await asyncio.to_thread(client.end, receipt['end_request_id'])
                    state = await asyncio.to_thread(client.status)
                receipt['status_after_release'] = state
                if state.get('broken') is not False or state.get('active') is not None or state.get('owner_connected') is not False:
                    raise RuntimeError('Resident owner/session release not confirmed')
                receipt['release_confirmed'] = True
        except BaseException as exc:
            cleanup_failure = exc
            receipt['cleanup_error'] = traceback.format_exc()
        receipt.update(status='COMPLETED' if failure is None and cleanup_failure is None else 'FAILED',
                       finished_at=time.time(), seconds=time.time()-started)
        save()
    if cleanup_failure is not None:
        raise RuntimeError('Resident policy cleanup failed; service cannot be reused') from cleanup_failure
    if failure is not None:
        raise failure.with_traceback(failure_traceback)
    return receipt


async def main():
    root = Path(os.environ['FASTRSI_NATIVE_RUN'])
    roster = json.loads((root.parent / 'ROSTER.json').read_text())
    valid_roster(roster)
    from fastrsi.resident_client import from_environment
    client = from_environment()
    if client is None:
        raise ValueError('Resident runner needs explicit policy port and Session identity')
    max_inferences = int(os.environ['FASTRSI_RESIDENT_MAX_INFERENCES'])
    os.environ['FASTRSI_TRACE'] = str(root / 'OS_COST_EVENTS.jsonl')
    from cost_trace import span
    import run_native_baseline as native
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        await run_owned_session(native.main, client, roster, root, max_inferences, span_factory=span)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


if __name__ == '__main__':
    asyncio.run(main())
