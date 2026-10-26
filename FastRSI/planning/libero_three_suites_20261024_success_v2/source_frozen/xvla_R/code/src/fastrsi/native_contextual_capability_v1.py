"""Root-only actual-history binding over the unchanged native policy loop."""
import hashlib
import time
import numpy as np

from fastrsi.native_stage_continuation_v1 import NativeStageContinuation
from fastrsi.contextual_policy_probe_v1 import CarryAttemptSchedule, KEY, AUTHORITY


class NativeContextualCapability(NativeStageContinuation):
    def __init__(self, *, artifact_sha256, **kwargs):
        super().__init__(pause_steps=(), **kwargs)
        self.digest = artifact_sha256
        self.schedule = CarryAttemptSchedule()
        self.used = 0

    def run(self, original, runtime, skill_ctx, handle, adapter_plan, policy_client):
        if skill_ctx.session.session_id != self.session_id:
            return original(runtime, skill_ctx, handle, adapter_plan, policy_client)
        owner = self

        class TrackedHandle:
            def action_chunk(self, chunk):
                before = handle.session_state.step_index
                status = handle.action_chunk(chunk)
                consumed = handle.session_state.step_index-before
                owner.schedule.committed(chunk['actions'], consumed)
                return status
            def __getattr__(self, name): return getattr(handle, name)

        class TrackedPolicy:
            def infer(self, observation):
                if observation.get('prompt') != owner.goal:
                    raise ValueError('Root goal changed')
                tick = time.perf_counter()
                attempt = owner.schedule.request()
                owner._event('contextual_trigger_cost', seconds=time.perf_counter()-tick, attempt=attempt)
                if attempt is None:
                    return policy_client.infer(observation)
                request = dict(observation)
                request[KEY] = dict(authority=AUTHORITY,session_id=owner.session_id,
                    artifact_sha256=owner.digest,attempt=attempt)
                result = policy_client.infer(request)
                receipt = result.pop(KEY, None)
                if (not isinstance(receipt,dict) or set(receipt) != {'used','attempt','artifact_sha256'}
                        or type(receipt['used']) is not bool or receipt['attempt'] != attempt
                        or receipt['artifact_sha256'] != owner.digest):
                    raise ValueError('Missing or foreign server component reply')
                owner.schedule.reply(receipt['used'])
                owner.used += int(receipt['used'])
                owner._event('contextual_capability_result', attempt=attempt,used=receipt['used'],
                    physical_steps=handle.session_state.step_index,artifact_sha256=owner.digest,
                    actions_float32_sha256=hashlib.sha256(np.asarray(result['actions'],np.float32).tobytes()).hexdigest())
                return result
            def __getattr__(self, name): return getattr(policy_client, name)

        return super().run(original,runtime,skill_ctx,TrackedHandle(),adapter_plan,TrackedPolicy())

    def audit_snapshot(self):
        return dict(**super().audit_snapshot(), contextual_attempts=self.schedule.attempts,
                    contextual_used=self.used, immutable_component=self.digest,
                    parameter_updates_during_task=0)
