"""Opt-in stage instruction binding for the original pi0.5 adapter.

The native Session retains its complete goal for verification and memory. Only
the policy request uses the current, predeclared stage instruction. Registration
is explicit and leaves the ordinary OpenPI adapter URI unchanged.
"""
from dataclasses import dataclass

from PhyAgentOS.runtime.adapters.factory import register_policy_adapter
from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter
from PhyAgentOS.runtime.watchdog.errors import AdapterError


ADAPTER_URI = 'policy_adapter://fastrsi_same_world_pi05_v1'


@dataclass(frozen=True)
class StageInstructionBinding:
    session_id: str
    joint_goal: str
    stage_instructions: tuple[str, ...]

    def __post_init__(self):
        for name, value in (('session_id', self.session_id), ('joint_goal', self.joint_goal)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'Nonempty {name} required')
        instructions = tuple(self.stage_instructions)
        if (isinstance(self.stage_instructions, str) or not instructions
                or any(not isinstance(x, str) or not x.strip() for x in instructions)
                or len(set(instructions)) != len(instructions)):
            raise ValueError('Distinct nonempty stage instructions required')
        object.__setattr__(self, 'stage_instructions', instructions)


class SameWorldPi05Adapter(OpenPIPi05Adapter):
    def __init__(self, binding: StageInstructionBinding):
        if not isinstance(binding, StageInstructionBinding):
            raise TypeError('Explicit StageInstructionBinding required')
        self.binding = binding

    def to_policy_input(self, runtime_observation, session_ctx):
        if (session_ctx.get('session_id') != self.binding.session_id
                or session_ctx.get('task_description') != self.binding.joint_goal):
            raise AdapterError('Same-world Session identity or joint goal changed')
        libero = runtime_observation.get('libero')
        instruction = libero.get('task_description') if isinstance(libero, dict) else None
        if not isinstance(instruction, str) or instruction not in self.binding.stage_instructions:
            raise AdapterError('Missing or unregistered same-world stage instruction')
        # The outer Session and its complete verification goal are never edited.
        stage_ctx = dict(session_ctx, task_description=instruction)
        return super().to_policy_input(runtime_observation, stage_ctx)


def register_same_world_policy_adapter(*, session_id, joint_goal, stage_instructions):
    """Bind one native-run Session before constructing its Watchdog adapters.

    Set the selected SKILLRUNTIME entry's policy.policy_adapter to the returned
    URI. Rebinding is per runner process; existing adapter instances retain their
    immutable binding. The registered factory accepts no arguments, as required
    by the native build_policy_adapter/build_adapter_stack interfaces.
    """
    binding = StageInstructionBinding(session_id, joint_goal, stage_instructions)
    register_policy_adapter(ADAPTER_URI, lambda: SameWorldPi05Adapter(binding))
    return ADAPTER_URI
