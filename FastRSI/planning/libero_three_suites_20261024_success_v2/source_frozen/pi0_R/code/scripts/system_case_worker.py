"""One paired case, with the common resident RNG reset and native execution."""
import asyncio
import json
import os
from pathlib import Path

from PhyAgentOS.agent.session_verifier import SessionVerifier

from pi_recovery_budget_v8 import install as install_pi_budget
install_pi_budget(SessionVerifier, os.environ['FASTRSI_VLA_MODEL'], os.environ['SYSTEM_EFFICIENCY_ARM'])

original_verify = SessionVerifier.verify_session

async def captured_verify(self, session_id, *, source='tool'):
    outcome = await original_verify(self, session_id, source=source)
    if source == 'auto':
        directory = Path(self.workspace) / 'reports/evidence'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ('verification-' + session_id + '.json')
        content = json.dumps(outcome, ensure_ascii=False, indent=2)
        if path.exists() and path.read_text() != content:
            raise ValueError('Semantic verification receipt collision')
        path.write_text(content)
    return outcome

SessionVerifier.verify_session = captured_verify

if os.environ['SYSTEM_EFFICIENCY_ARM'] == 'R':
    import fastrsi.terminal_results as terminal
    from system_fast_terminal_v1 import consume_terminal
    terminal.consume_terminal = consume_terminal
    if os.environ.get('SYSTEM_NATIVE_FAILURE_RETRY') == '1':
        from system_failure_recovery_v1 import install
        install()

# User-authorized success-first v2: at most six physical attempts.
_success_v2_init = SessionVerifier.__init__
def _success_v2_construct(self, *args, **kwargs):
    _success_v2_init(self, *args, **kwargs)
    self.max_replans = 5
SessionVerifier.__init__ = _success_v2_construct
_success_v2_child = SessionVerifier._build_replanned_session
def _success_v2_build(self, session, description):
    child = _success_v2_child(self, session, description)
    child.execution.max_steps = {'libero_spatial':220, 'libero_object':280, 'libero_goal':300}[os.environ['LIBEROPRO_BASE_SUITE']]
    return child
SessionVerifier._build_replanned_session = _success_v2_build

import run_native_baseline as native
from multi_vla_binding import install as install_binding
install_binding(native)

# The frozen AgentLoop and native configuration both default to forty turns.
# The old diagnostic harness lowered this to ten; restore the native ceiling
# identically for both arms, with the full completed call cost still recorded.
original_bind_agent = native.bind_agent

def native_iteration_ceiling(*args, **kwargs):
    kwargs['max_iterations'] = 40
    return original_bind_agent(*args, **kwargs)

native.bind_agent = native_iteration_ceiling

from pi0_control_prefix_v1 import install as install_control_prefix
install_control_prefix(native, Path(__file__).with_name('PI0_CONTROL_CONFIG.json'))

from run_resident_native_baseline import main

try:
    asyncio.run(main())
finally:
    # Preserve actual attempts even if a later Agent/report stage fails. Never
    # replace the native RESULT or infer success from chain completion.
    from PhyAgentOS.runtime.state_io.markdown_yaml import read_yaml_block
    root = Path(os.environ['FASTRSI_NATIVE_RUN'])
    case = json.loads((root.parent / 'ROSTER.json').read_text())[0]
    workspace = root / case['id'] / 'workspace'
    session_file = workspace / 'SESSIONS.md'
    if session_file.exists():
        sessions = [s for s in read_yaml_block(session_file).get('sessions', [])
                    if s['session_id'] == case['id'] or s['session_id'].startswith(case['id'] + '_replan')]
        snapshot = {'case_id': case['id'], 'sessions': sessions,
                    'scope': 'Unmodified canonical Sessions after native cleanup; not a success override.'}
        path = root / case['id'] / 'CANONICAL_TERMINAL_SNAPSHOT.json'
        if path.exists():
            raise FileExistsError('Refusing to replace a canonical snapshot')
        path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
