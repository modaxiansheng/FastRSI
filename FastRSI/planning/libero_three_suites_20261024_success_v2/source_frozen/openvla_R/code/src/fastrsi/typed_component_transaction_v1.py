"""Opt-in original-tool transaction and lossless feedback presentation.

A candidate identifier authorizes only the existing typed human-defined program.
No proposal is inferred from prose; no execution/admission or new facts arise.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import time

TOOL_NAME = 'commit_goal_component'
CANDIDATES = ('unchanged', 'dual_anchor_v1')
ORIGINAL_REL = 'skills/goal-component-feedback/DEVELOPMENT_FEEDBACK.ORIGINAL.json'
_TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[{}\[\],:]')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def compact_feedback(raw):
    """Keep JSON semantics exactly; reject inputs too large for one native read."""
    value = json.loads(raw)
    encoded = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    tokens = _TOKEN.findall(encoded)
    if ''.join(tokens) != encoded:
        raise ValueError('Cannot losslessly tokenize feedback JSON')
    lines, line = [], ''
    for token in tokens:
        if len(token) > 2048:
            raise ValueError('An indivisible JSON token exceeds the bounded line budget')
        if line and len(line) + len(token) > 512:
            lines.append(line); line = ''
        line += token
    if line: lines.append(line)
    compact = ('\n'.join(lines) + '\n').encode('utf-8')
    rendered = '\n'.join('%d| %s' % (i+1, text) for i,text in enumerate(lines))
    rendered += '\n\n(End of file — %d lines total)' % len(lines)
    if len(lines) > 100 or len(rendered) >= 15000:
        raise ValueError('Lossless feedback does not fit one <=100-line native tool read')
    if json.loads(compact) != value:
        raise ValueError('Compact representation changed JSON facts')
    return compact, dict(original_raw_sha256=sha(raw), compact_raw_sha256=sha(compact),
        semantic_json_equal=True, lines=len(lines), max_line_characters=max(map(len,lines)),
        native_read_rendered_characters=len(rendered), scope='Formatting only; no new evidence or learning')


def stage_compact_feedback(workspace, raw, manifest, protected, *, feedback_rel, manifest_rel):
    start = time.perf_counter()
    root = Path(workspace)
    compact, receipt = compact_feedback(raw)
    original = root / ORIGINAL_REL
    if original.exists(): raise ValueError('Original feedback archive already exists')
    original.write_bytes(raw)
    (root / feedback_rel).write_bytes(compact)
    manifest['feedback_representation'] = {**receipt, 'original_path':ORIGINAL_REL,
        'seconds':time.perf_counter()-start}
    manifest['staged_sha256'].update({ORIGINAL_REL:sha(raw), feedback_rel:sha(compact)})
    blob = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    (root / manifest_rel).write_bytes(blob)
    protected.update({ORIGINAL_REL:raw, feedback_rel:compact, manifest_rel:blob})
    return receipt


def install(agent, workspace, spec, tracker, protected, validate, backend, log, *, model_config_sha256,
            candidate_catalogue=None, catalogue_path=None):
    """Register inside the existing Agent ToolRegistry, and return an uninstaller."""
    from PhyAgentOS.agent.tools.base import Tool
    from . import native_goal_component_v1 as component
    root = Path(workspace)
    if agent.tools.has(TOOL_NAME): raise ValueError('Typed component tool already registered')
    expected_ref = 'skillruntime://pi05_libero_remote'
    if spec['skillruntime_ref'] != expected_ref:
        raise ValueError('Typed transaction supports the existing base pi05 skill only')
    declaration = dict(version=component.VERSION,
        source_artifact=dict(path='configs/native_goal_component/SOURCE.json',sha256=component.SOURCE31_RAW_SHA256),
        transform=dict(version=component.TRANSFORM_VERSION,
            source_rgb=dict(path='configs/native_goal_component/SOURCE.png',sha256=component.SOURCE_RGB_RAW_SHA256),
            calibration=dict(path='configs/native_goal_component/CALIBRATION.json',sha256=component.CALIBRATION_RAW_SHA256),
            source_rois_xyxy=copy.deepcopy(component.SOURCE_ROIS),anchor_plan=copy.deepcopy(component.ANCHOR_PLAN),max_offset_m=.06),
        bindings=dict(source_goal=component.SOURCE_GOAL,model_config_sha256=model_config_sha256,
            target_ref=spec['target_ref'],action_dim=7,robot_state_dim=8))
    component.validate_declaration(declaration)
    catalogue = {'dual_anchor_v1': declaration}
    if candidate_catalogue is not None:
        if (not isinstance(catalogue_path, str) or catalogue_path not in protected
                or catalogue_path not in tracker.expected):
            raise ValueError('Successor catalogue requires an immutable native-read input')
        if json.loads(protected[catalogue_path]) != dict(version='native_goal_component_catalogue_v2', candidates=candidate_catalogue):
            raise ValueError('Successor catalogue differs from its staged native-read input')
        if not isinstance(candidate_catalogue, dict) or not candidate_catalogue or any(
                not isinstance(k, str) or not re.fullmatch(r'[A-Za-z0-9_\-]{1,80}', k)
                or k == 'unchanged' for k in candidate_catalogue):
            raise ValueError('Explicit candidate identifiers required')
        catalogue = copy.deepcopy(candidate_catalogue)
        for candidate in catalogue.values():
            component.validate_declaration(candidate)
            if candidate['version'] != 'native_goal_component_v2':
                raise ValueError('Successor catalogue requires bound actual development records')
            refs = [candidate['source_artifact'], *candidate['source_records'].values(),
                    candidate['transform']['source_rgb'], candidate['transform']['calibration']]
            for ref in refs:
                if ref['path'] not in protected or sha(protected[ref['path']]) != ref['sha256']:
                    raise ValueError('Catalogue source must be staged as immutable input')
            component.prepare_component(root, candidate, expected_goal=spec['task_description'],
                expected_model_config_sha256=model_config_sha256, expected_target_ref=spec['target_ref'])
    candidate_ids = ('unchanged', *catalogue)
    receipts = []

    class Commit(Tool):
        @property
        def name(self): return TOOL_NAME
        @property
        def description(self):
            return ('After completely reading required actual facts and current registries, explicitly keep the current '
                'component unchanged or register an explicitly provided, source-bound candidate as an unvalidated development '
                'draft. Uses the existing native registry patch and compatibility validator. Neither option is preselected. '
                'Does not execute a robot, admit a skill, or establish learning or generalization. File tools remain available.')
        @property
        def parameters(self):
            return dict(type='object',properties=dict(candidate_id=dict(type='string',enum=list(candidate_ids))),
                        required=['candidate_id'],additionalProperties=False)
        async def execute(self, candidate_id):
            start = time.perf_counter()
            before = None
            try:
                if candidate_id not in candidate_ids: raise ValueError('Unknown candidate identifier')
                if (not getattr(agent,'_fastrsi_component_updates',False)
                        or getattr(agent,'_fastrsi_component_lock',False)
                        or (root/'.FASTRSI_EVALUATION_CLOSED.json').exists()):
                    raise ValueError('Typed transaction is unavailable in evaluation or locked/closed Sessions')
                tracker.require()
                for rel,blob in protected.items():
                    if (root/rel).read_bytes()!=blob: raise ValueError('Immutable actual input changed: '+rel)
                validate()
                path=root/'SKILLRUNTIME.md'
                before=path.read_bytes()
                expected=copy.deepcopy(backend.read_yaml_block(path))
                matches=[s for s in expected['skillruntimes'] if 'skillruntime://'+s['id']==expected_ref]
                if len(matches)!=1: raise ValueError('Base native SkillRuntime absent or duplicate')
                selected_declaration = catalogue.get(candidate_id)
                if selected_declaration is not None:
                    component.prepare_component(root,selected_declaration,expected_goal=spec['task_description'],
                        expected_model_config_sha256=model_config_sha256,expected_target_ref=spec['target_ref'])
                    matches[0]['output_contract']['action']['chunk'][component.FIELD]=copy.deepcopy(selected_declaration)
                    outcome=await agent.tools.execute('patch_yaml_item',dict(path='SKILLRUNTIME.md',
                        list_key='skillruntimes',match_key='id',match_value='pi05_libero_remote',
                        patch={'output_contract':{'action':{'chunk':{component.FIELD:copy.deepcopy(selected_declaration)}}}}))
                    if outcome.startswith('Error'): raise ValueError('Native patch rejected: '+outcome)
                if backend.read_yaml_block(path)!=expected: raise ValueError('Native transaction changed unrelated fields')
                compatibility, actual = validate()
                for rel,blob in protected.items():
                    if (root/rel).read_bytes()!=blob: raise ValueError('Immutable actual input changed: '+rel)
                if candidate_id=='unchanged' and path.read_bytes()!=before:
                    raise ValueError('Unchanged option must preserve original bytes')
                if selected_declaration is not None and actual!=selected_declaration:
                    raise ValueError('Requested declaration was not actually saved')
                row=dict(candidate_id=candidate_id,status='UNCHANGED' if candidate_id=='unchanged' else 'UNVALIDATED_DRAFT_REGISTERED',
                    registry_before_sha256=sha(before),registry_after_sha256=sha(path.read_bytes()),
                    declaration=actual,complete_reads=tracker.complete(),seconds=time.perf_counter()-start,
                    empirical_admission=False,robot_episodes=0,learning_gain_claimed=False)
                backend.record(log,event='typed_component_transaction',**row)
                receipts.append(row)
                return json.dumps(row,ensure_ascii=False,allow_nan=False)
            except BaseException as exc:
                if before is not None and (root/'SKILLRUNTIME.md').read_bytes()!=before:
                    (root/'SKILLRUNTIME.md').write_bytes(before)
                if not isinstance(exc,Exception): raise
                backend.record(log,event='typed_component_transaction_rejected',candidate_id=candidate_id,
                    error=str(exc),seconds=time.perf_counter()-start,registry_restored=before is not None)
                return 'Error: '+str(exc)

    agent.tools.register(Commit())
    def restore(): agent.tools.unregister(TOOL_NAME)
    return restore, receipts
