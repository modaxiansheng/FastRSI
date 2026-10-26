"""Opt-in registry inheritance across native Sessions, not capability approval.

Call capture after copying the accepted workspace, before the demo initializer;
merge after creating fresh case bindings. SESSIONS is deliberately not restored.
Factories must already be registered by trusted host code; no dynamic imports.
"""
import copy
import hashlib
import os
from pathlib import Path
import tempfile

VERSION = 'native_registry_lifecycle_v1'
BASE_TARGET = 'libero_real_remote'
BASE_SKILL = 'pi05_libero_remote'
CASE_FIELDS = ('benchmark_name', 'task_id', 'init_state_id', 'max_steps')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _path(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('Workspace-relative registry/configuration path required')
    path = root/rel
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError('Symlink registry/configuration path is not supported')
    return path


def _documents(root):
    from PhyAgentOS.runtime.state_io.markdown_yaml import read_yaml_block
    return (read_yaml_block(_path(root, 'TARGETS.md')),
            read_yaml_block(_path(root, 'SKILLRUNTIME.md')))


def _validate(targets, skills, config_bytes):
    from PhyAgentOS.runtime.schemas import TargetsDocument, SkillRuntimeDocument, TargetRuntimeContractDocument
    from PhyAgentOS.runtime.adapters import factory as adapters
    from PhyAgentOS.runtime.targets import factory as targets_factory
    from PhyAgentOS.runtime.watchdog.runtime_registry import _SKILL_FACTORIES
    def registered_uri(value, prefix, factories):
        try:
            key = adapters._require_uri(value, prefix)
        except Exception as exc:
            raise ValueError('Invalid native adapter URI') from exc
        return key in factories
    tdoc = TargetsDocument.model_validate(targets)
    sdoc = SkillRuntimeDocument.model_validate(skills)
    tids = [t.id for t in tdoc.targets]
    sids = [s.id for s in sdoc.skillruntimes]
    if len(set(tids)) != len(tids) or tids.count(BASE_TARGET) != 1:
        raise ValueError('Exactly one base target and unique target IDs required')
    if len(set(sids)) != len(sids) or sids.count(BASE_SKILL) != 1:
        raise ValueError('Exactly one base skill and unique SkillRuntime IDs required')
    base_target = next(t for t in tdoc.targets if t.id == BASE_TARGET)
    if BASE_SKILL not in base_target.supported_skillruntimes:
        raise ValueError('Base target must support the fresh Session base skill')
    for skill in sdoc.skillruntimes:
        if skill.runtime not in _SKILL_FACTORIES:
            raise ValueError('Skill runtime factory must already be registered: '+skill.runtime)
        if skill.policy:
            if not registered_uri(skill.policy.policy_adapter, 'policy_adapter://', adapters._POLICY_ADAPTER_FACTORIES):
                raise ValueError('Policy adapter factory must already be registered: '+skill.policy.policy_adapter)
            # policy_client is metadata in this SDK; actual client selection is
            # from the fresh Session endpoint and stays with native preflight.
    for target in tdoc.targets:
        factories = (targets_factory._REMOTE_TARGET_FACTORIES if target.target_class == 'remote'
                     else targets_factory._LOCAL_TARGET_FACTORIES)
        if target.runtime.target_runtime not in factories:
            raise ValueError('Target runtime factory must already be registered')
        if not registered_uri(target.runtime.target_adapter, 'target_adapter://', adapters._TARGET_ADAPTER_FACTORIES):
            raise ValueError('Target adapter factory must already be registered')
        if not set(target.supported_skillruntimes) <= set(sids):
            raise ValueError('Target refers to absent SkillRuntime')
        contract = Path(target.runtime.runtime_contract_ref)
        if contract.is_absolute() or '..' in contract.parts or not contract.parts or contract.parts[0] != 'configs':
            raise ValueError('Runtime contracts must be captured workspace configs')
        if contract.as_posix() not in config_bytes:
            raise ValueError('Referenced runtime contract is missing')
        import yaml
        native_contract = TargetRuntimeContractDocument.model_validate(yaml.safe_load(config_bytes[contract.as_posix()]))
        if native_contract.target_id != target.id or native_contract.target_adapter != target.runtime.target_adapter:
            raise ValueError('Runtime contract target/adapter binding mismatch')
    return next(t for t in targets['targets'] if t['id'] == BASE_TARGET)


def _configs(root):
    from .native_registry_files_v1 import read_config_files
    return read_config_files(root)


def capture_inherited(workspace, *, enabled=False):
    """Capture exact inherited registry/config bytes before demo overwrites them."""
    if type(enabled) is not bool:
        raise ValueError('Boolean opt-in required')
    if not enabled:
        return None
    root = Path(workspace).resolve()
    targets, skills = _documents(root)
    configs = _configs(root)
    _validate(targets, skills, configs)
    files = {name:_path(root,name).read_bytes() for name in ('TARGETS.md','SKILLRUNTIME.md')}
    files.update(configs)
    return dict(version=VERSION, workspace=str(root), files=files,
                sha256={name:_sha(raw) for name,raw in files.items()},
                targets=copy.deepcopy(targets), skills=copy.deepcopy(skills),
                capability_status='inherited_unvalidated')


def merge_with_fresh_bindings(workspace, snapshot, *, enabled=False):
    """Validate the whole merge before writes; leave fresh SESSIONS untouched.

    Schema/factory checks establish structural consumability only. They do not
    certify a candidate's physical efficacy or replace native preflight.
    """
    if type(enabled) is not bool:
        raise ValueError('Boolean opt-in required')
    if not enabled:
        return dict(version=VERSION,status='DISABLED',changed_files=[])
    if snapshot is None:
        return dict(version=VERSION,status='NO_INHERITED_WORKSPACE',changed_files=[])
    root=Path(workspace).resolve()
    if snapshot['version'] != VERSION or snapshot['workspace'] != str(root):
        raise ValueError('Inherited snapshot belongs to another workspace/version')
    files=snapshot['files']
    if {name:_sha(raw) for name,raw in files.items()} != snapshot['sha256']:
        raise ValueError('Captured inherited bytes changed')
    # Parse the exact captured bytes, never trust mutable parallel documents.
    from PhyAgentOS.runtime.state_io.markdown_yaml import _YAML_BLOCK_RE
    import yaml
    def parse(name):
        match=_YAML_BLOCK_RE.search(files[name].decode('utf-8'))
        if match is None:raise ValueError('Captured YAML block missing')
        return yaml.safe_load(match.group('body'))
    targets,skills=parse('TARGETS.md'),parse('SKILLRUNTIME.md')
    configs={n:b for n,b in files.items() if Path(n).parts[0]=='configs'}
    original=_validate(targets,skills,configs)
    fresh_targets,fresh_skills=_documents(root)
    fresh_configs=_configs(root)
    fresh=_validate(fresh_targets,fresh_skills,fresh_configs)
    for field in CASE_FIELDS:
        if field not in fresh.get('config',{}):
            raise ValueError('Fresh base target lacks case field: '+field)
    if not isinstance(fresh['config']['benchmark_name'],str) or not fresh['config']['benchmark_name']:
        raise ValueError('Fresh benchmark name required')
    for field in ('task_id','init_state_id','max_steps'):
        if type(fresh['config'][field]) is not int or fresh['config'][field] < (1 if field=='max_steps' else 0):
            raise ValueError('Invalid fresh case binding')
    endpoint=fresh['runtime'].get('target_endpoint')
    from urllib.parse import urlparse
    parsed=urlparse(endpoint or '')
    if parsed.scheme!='targetws' or not parsed.hostname or parsed.port is None:
        raise ValueError('Explicit fresh TargetWS endpoint required')
    original.setdefault('config',{}).update({k:copy.deepcopy(fresh['config'][k]) for k in CASE_FIELDS})
    original['runtime']['target_endpoint']=endpoint
    _validate(targets,skills,configs)
    updates=dict(configs)
    updates['SKILLRUNTIME.md']=files['SKILLRUNTIME.md']
    # Preserve all Markdown instructions and fence bytes outside the native
    # YAML body; these may themselves be legitimate Agent context.
    target_text=files['TARGETS.md'].decode('utf-8')
    target_match=_YAML_BLOCK_RE.search(target_text)
    body_start,body_end=target_match.span('body')
    target_body=yaml.safe_dump(targets,sort_keys=False,allow_unicode=True).rstrip('\n')
    updates['TARGETS.md']=(target_text[:body_start]+target_body+target_text[body_end:]).encode('utf-8')
    for name in updates:
        target=_path(root,name)
        if target.exists() and not target.is_file():
            raise ValueError('Configuration write target is not a regular file')
        for parent in target.parents:
            if parent == root:break
            if parent.exists() and not parent.is_dir():raise ValueError('Configuration parent is not a directory')
    changed=[name for name,raw in updates.items() if not (root/name).exists() or (root/name).read_bytes()!=raw]
    # All structural validation is complete. Stage files before replacing any destination.
    staged=[]
    try:
        for name in changed:
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
            fd,tmp=tempfile.mkstemp(prefix='.registry-lifecycle-',dir=str(target.parent))
            staged.append((Path(tmp),target))
            with os.fdopen(fd,'wb') as stream:stream.write(updates[name])
        for tmp,target in staged:os.replace(tmp,target)
    finally:
        for tmp,_ in staged:
            if tmp.exists():tmp.unlink()
    return dict(version=VERSION,status='MERGED',capability_status='inherited_unvalidated',
                refreshed_target=BASE_TARGET,refreshed_fields=['config.'+k for k in CASE_FIELDS]+['runtime.target_endpoint'],
                changed_files=changed,inherited_sha256=snapshot['sha256'],
                restored_config_sha256={n:_sha(b) for n,b in configs.items()},
                skillruntime_sha256=_sha(files['SKILLRUNTIME.md']),
                preserved_target_ids=[t['id'] for t in targets['targets']],
                preserved_skillruntime_ids=[s['id'] for s in skills['skillruntimes']],
                scope='Registry lifecycle only; no executable capability approval, policy update or Session history restore')
