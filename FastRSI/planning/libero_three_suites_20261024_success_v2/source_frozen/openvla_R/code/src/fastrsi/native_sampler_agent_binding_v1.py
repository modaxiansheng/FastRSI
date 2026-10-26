"""Explicit sampler binding for the unchanged native Agent recovery lifecycle.

One root plus at most two native verifier child Sessions. Each Session pays its
own exact native draw and 400-action budget. This is not same-episode recovery.
Sampling identity is evaluator bookkeeping, never a selector feature.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import time
import yaml

from .native_sampler_draw_restore_v1 import NativeSamplerDrawRestore

VERSION = 'native_sampler_agent_binding_v1'
_YAML = re.compile(r'(?P<fence>`{3,}|~{3,})\s*yaml\s*\n(?P<body>.*?)(?:\n(?P=fence)\s*)', re.DOTALL | re.IGNORECASE)
_FIELDS = ('session_id','parent_session_id','replan_attempt','target_ref','skillruntime_ref',
           'task_description','verification_profile','execution','benchmark','routing')


# Native SessionRegistry._save_unlocked serializes exclude_none=True, while
# SessionRunner.reset sends model_dump(mode="json") including these nulls.
# Only nullable fields declared by the frozen native model are normalized.
# Never strip arbitrary nested dict entries (e.g. executable steps parameters).
_NATIVE_NULLABLE = {
    'execution': frozenset(('control_hz','replan_every_steps','steps','timeline')),
    'routing': frozenset(('target_endpoint','policy_endpoint','adapter_overrides')),
    'benchmark': frozenset(('benchmark_id','suite_id','task_name','task_index','instance_id','policy_id','run_id')),
}

def native_serialized_field(key, value):
    if key not in _NATIVE_NULLABLE or not isinstance(value,dict):return value
    return {name:copy.deepcopy(item) for name,item in value.items()
            if not (name in _NATIVE_NULLABLE[key] and item is None)}


def read_sessions(workspace):
    path = Path(workspace)/'SESSIONS.md'
    if path.is_symlink():
        raise ValueError('Session registry cannot be a symlink')
    raw = path.read_bytes()
    match = _YAML.search(raw.decode())
    if match is None:
        raise ValueError('Missing native Session YAML block')
    doc = yaml.safe_load(match.group('body'))
    if not isinstance(doc,dict) or doc.get('version') != 'runtime_sessions_v1':
        raise ValueError('Invalid native Session document')
    rows = doc.get('sessions')
    if not isinstance(rows,list) or any(not isinstance(r,dict) for r in rows):
        raise ValueError('Invalid native Session list')
    ids = [r.get('session_id') for r in rows]
    if any(not isinstance(s,str) for s in ids) or len(ids) != len(set(ids)):
        raise ValueError('Missing or duplicate native Session identity')
    return {r['session_id']:r for r in rows},hashlib.sha256(raw).hexdigest()


def benchmark_metadata(value, run_id):
    result = copy.deepcopy(value)
    if (result.get('suite_id'),result.get('task_index')) != ('libero_90',64):
        raise ValueError('Sampler extension requires the original task64')
    result.update(instance_id=None,task_name='native_sampler_extension_task64',run_id=run_id)
    return result


class NativeSamplerAgentBinding:
    def __init__(self, config, *, runtime_source, robot8):
        if config.get('version') != VERSION:
            raise ValueError('Explicit native sampler Agent binding required')
        if set(config) != {'version','workspace','run_id','restore'}:
            raise ValueError('Unknown or missing native sampler binding fields')
        self.config = copy.deepcopy(config)
        self.workspace = Path(config['workspace']).resolve()
        self.base = copy.deepcopy(config['restore'])
        self.root_session = self.base['root_session']
        self.run_id = config['run_id']
        if not isinstance(self.run_id,str) or not self.run_id.startswith('native_sampler_'):
            raise ValueError('Explicit nonofficial sampler run identifier required')
        self.runtime_source,self.robot8 = runtime_source,robot8
        # Validate the fixed registration, role, seed and all source files now.
        prototype = self._restorer(self.root_session)
        self.goal = prototype.registration['task_description']
        self.input_role = prototype.input_role
        self.receipts = []
        self.last_receipt = None
        self.used = set()
        self.root_core = None

    def _restorer(self, sid):
        config = copy.deepcopy(self.base)
        config['root_session'] = sid
        return NativeSamplerDrawRestore(config,runtime_source=self.runtime_source,robot8=self.robot8)

    def validate_context(self, context):
        rows,sha = read_sessions(self.workspace)
        sid = context.get('session_id')
        if sid not in rows or self.root_session not in rows:
            raise ValueError('Session is absent from the actual native registry')
        row,root = rows[sid],rows[self.root_session]
        for key in _FIELDS:
            default = 0 if key == 'replan_attempt' else None
            if native_serialized_field(key,context.get(key,default)) != native_serialized_field(key,row.get(key,default)):
                raise ValueError('RPC Session differs from registered '+key)
        if row.get('status') != 'running':
            raise ValueError('Only the claimed running native Session may reset')
        core = {k:root.get(k,0 if k=='replan_attempt' else None) for k in _FIELDS}
        if self.root_core is not None and core != self.root_core:
            raise ValueError('Root Session contract changed after execution')
        if root.get('parent_session_id') is not None or root.get('replan_attempt',0) != 0 or root.get('task_description') != self.goal:
            raise ValueError('Original root goal or lineage differs')
        expected_benchmark = benchmark_metadata(root.get('benchmark') or {},self.run_id)
        if native_serialized_field('benchmark',root.get('benchmark')) != native_serialized_field('benchmark',expected_benchmark):
            raise ValueError('Official fixed-state benchmark identity must be removed')
        current = row
        chain = []
        while current['session_id'] != self.root_session:
            attempt = current.get('replan_attempt')
            parent_id = current.get('parent_session_id')
            if type(attempt) is not int or attempt not in (1,2) or parent_id not in rows:
                raise ValueError('Unregistered native recovery lineage')
            parent = rows[parent_id]
            metadata = (parent.get('result') or {}).get('metadata') or {}
            verification = metadata.get('verification') or {}
            applied = [a for a in verification.get('attempts',[]) if a.get('attempt_id') == verification.get('applied_attempt_id')]
            if (current['session_id'] != parent_id+'_replan_'+str(attempt) or
                    parent.get('replan_attempt',0)+1 != attempt or parent.get('status') != 'replanned' or
                    metadata.get('replanned_session_id') != current['session_id'] or
                    len(applied) != 1 or applied[0].get('mode') != 'apply' or applied[0].get('verdict') != 'replan'):
                raise ValueError('Child is not the native verifier registered replan')
            if parent_id not in self.used or not any(r['session_id']==parent_id and r.get('status')=='MATCHED_FROZEN_INPUT' for r in self.receipts):
                raise ValueError('Recovery parent was not executed by this target binding')
            chain.append(parent_id)
            if len(chain)>2:raise ValueError('Native replan limit exceeded')
            current=parent
        for item in (root,row):
            ex=item.get('execution') or {}
            if (item.get('target_ref') != 'target://libero_real_remote' or
                    item.get('verification_profile') != 'recovery' or ex.get('max_steps') != 400 or
                    ex.get('reset_policy') != 'session_runner' or ex.get('replan_every_steps') != 10):
                raise ValueError('Original target, recovery or per-Session400/H10 contract differs')
        for key in ('target_ref','skillruntime_ref','benchmark','execution','routing'):
            if row.get(key) != root.get(key):
                raise ValueError('Child changed inherited native '+key)
        if not str(row.get('task_description','')).strip():
            raise ValueError('Empty native task instruction')
        return row,sha,core

    def reset(self,runtime,context,original_reset):
        tick=time.perf_counter()
        row,registry_sha,core = self.validate_context(context)
        sid=row['session_id']
        if sid in self.used or len(self.used)>=3:
            raise ValueError('Duplicate reset or native recovery limit exceeded')
        if runtime.session_id != sid or runtime.env is not None:
            raise ValueError('Original native close and current Session configure/start are required')
        # Slot0 only satisfies the unchanged native table indexing. Its state
        # assignment is bypassed by the exact draw restorer, never scored as0.
        if {**runtime.config,**context.get('libero',{})}.get('init_state_id') != 0:
            raise ValueError('Internal unused official slot must remain0')
        self.used.add(sid)
        self.root_core=copy.deepcopy(core)
        record=self.last_receipt=dict(version=VERSION,session_id=sid,parent_session_id=row.get('parent_session_id'),
            replan_attempt=row.get('replan_attempt',0),status='RUNNING',registry_sha256=registry_sha,
            native_sampler={'role':self.input_role,'sample_id':self.base['sample_id'],
                'draw_seed':int(self.base['sample_id'].split('_')[-1]),
                'collection_manifest_sha256':self.base['collection_manifest_sha256']},
            instance_id=None,internal_ignored_official_slot=0,per_session_action_limit=400,
            maximum_chain_sessions=3,maximum_chain_actions=1200,same_episode_recovery=False)
        self.receipts.append(record)
        restorer=None
        try:
            restorer=self._restorer(sid)
            result=restorer.reset(runtime,context,original_reset)
            record.update(status=restorer.last_receipt['status'])
            return result
        except BaseException as error:
            record.update(status='FAILED',error=repr(error))
            raise
        finally:
            record['restore']=None if restorer is None else copy.deepcopy(restorer.last_receipt)
            record['seconds']=time.perf_counter()-tick
            record['cost_relationship']='seconds inclusive of nested restore.seconds; never add both'


def install(runtime_module, config):
    if config is None:return None
    binding=NativeSamplerAgentBinding(config,runtime_source=runtime_module.__file__,robot8=runtime_module._libero_state)
    klass=runtime_module.LiberoRealRuntime
    original=klass.reset
    def reset(runtime,context):return binding.reset(runtime,context,original)
    klass.reset=reset
    # Public records identify the extension, not an official init0 result.
    # The draw seed is only in evaluator receipts above, not model observations.
    for name in ('_libero_metadata','_episode_summary'):
        old=getattr(klass,name)
        def labelled(runtime,_old=old):
            result=_old(runtime)
            result['init_state_id']=None
            result['initialization_source']='native_sampler_extension'
            return result
        setattr(klass,name,labelled)
    return binding
