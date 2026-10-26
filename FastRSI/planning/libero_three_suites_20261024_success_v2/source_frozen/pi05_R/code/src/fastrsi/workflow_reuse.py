"""Trace-derived fixed workflow reuse, inspired by Agent Workflow Memory.

Only procedural file names are reusable. No past observation/result is cached.
This is an engineering adaptation, not a reproduction of AWM's learned inducer.
"""
import hashlib
import json
import time
from pathlib import Path
from PhyAgentOS.agent.tools.base import Tool
from .experience_admission import validate_snapshot, assess, tree_hashes, SEAL

DISCOVERY = {'RUNTIME.md','TARGETS.md','SKILLRUNTIME.md','SESSIONS.md','LOG.md','REQUEST.json'}
SKILL_REL = 'skills/accepted-discovery/SKILL.md'


class ReadFilesTool(Tool):
    """Compose the same bounded native reads; child reads retain individual logs."""
    def __init__(self, execute, visible=None):
        self.execute_one = execute
        self.visible = visible
    @property
    def name(self): return 'read_files'
    @property
    def description(self):
        return 'Read 1 to 8 independent current workspace text files in one call. Each item contains a fresh native read_file result or, when enabled and the complete identical text is still in this conversation, a reference to that earlier tool result. This tool only reads; it never changes files, queues tasks or uses cached results. Use read_file with pagination if any returned file is truncated.'
    @property
    def parameters(self):
        return {'type':'object','properties':{'paths':{'type':'array','items':{'type':'string'},'minItems':1,'maxItems':8}},'required':['paths'],'additionalProperties':False}
    async def execute(self, paths):
        if not isinstance(paths,list) or not 1<=len(paths)<=8 or any(not isinstance(x,str) for x in paths):
            return 'Error: supply 1 to 8 file paths'
        if len(set(paths))!=len(paths): return 'Error: duplicate paths'
        rows=[]
        for path in paths:
            result = await self.execute_one('read_file',{'path':path})
            rows.append({'path':path,'result':self.visible.render(path,result) if self.visible else result})
        return json.dumps({'fresh_reads':rows},ensure_ascii=False)


def compile_workflow(source, trace_folder, output):
    """Extract an ordered discovery block from one admitted development trace."""
    tick=time.perf_counter();source=Path(source);trace=Path(trace_folder);out=Path(output)
    seal=validate_snapshot(source);row=json.loads((trace/'RESULT.json').read_text())
    if seal['source_case']!=row['case']['id']: raise ValueError('Trace/source case mismatch')
    if not assess(trace/'workspace',row)['eligible']: raise ValueError('Unaccepted trace feedback')
    if tree_hashes(trace/'workspace')!=seal['files']: raise ValueError('Trace not bound to admitted bytes')
    events=json.loads('['+','.join((trace/'AGENT_EVENTS.jsonl').read_text().splitlines())+']')
    paths=[];queued=False
    for e in events:
        if e.get('event')=='queue_accepted': queued=True;break
        if e.get('event')!='tool' or e.get('name')!='read_file':continue
        p=Path(e['params']['path']);name=p.name
        if name not in DISCOVERY:continue
        result=e.get('result','')
        if result.startswith('Error') and not (name=='LOG.md' and result.startswith('Error: File not found:')):continue
        if name not in paths:paths.append(name)
    if not queued or set(paths)!=DISCOVERY:raise ValueError('Incomplete observed discovery workflow')
    content='''---
name: accepted-discovery
description: Reuse an admitted LIBERO policy_loop discovery procedure using fresh file reads.
always: true
---
# Accepted discovery procedure
Applies only to LIBERO policy_loop queue preparation using the existing benchmarking skill.
First read skills/benchmarking/SKILL.md with read_file as usual. If its result includes a COMPILED DISCOVERY EXPANSION, all listed current files have already been read: use those results without redundant reads unless an error or truncation requires resolving. Otherwise gather the independent
current discovery files together with ONE read_files call, with paths = PATHS_PLACEHOLDER.
These are current inputs, not old answers. Inspect each returned file and any relevant feedback.
Continue the original skill's checks and append the NEW Session with native file tools.
Never copy previous task parameters, observations, object positions, success or verification conclusions.
If a required read fails or is truncated, use ordinary read_file/pagination to resolve it before queueing.
An absent initial LOG is observed absence, not evidence of success. If the runtime or skill does not
match this procedure, fall back to its original discovery process. Do not skip execution or verification.
'''.replace('PATHS_PLACEHOLDER',json.dumps(paths))
    out.mkdir(parents=True,exist_ok=False);p=out/'SKILL.md';p.write_text(content)
    receipt={'version':'fixed_discovery_v1','status':'PROCEDURAL_CANDIDATE','source_case':seal['source_case'],
        'source_seal_sha256':hashlib.sha256((source/SEAL).read_bytes()).hexdigest(),
        'trace_sha256':hashlib.sha256((trace/'AGENT_EVENTS.jsonl').read_bytes()).hexdigest(),
        'skill_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'paths':paths,
        'reference':'https://arxiv.org/abs/2409.07429',
        'extraction':'deterministic ordered read-set extraction plus engineering-authored grouping/fallback template; not learned AWM induction',
        'quality_scope':'Source observation admitted; procedural effectiveness still requires real-model validation',
        'compilation_seconds':time.perf_counter()-tick,'model_calls':0}
    (out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2));return receipt


def install_workflow(workspace, artifact):
    """Install the predeclared fixed method before Agent binding/evaluation freeze."""
    artifact=Path(artifact);receipt=json.loads((artifact/'RECEIPT.json').read_text());data=(artifact/'SKILL.md').read_bytes()
    if receipt['version']!='fixed_discovery_v1' or hashlib.sha256(data).hexdigest()!=receipt['skill_sha256']:
        raise ValueError('Workflow artifact changed')
    p=Path(workspace)/SKILL_REL;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(data)
    (p.parent/'WORKFLOW.json').write_text(json.dumps(receipt,indent=2))
    return receipt


def load_macro(workspace):
    p=Path(workspace)/SKILL_REL
    value=json.loads((p.parent/'WORKFLOW.json').read_text())
    if value['version']!='fixed_discovery_v1' or hashlib.sha256(p.read_bytes()).hexdigest()!=value['skill_sha256']:
        raise ValueError('Installed workflow changed')
    if set(value['paths'])!=DISCOVERY or len(value['paths'])!=len(DISCOVERY):
        raise ValueError('Invalid discovery expansion')
    return value
