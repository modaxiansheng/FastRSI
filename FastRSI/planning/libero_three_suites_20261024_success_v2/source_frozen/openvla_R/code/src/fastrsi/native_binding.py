"""Bind the frozen AgentLoop to bounded tools and fixed-experience evaluation.

Development uses the native memory triggers. Evaluation freezes learning for the
entire workspace lifetime, including delayed callbacks, /new and system messages.
"""
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from .tool_origin import TOOL_ORIGIN
from PhyAgentOS.agent.loop import AgentLoop
from PhyAgentOS.agent.memory import MemoryStore, MemoryConsolidator
from PhyAgentOS.runtime.watchdog.result_writer import ResultWriter
from PhyAgentOS.bus.queue import MessageBus
from .file_append import AppendYamlItemTool
from .file_patch import PatchYamlItemTool
from .workflow_reuse import ReadFilesTool, load_macro
from .visible_reads import VisibleReads

FROZEN = {}
INSTALLED = False
CLOSED_EVALUATIONS = set()
FIXED_FILES = {'LESSONS.md','MEMORY_SPATIAL.md','TIMELINE.md','AGENTS.md',
               'ORCHESTRATOR.md','SOUL.md','USER.md','TOOLS.md','SKILLS.md','TASK.md','RUNTIME.md','EMBODIED.md',
               'TARGETS.md','SKILLRUNTIME.md','REQUEST.json'}
FIXED_DIRS = {'memory','skills','sessions','configs','routing','policies'}


def native_maintenance_enabled(case):
    """Optional native development permission; evaluation always remains read-only."""
    enabled=case.get('native_experience_maintenance',False)
    if type(enabled) is not bool:raise ValueError('Explicit boolean maintenance permission required')
    return enabled and case.get('mode')=='development'


def record(path, **event):
    with Path(path).open('a') as f:
        f.write(json.dumps({'time':time.time(), **event}, ensure_ascii=False, default=str)+'\n')


def hashes(workspace):
    root = Path(workspace)
    paths = [*(root/name for name in sorted(FIXED_FILES)),
             *(p for name in sorted(FIXED_DIRS) for p in root.glob(name+'/**/*'))]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


def frozen_for(path):
    path = Path(path).resolve()
    for root, log in FROZEN.items():
        if path == root or root in path.parents:
            return root, log
    return None


def install_freeze(workspace, log):
    """Process-lifetime guard; intentionally never removed while callbacks may exist."""
    global INSTALLED
    root = Path(workspace).resolve()
    FROZEN[root] = Path(log)
    if INSTALLED:
        return
    INSTALLED = True
    for name in ('write_long_term', 'append_history'):
        original = getattr(MemoryStore, name)
        def guarded(self, *a, _name=name, _original=original, **kw):
            match = frozen_for(self.memory_dir)
            if match:
                record(match[1], event='update_blocked', operation=_name)
                return None
            return _original(self, *a, **kw)
        setattr(MemoryStore, name, guarded)
    original_consolidate = MemoryStore.consolidate
    async def store_consolidate(self, *a, **kw):
        match = frozen_for(self.memory_dir)
        if match:
            record(match[1], event='update_blocked', operation='store_consolidate')
            return True
        return await original_consolidate(self, *a, **kw)
    MemoryStore.consolidate = store_consolidate
    for name in ('maybe_consolidate_by_tokens', 'archive_unconsolidated', 'consolidate_messages'):
        original = getattr(MemoryConsolidator, name)
        async def guarded(self, *a, _name=name, _original=original, **kw):
            match = frozen_for(self.store.memory_dir)
            if match:
                record(match[1], event='update_blocked', operation=_name)
                return True if _name != 'maybe_consolidate_by_tokens' else None
            return await _original(self, *a, **kw)
        setattr(MemoryConsolidator, name, guarded)
    original = ResultWriter.write_lesson
    def lesson(self, *a, **kw):
        match = frozen_for(self.workspace)
        if match:
            record(match[1], event='update_blocked', operation='write_lesson')
            return None
        return original(self, *a, **kw)
    ResultWriter.write_lesson = lesson

    def audit(event, args):
        paths = []
        if event == 'open':
            path, mode, flags = args
            if isinstance(path, (str, bytes, os.PathLike)) and (
                (mode and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):
                paths = [path]
        elif event in ('os.rename','os.link','os.symlink'):
            paths = list(args[:2])
        elif event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.truncate'):
            paths = [args[0]]
        for path in paths:
            if not isinstance(path, (str, bytes, os.PathLike)):
                continue
            path = Path(os.fsdecode(path)).resolve()
            if event == 'os.mkdir' and path.is_dir():
                continue  # ensure_dir(..., exist_ok=True) does not change experience
            match = frozen_for(path)
            if not match:
                continue
            rel = path.relative_to(match[0])
            if str(rel) in FIXED_FILES or (rel.parts and rel.parts[0] in FIXED_DIRS):
                record(match[1], event='filesystem_update_blocked', operation=event, target_path=str(path))
                raise PermissionError('Fixed evaluation experience is read-only: '+str(rel))
    sys.addaudithook(audit)


def bind_agent(workspace, provider, log, *, evaluation=False, verifier=None, max_iterations=10, read_files=False, workflow_macro=False, visible_reads=False, component_updates=False):
    if type(component_updates) is not bool:
        raise ValueError('Boolean component update permission required')
    workspace = Path(workspace)
    if workspace.resolve() in CLOSED_EVALUATIONS or (workspace/'.FASTRSI_EVALUATION_CLOSED.json').exists():
        raise RuntimeError('Finished evaluation workspace is evidence only; create the next task from the fixed development snapshot')
    (workspace/'memory').mkdir(exist_ok=True)
    (workspace/'skills').mkdir(exist_ok=True)
    (workspace/'sessions').mkdir(exist_ok=True)
    agent = AgentLoop(bus=MessageBus(), provider=provider, workspace=workspace,
        model=os.environ.get('FASTRSI_AGENT_MODEL','qwen25vl7b'), max_iterations=max_iterations, context_window_tokens=65536,
        restrict_to_workspace=True, runtime_workspace=workspace, runtime_enabled=True,
        session_verifier=verifier)
    agent._fastrsi_component_updates = component_updates and not evaluation
    agent._fastrsi_component_lock = False
    native_counter = agent.memory_consolidator.estimate_session_prompt_tokens
    def counted(session):
        tick = time.perf_counter()
        count, source = native_counter(session)
        record(log, event='native_token_count', session_key=session.key, tokens=count,
               source=source, seconds=time.perf_counter()-tick)
        if count <= 0:
            raise RuntimeError('Native token counting unavailable; cannot certify memory trigger execution')
        return count, source
    agent.memory_consolidator.estimate_session_prompt_tokens = counted
    agent.tools.register(AppendYamlItemTool(workspace=workspace, allowed_dir=workspace))
    agent.tools.register(PatchYamlItemTool(workspace=workspace, allowed_dir=workspace))
    allowed = {'read_file','write_file','edit_file','patch_yaml_item','append_yaml_item','list_dir','verify_session'}
    for name in list(agent.tools.tool_names):
        if name not in allowed:
            agent.tools.unregister(name)
    original_execute = agent.tools.execute
    macro = load_macro(workspace) if workflow_macro else None
    if macro and not read_files:raise ValueError('Compiled workflow requires common read_files tool')
    visible = None
    if visible_reads:
        if not read_files: raise ValueError("Visible reads require read_files")
        visible = VisibleReads(workspace, lambda **kw: record(log,event="visible_read_reused",**kw),
                               lambda **kw: record(log,event="visible_read_restored",**kw))
        original_chat = provider.chat
        async def chat(*args, **kwargs):
            if "messages" in kwargs:
                kwargs["messages"] = visible.prepare(kwargs["messages"])
            elif args:
                args = (visible.prepare(args[0]), *args[1:])
            return await original_chat(*args, **kwargs)
        provider.chat = chat
    workflow_used = False
    async def execute(name, params):
        nonlocal workflow_used
        started = time.perf_counter()
        if name in ('write_file','edit_file','patch_yaml_item','append_yaml_item'):
            p = Path(params.get('path','')); p = (workspace/p).resolve() if not p.is_absolute() else p.resolve()
            try:
                rel = p.relative_to(workspace.resolve())
            except ValueError:
                rel = Path('../outside')
            allowed_write = rel == Path('SESSIONS.md') or (rel.parts and rel.parts[0]=='reports') or (not evaluation and (str(rel) in {'LESSONS.md','MEMORY_SPATIAL.md','TIMELINE.md'} or (rel.parts and rel.parts[0] in ('memory','skills'))))
            from .native_component_updates_v1 import component_relative_path
            component = component_relative_path(workspace, params.get('path', ''))
            if component is not None:
                allowed_write = agent._fastrsi_component_updates and not agent._fastrsi_component_lock
            if (workspace/'skills/capability-interface').exists():
                from .capability_interface import protected_path
                if protected_path(rel):allowed_write=False
            if not allowed_write:
                result = 'Error: file write outside the bounded experiment contract'
                record(log,event='tool_blocked',name=name,params=params,result=result)
                return result
        result = await original_execute(name,params)
        record(log,event='tool',origin=TOOL_ORIGIN.get(),name=name,params=params,result=result,seconds=time.perf_counter()-started)
        if macro and not workflow_used and name=='read_file' and not result.startswith('Error'):
            raw=Path(params.get('path',''))
            fp=(workspace/raw).resolve() if not raw.is_absolute() else raw.resolve()
            if fp==(workspace/'skills/benchmarking/SKILL.md').resolve():
                workflow_used=True
                # Execute the frozen read-only procedure, never generate a Session.
                expansion=await agent.tools.execute('read_files',{'paths':macro['paths']})
                record(log,event='workflow_expanded',source_case=macro['source_case'],skill_sha256=macro['skill_sha256'],paths=macro['paths'],model_selected_batch=False)
                result+='\n\nCOMPILED DISCOVERY EXPANSION: These current files were just read by the fixed read-only procedure. Use their contents now; repeat reads only to resolve errors or truncation.\n'+expansion
        return result
    agent.tools.execute = execute
    if read_files:
        # Both comparison arms receive this identical read-only composition tool.
        async def child_read(name, params):
            return await agent.tools.execute(name, params)
        agent.tools.register(ReadFilesTool(child_read, visible))
    if evaluation:
        install_freeze(workspace, log)
        # Keep native task-local conversation in memory; never load previous eval files.
        from PhyAgentOS.session.manager import Session
        def closed():
            return workspace.resolve() in CLOSED_EVALUATIONS or (workspace/'.FASTRSI_EVALUATION_CLOSED.json').exists()
        def get(key):
            if closed():
                raise RuntimeError('Evaluation has ended; its task context cannot be reused')
            if key not in agent.sessions._cache:
                agent.sessions._cache[key] = Session(key=key)
            return agent.sessions._cache[key]
        def save(session):
            if closed():
                record(log,event='task_context_discarded_after_close',key=session.key)
                return
            agent.sessions._cache[session.key] = session
            record(log,event='task_context_retained',key=session.key,messages=len(session.messages))
        agent.sessions.get_or_create = get
        agent.sessions.save = save
    record(log,event='agent_bound',agent_class=type(agent).__module__+'.'+type(agent).__name__,
           evaluation=evaluation,tools=agent.tools.tool_names,context_window_tokens=agent.context_window_tokens,
           max_iterations=agent.max_iterations,experience=hashes(workspace))
    return agent


async def finish_agent(agent, log):
    """Drain actual tasks before discarding task-local context; guard remains active."""
    agent.stop()
    tasks = [t for values in agent._active_tasks.values() for t in values if not t.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    await agent.close_mcp()
    agent.sessions._cache.clear()
    workspace = Path(agent.workspace).resolve()
    if workspace in FROZEN:
        CLOSED_EVALUATIONS.add(workspace)
        (workspace/'.FASTRSI_EVALUATION_CLOSED.json').write_text(json.dumps({'closed_at':time.time(),'scope':'evaluation evidence only; never inherit as development experience'}))
    record(log,event='agent_closed',active_tasks_remaining=sum(not t.done() for t in tasks))
