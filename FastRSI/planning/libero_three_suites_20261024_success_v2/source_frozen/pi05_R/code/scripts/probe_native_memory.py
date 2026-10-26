"""One native /new rollover after a completed development batch; no robot episodes.

This probes the real memory path, not an extra reflection step per episode.
An unchanged decision is reported as unchanged, never as a learning benefit.
"""
import asyncio
import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path

from fastrsi.native_binding import bind_agent, finish_agent, hashes
from run_native_baseline import measured_provider, setup_workspace, plan_session

ROOT = Path(os.environ['FASTRSI_NATIVE_RUN'])
SOURCE = Path('/project/runs/native_baseline_v5_20261002/stream/dev2_t4_i23/accepted_experience')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


async def main():
    report = {'status': 'RUNNING', 'source': str(SOURCE),
              'scope': 'Native end-of-development /new consolidation and fresh conversation planning. No additional robot episode, speedup claim or forced per-round consolidation.'}
    started = time.time()
    try:
        assert json.loads((SOURCE/'RECEIPT.json').read_text())['chain_status'] == 'PASSED'
        folder = ROOT/'rollover'; folder.mkdir()
        workspace = folder/'workspace'; shutil.copytree(SOURCE, workspace)
        before = hashes(workspace)
        log = folder/'AGENT_EVENTS.jsonl'
        # Restore the frozen provider's 4096-token default for archival. The
        # 2048-token planning cap is not a native limit on a full memory update.
        provider = measured_provider(folder, 'native_memory', 4096)
        provider._client.timeout = 600.0
        agent = bind_agent(workspace, provider, log)
        key = 'cli:dev2_t4_i23'
        session = agent.sessions.get_or_create(key)
        save(folder/'SOURCE_CONVERSATION.json', session.messages)
        assert session.messages, 'Native saved conversation was not restored'
        try:
            # The frozen AgentLoop owns archive_unconsolidated, save_memory and clear.
            reply = await agent.process_direct('/new', session_key=key)
            assert reply == 'New session started.', reply
            assert (workspace/'memory/MEMORY.md').is_file(), 'No persistent native memory'
            assert (workspace/'memory/HISTORY.md').is_file(), 'No native history entry'
            assert not agent.sessions.get_or_create(key).messages
            save(folder/'RESULT.json', {'status': 'PASSED', 'trigger': '/new', 'reply': reply,
                'source_messages': len(json.loads((folder/'SOURCE_CONVERSATION.json').read_text())),
                'before': before, 'after': hashes(workspace)})
        finally:
            await finish_agent(agent, log)
            await provider._client.close()

        # A fresh conversation reads what the original /new handler actually wrote.
        folder = ROOT/'fresh_session'; folder.mkdir()
        goal = json.loads((SOURCE/'REQUEST.json').read_text())['goal']
        case = {'id': 'memory_probe_t4_i29', 'mode': 'development', 'task_id': 4,
                'init_state_id': 29, 'task_description': goal}
        w, spec, previous = setup_workspace(folder, case, workspace)
        log = folder/'AGENT_EVENTS.jsonl'
        provider = measured_provider(folder, 'agent', 2048)
        agent = bind_agent(w, provider, log)
        try:
            response, planned = await plan_session(agent, case, w, spec, previous, log)
            memory = (w/'memory/MEMORY.md').read_text().strip()
            requests = [json.loads(p.read_text()) for p in folder.glob('agent_call_*/REQUEST.json')]
            read = any(memory in str(m.get('content', '')) for r in requests for m in r.get('messages', []))
            assert read, 'Native memory absent from actual fresh-session model requests'
            save(folder/'RESULT.json', {'status': 'PASSED', 'response': response,
                'planned_session': planned, 'actual_memory_in_model_request': read,
                'memory_sha256': hashlib.sha256((w/'memory/MEMORY.md').read_bytes()).hexdigest(),
                'description_changed_from_goal': planned['task_description'] != goal,
                'decision_influence': 'NOT_ESTABLISHED_WITHOUT_COUNTERFACTUAL',
                'robot_dispatched': False})
            # Replay the actual Session-writing decision with only the native
            # long-term memory removed. Repeated controls expose nondeterminism.
            source_request = None
            for p in sorted(folder.glob('agent_call_*/RESPONSE.json')):
                output = json.loads(p.read_text())
                if any(t.get('name') in ('append_yaml_item', 'write_file', 'edit_file') and
                       str(t.get('arguments', {}).get('path', '')).endswith('SESSIONS.md')
                       for t in output.get('tool_calls', [])):
                    source_request = json.loads((p.parent/'REQUEST.json').read_text())
                    source_decision = [{'name': t['name'], 'arguments': t['arguments']} for t in output['tool_calls']]
                    break
            assert source_request is not None
            variant = json.loads(json.dumps(source_request))
            removed = 0
            for message in variant['messages']:
                content = message.get('content')
                if message.get('role') == 'system' and isinstance(content, str) and memory in content:
                    removed += content.count(memory)
                    message['content'] = content.replace(memory, '(No long-term memory in this ablation.)')
            assert removed == 1, 'Counterfactual must remove exactly one native memory block'
            replay = ROOT/'decision_counterfactual'; replay.mkdir()
            replay_provider = measured_provider(replay, 'decision_replay', 2048)
            signatures = []
            try:
                for label, request in [('with_memory_1', source_request), ('without_memory', variant), ('with_memory_2', source_request)]:
                    output = await replay_provider.chat(**request)
                    signature = [{'name': t.name, 'arguments': t.arguments} for t in output.tool_calls]
                    signatures.append(signature)
                    save(replay/(label+'.json'), {'request': request, 'decision': signature, 'text': output.content})
            finally:
                await replay_provider._client.close()
            consistent = signatures[0] == signatures[2] == source_decision
            changed = signatures[0] != signatures[1]
            influence = ('OBSERVED_IN_PAIRED_DECISION_REPLAY' if consistent and changed else
                         'NO_OBSERVED_DECISION_CHANGE' if consistent else 'INCONCLUSIVE_CONTROL_VARIATION')
            save(replay/'RESULT.json', {'status': 'COMPLETED', 'controls_match_actual_decision': consistent,
                'decision_changed_without_memory': changed, 'decision_influence': influence,
                'intervention': 'Only the native long-term memory text in the identical saved system message was removed.',
                'limitation': 'This is one planning decision; replay candidates are not executed and do not demonstrate robot quality or cost reduction.'})
        finally:
            await finish_agent(agent, log)
            await provider._client.close()
        report.update(status='COMPLETED', native_consolidation='PASSED', fresh_session_read='PASSED',
                      decision_influence=influence)
    except BaseException:
        report.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        report['seconds'] = time.time()-started
        save(ROOT/'RESULT.json', report)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
