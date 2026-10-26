"""Deduplicate complete batch results only against the next call's actual context.

No file cache: callers must perform the native read before asking for a reference.
No persistent state, model inference, or cross-Session memory is introduced.
"""
import json
import hashlib
import re
from pathlib import Path

EXPANSION = '\n\nCOMPILED DISCOVERY EXPANSION:'


def complete(result):
    if not isinstance(result, str):
        return False
    body, sep, footer = result.rpartition('\n\n')
    match = re.fullmatch(r'\(End of file — (\d+) lines total\)', footer)
    if not sep or not match:
        return False
    lines = body.splitlines()
    return len(lines) == int(match[1]) and all(
        line.startswith(str(i) + '| ') for i, line in enumerate(lines, 1))


class VisibleReads:
    def __init__(self, workspace, record=None, restored=None):
        self.workspace = Path(workspace).resolve()
        self.record = record
        self.visible = {}
        self.snapshots = {}
        self.restored = restored

    def prepare(self, messages):
        """Restore old read snapshots if native history pruning orphaned a reference.

        These are the exact past tool results, not a claim about the file now.
        The original Agent history is not mutated; only its outbound representation
        is made self-contained. Every actual new read still reads the file afresh.
        """
        self.observe(messages)
        prepared = list(messages)
        for index, message in enumerate(messages):
            if message.get('role') != 'tool' or not isinstance(message.get('content'), str):
                continue
            content = message['content']
            prefix = ''
            if EXPANSION in content:
                head, sep, content = content.partition(EXPANSION)
                heading, newline, content = content.partition('\n')
                prefix = head + sep + heading + newline
            try:
                payload = json.loads(content)
                changed = False
                for row in payload['fresh_reads']:
                    snapshot = self.snapshots.get(row.get('result'))
                    if snapshot is None:
                        continue
                    path, original, source = snapshot
                    if self.visible.get((path, original)) == source:
                        continue
                    row['result'] = original
                    changed = True
                    if self.restored:
                        self.restored(target_path=path, source_tool_call_id=source,
                                      original_chars=len(original))
                if changed:
                    prepared[index] = dict(message, content=prefix+json.dumps(payload, ensure_ascii=False))
            except (ValueError, KeyError, TypeError):
                # Native save_turn can also cut the containing JSON after a
                # complete reference. Restore known result strings even there;
                # keep the native truncation marker and remaining text intact.
                raw = message['content']
                for reference, (path, original, source) in self.snapshots.items():
                    needle = '"result": ' + json.dumps(reference, ensure_ascii=False)
                    if needle not in raw or self.visible.get((path, original)) == source:
                        continue
                    raw = raw.replace(needle, '"result": ' + json.dumps(original, ensure_ascii=False))
                    if self.restored:
                        self.restored(target_path=path, source_tool_call_id=source,
                                      original_chars=len(original), truncated_container=True)
                if raw != message['content']:
                    prepared[index] = dict(message, content=raw)
        self.observe(prepared)
        return prepared

    def canonical(self, path):
        p = Path(path)
        p = (self.workspace / p).resolve() if not p.is_absolute() else p.resolve()
        p.relative_to(self.workspace)
        return str(p)

    def observe(self, messages):
        # Replace, never merge: removed/truncated history cannot authorize reuse.
        self.visible = {}
        calls = {}
        for message in messages:
            if message.get('role') == 'assistant':
                for call in message.get('tool_calls') or []:
                    calls[call.get('id')] = call.get('function', {})
            if message.get('role') != 'tool':
                continue
            origin = calls.get(message.get('tool_call_id'), {})
            content = message.get('content')
            if not isinstance(content, str):
                continue
            try:
                if origin.get('name') == 'read_files':
                    payload = json.loads(content)
                elif origin.get('name') == 'read_file':
                    args = origin.get('arguments', '{}')
                    args = json.loads(args) if isinstance(args, str) else args
                    if self.canonical(args['path']) != str(self.workspace/'skills/benchmarking/SKILL.md'):
                        continue
                    # Exact production expansion follows its one-line heading.
                    payload = json.loads(content.split(EXPANSION, 1)[1].split('\n', 1)[1])
                else:
                    continue
                for row in payload['fresh_reads']:
                    if complete(row.get('result')):
                        key = self.canonical(row['path'])
                        self.visible[(key, row['result'])] = message['tool_call_id']
            except (ValueError, KeyError, TypeError, IndexError):
                continue

    def render(self, path, fresh_result):
        try:
            source = self.visible.get((self.canonical(path), fresh_result))
        except (ValueError, TypeError):
            source = None
        if source is None:
            return fresh_result
        text = ('Fresh read matches the complete numbered contents already visible in '
                f'tool result {source}, file {path}, SHA256 {hashlib.sha256(fresh_result.encode()).hexdigest()}. '
                'Use that original text; no content changed.')
        if len(text) >= len(fresh_result):
            return fresh_result
        if self.record:
            self.record(target_path=path, source_tool_call_id=source,
                        original_chars=len(fresh_result), returned_chars=len(text))
        self.snapshots[text] = (self.canonical(path), fresh_result, source)
        return text
