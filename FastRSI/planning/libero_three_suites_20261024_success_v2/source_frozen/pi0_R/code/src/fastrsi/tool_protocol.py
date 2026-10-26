"""Lossless, explicit Qwen tool wire adapter; no Agent implementation here."""
import copy
import json
import re
import uuid


class ProtocolError(ValueError):
    pass


def file_edit_wire_guidance(tools):
    """Clarify the existing edit API without changing its accepted arguments."""
    result = copy.deepcopy(tools)
    for tool in result or []:
        function = tool.get('function', {})
        if function.get('name') != 'edit_file':
            continue
        function['description'] = function.get('description', '') + (
            ' Use the shortest unique old_text fragment that anchors the edit. '
            'Do not copy the full existing file to append content. '
            'To insert a YAML list item before a Markdown closing fence, '
            'old_text can be exactly "\\n```\\n" and new_text contains only the new item followed by that fence. '
            'Keep all previous file contents outside the chosen fragment unchanged.')
    return result


def strict_json(text):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ProtocolError('Duplicate JSON key: ' + k)
            out[k] = v
        return out
    def constant(value):
        raise ProtocolError('Nonfinite JSON: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def prepare_messages(body):
    """Preserve each role and ordering; encode tool envelopes for the VL template."""
    tools = body.get('tools') or []
    names = [t['function']['name'] for t in tools]
    if len(set(names)) != len(names):
        raise ProtocolError('Duplicate tool names')
    choice = body.get('tool_choice', 'auto')
    if choice not in ('auto', 'none', 'required') and not isinstance(choice, dict):
        raise ProtocolError('Unsupported tool_choice')
    if choice == 'required' and not tools:
        raise ProtocolError('Required tools are missing')
    if isinstance(choice, dict) and choice.get('function', {}).get('name') not in names:
        raise ProtocolError('Requested tool is not declared')
    result = []
    pending = set()
    seen = set()
    tool_preamble = None
    if tools:
        descriptions = []
        for tool in tools:
            f = tool['function']
            descriptions.append('### '+f['name']+'\n\n'+f['name']+': '+f.get('description','')+
                ' Parameters: '+json.dumps(f.get('parameters',{}),ensure_ascii=False)+
                ' Format the arguments as a JSON object.')
        # Wire protocol follows Qwen-Agent QwenFnCallPrompt (Apache-2.0),
        # pinned source and license in planning/vendor_references. This is
        # message formatting only; the native OS AgentLoop remains unchanged.
        tool_preamble = ('# Tools\n\nYou may call one or more functions to assist with the user query.\n\n'+
            '\n\n'.join(descriptions)+'\n\n'+
            '## When you need to call a tool, insert the following command in your reply:\n\n'+
            '✿FUNCTION✿: The tool to use, should be one of ['+','.join(names)+']\n'+
            '✿ARGS✿: The input of the tool as a JSON object\n'+
            '✿RESULT✿: Tool results\n✿RETURN✿: Reply based on tool results.\n'+
            'Tool choice: '+json.dumps(choice)+'. Do not fabricate tool results.')
    for original in body.get('messages') or []:
        m = copy.deepcopy(original)
        role = m.get('role')
        if role not in ('system', 'user', 'assistant', 'tool'):
            raise ProtocolError('Unsupported role')
        calls = m.get('tool_calls') or []
        if role == 'tool':
            ident = m.get('tool_call_id')
            if ident not in pending:
                raise ProtocolError('Orphan or duplicate tool response')
            pending.remove(ident)
            # Official Qwen wire format appends results to the assistant turn.
            # Keep correlation metadata and complete content in the result data.
            if not result or result[-1]['role'] != 'assistant' or not isinstance(result[-1]['content'],str):
                raise ProtocolError('Tool result has no assistant prefix')
            content = result[-1]['content']
            if content.endswith('\n✿RETURN✿: '):
                content = content[:-len('\n✿RETURN✿: ')]
            result[-1]['content'] = content+'\n✿RESULT✿: '+json.dumps(m,ensure_ascii=False)+'\n✿RETURN✿: '
            continue
        if pending:
            raise ProtocolError('Missing tool response before next message')
        if calls and role != 'assistant':
            raise ProtocolError('Tool calls must be assistant messages')
        content = m.get('content') or ''
        if calls:
            if not isinstance(content, str):
                raise ProtocolError('Multimodal assistant tool message unsupported')
            for call in calls:
                ident = call['id']
                if ident in seen:
                    raise ProtocolError('Duplicate tool id')
                seen.add(ident); pending.add(ident)
                args = call['function']['arguments']
                args = strict_json(args) if isinstance(args, str) else args
                content += '\n✿FUNCTION✿: '+call['function']['name']+'\n✿ARGS✿: '+json.dumps(args,ensure_ascii=False)
        result.append({'role': role, 'content': content})
    if pending:
        raise ProtocolError('Unanswered tool calls')
    if tool_preamble:
        # The frozen VL template has no native tools slot. Put the tool contract
        # in the SAME system message after the OS context, not in a competing
        # earlier system message. All original content and ordering is retained.
        if result and result[0]['role']=='system':
            content=result[0]['content']
            if isinstance(content,str):result[0]['content']=content+'\n\n'+tool_preamble
            elif isinstance(content,list):content.append({'type':'text','text':tool_preamble})
            else:raise ProtocolError('Unsupported system content')
        else:
            result.insert(0,{'role':'system','content':tool_preamble})
    if not result:
        raise ProtocolError('Empty messages')
    if result[-1]['role']=='assistant' and isinstance(result[-1]['content'],str) and result[-1]['content'].endswith('✿RETURN✿: '):
        result[-1]['content']=result[-1]['content'][:-2]
    return result


def parse_completion(text, tools=None, choice='auto', *, truncated=False, max_calls=4):
    tools = tools or []
    declared = {t['function']['name']: t['function'] for t in tools}
    # Some VL generations end with EOS after a complete JSON call. Accept that
    # explicit wire dialect only at a normal generation end, with a fully parsed
    # object and no trailing prose. Never complete truncated JSON or infer args.
    if not truncated and text.count('<tool_call>') == 1 and '</tool_call>' not in text:
        _, tail = text.split('<tool_call>', 1)
        strict_json(tail.strip())  # full-document decode rejects partial/trailing text
        text = text.rstrip() + '</tool_call>'
    chunks = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', text, flags=re.S)
    if ('<tool_call' in text or '</tool_call' in text) and not chunks:
        raise ProtocolError('Incomplete tool envelope')
    residue = re.sub(r'<tool_call>\s*.*?\s*</tool_call>', '', text, flags=re.S).strip()
    if '<tool_call' in residue or '</tool_call' in residue:
        raise ProtocolError('Malformed tool envelope')
    if truncated and chunks:
        raise ProtocolError('Truncated tool generation cannot execute')
    if len(chunks) > max_calls:
        raise ProtocolError('Too many simultaneous tool calls')
    if chunks and choice == 'none':
        raise ProtocolError('Tool call forbidden by tool_choice')
    if not chunks and (choice == 'required' or isinstance(choice, dict)):
        raise ProtocolError('Required tool call missing')
    calls = []
    for chunk in chunks:
        value = strict_json(chunk)
        if not isinstance(value, dict) or not {'name','arguments'} <= set(value) or set(value)-{'name','arguments','id'}:
            raise ProtocolError('Tool call requires name, arguments and optional transport id only')
        if 'id' in value and (not isinstance(value['id'],str) or not 1 <= len(value['id']) <= 128):
            raise ProtocolError('Invalid optional transport id')
        name, args = value['name'], value['arguments']
        if name not in declared or not isinstance(args, dict):
            raise ProtocolError('Unknown tool or non-object arguments')
        if isinstance(choice, dict) and name != choice['function']['name']:
            raise ProtocolError('Wrong forced tool')
        import jsonschema
        try:
            jsonschema.validate(args, declared[name].get('parameters', {}))
        except jsonschema.ValidationError as exc:
            raise ProtocolError('Tool arguments violate schema: ' + exc.message) from exc
        # Provider owns call identity; generated optional ids are retained in RAW
        # evidence but never reused for dispatch or result correlation.
        calls.append({'id': 'call_' + uuid.uuid4().hex, 'type': 'function',
                      'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}})
    message = {'role': 'assistant', 'content': residue or None}
    if calls:
        message['tool_calls'] = calls
    return message, ('tool_calls' if calls else 'length' if truncated else 'stop')


def parse_qwen_completion(text, tools=None, choice='auto', *, truncated=False):
    """Strict model-native markers; no extraction from hypothetical code blocks."""
    text = text.lstrip(': ')  # assistant continuation after the RETURN marker
    # Generation stops at these markers, before any invented tool result.
    for marker in ('✿RESULT✿','✿RETURN✿'):
        if marker in text:
            before, after = text.split(marker,1)
            if after.strip():
                raise ProtocolError('Generated content after tool-result stop marker')
            text=before
    if '✿FUNCTION✿' not in text:
        if '✿ARGS✿' in text:
            raise ProtocolError('Arguments without function')
        return parse_completion(text,tools,choice,truncated=truncated)
    if truncated:
        raise ProtocolError('Truncated function generation')
    if '```' in text:
        raise ProtocolError('Fenced function example is not an executable call')
    prefix,*parts=text.split('✿FUNCTION✿:')
    if len(parts)==0 or '✿' in prefix:
        raise ProtocolError('Malformed function marker')
    wire=prefix
    for part in parts:
        if part.count('✿ARGS✿:')!=1:
            raise ProtocolError('Missing or duplicate arguments marker')
        name,args=part.split('✿ARGS✿:',1)
        value={'name':name.strip(),'arguments':strict_json(args.strip())}
        wire += '<tool_call>'+json.dumps(value,ensure_ascii=False)+'</tool_call>'
    return parse_completion(wire,tools,choice)


def prepare_native_messages(body):
    """Validate correlation, then retain native roles for the checkpoint template."""
    prepare_messages(body)  # common validation, independent deep copy
    messages=copy.deepcopy(body['messages'])
    if any(m['role']=='system' for m in messages[1:]):
        raise ProtocolError('Checkpoint template supports only the initial system message')
    expected=[]
    for m in messages:
        if m['role']=='assistant':
            expected=[c['id'] for c in m.get('tool_calls',[])]
            m['content']=m.get('content') or ''
        elif m['role']=='tool':
            if not expected or m['tool_call_id']!=expected.pop(0):
                raise ProtocolError('Native template requires ordered tool responses')
    return messages


def parse_native_completion(text, tools=None, choice='auto', *, truncated=False):
    if '<tool_call>' in text and '</tool_call>' not in text:
        raise ProtocolError('Native checkpoint call requires a closed envelope')
    message,finish=parse_completion(text,tools,choice,truncated=truncated,max_calls=8)
    if message.get('tool_calls') and '```' in (message.get('content') or ''):
        raise ProtocolError('Fenced tool example is not executable')
    return message,finish
