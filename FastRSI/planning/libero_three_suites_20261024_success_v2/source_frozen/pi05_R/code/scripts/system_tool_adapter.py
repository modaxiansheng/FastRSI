"""Preserve native ToolRegistry validation feedback for model argument errors.

Transport rejects malformed, truncated, undeclared, non-object or forced-name
violations. Valid JSON argument objects are not rewritten: the native executor
retains its actual schema validation and error hint, permitting normal recovery.
"""
import copy
from contextvars import ContextVar
from fastrsi.tool_protocol import ProtocolError
from fastrsi.tool_protocol import parse_native_completion as native_parser
from fastrsi.tool_protocol import parse_qwen_completion as legacy_parser

_handoff=ContextVar('native_tool_argument_handoff',default=None)

def recent_handoff():
    return _handoff.get()

def parse_with_native_validation(parser,text,tools=None,choice='auto',**kwargs):
    _handoff.set(None)
    try:
        return parser(text,tools,choice,**kwargs)
    except ProtocolError as exc:
        if not str(exc).startswith('Tool arguments violate schema: '):
            raise
        # Names, envelopes, finite JSON, choice and truncation stay validated.
        # The original complete advertised schema is still used by the model
        # and the actual native ToolRegistry; only transport validation defers.
        transport=copy.deepcopy(tools)
        for tool in transport or []:
            tool['function']['parameters']={'type':'object'}
        result=parser(text,transport,choice,**kwargs)
        _handoff.set({'version':'native_tool_validation_handoff_v1','model_arguments_rewritten':False,
                      'original_transport_diagnostic':str(exc),
                      'execution_validation':'Original native ToolRegistry.cast_params/validate_params/execute unchanged'})
        return result

def parse_native_completion(text,tools=None,choice='auto',**kwargs):
    return parse_with_native_validation(native_parser,text,tools,choice,**kwargs)

def parse_qwen_completion(text,tools=None,choice='auto',**kwargs):
    return parse_with_native_validation(legacy_parser,text,tools,choice,**kwargs)
