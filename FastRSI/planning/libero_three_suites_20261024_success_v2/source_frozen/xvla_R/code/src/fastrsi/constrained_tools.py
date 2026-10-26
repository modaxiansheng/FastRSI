"""Constrain native tool JSON during generation; never repair generated output.

Ordinary assistant text and the model's decision to call a tool remain free.
After an emitted tool opener, LM Format Enforcer masks tokens until one complete
JSON object and closing tool envelope have been generated. Existing argument and
execution-contract validation still applies after generation.
"""
import time
from collections import OrderedDict
from lmformatenforcer import JsonSchemaParser, RegexParser
from lmformatenforcer.characterlevelparser import CharacterLevelParser, StringParser
from lmformatenforcer.tokenenforcer import TokenEnforcer

OPEN='<tool_call>'
CLOSE='</tool_call>'

def tool_parser(names):
    schema={'type':'object','properties':{'name':{'type':'string','enum':list(names)},
        'arguments':{'type':'object'}},
        'required':['name','arguments'],'additionalProperties':False}
    return ToolEnvelopeParser(JsonSchemaParser(schema))

class ToolEnvelopeParser(CharacterLevelParser):
    """Keep LMFE's free-string cache visible while requiring the closing tag."""
    def __init__(self,parser,suffix=False):
        super().__init__(getattr(parser,'_config',None));self.parser=parser;self.suffix=suffix
    def add_character(self,char):
        if not self.suffix and char=='<' and self.parser.can_end():
            return ToolEnvelopeParser(StringParser(CLOSE).add_character(char),True)
        return ToolEnvelopeParser(self.parser.add_character(char),self.suffix)
    def get_allowed_characters(self):
        chars=self.parser.get_allowed_characters()
        return chars+('<' if not self.suffix and self.parser.can_end() else '')
    def can_end(self):return self.suffix and self.parser.can_end()
    def shortcut_key(self):
        return self.parser.shortcut_key() if self.suffix or not self.parser.can_end() else None
    def cache_key(self):
        key=self.parser.cache_key()
        return ('tool-envelope',self.suffix,key) if key is not None else None

class ToolJSONLogitsProcessor:
    def __init__(self,tokenizer,tokenizer_data,prompt_length,names,variant='reference'):
        if variant not in ('reference','optimized'):raise ValueError('Unknown decoder variant')
        self.tokenizer=tokenizer;self.data=tokenizer_data;self.prompt_length=prompt_length
        self.names=names;self.active=None;self.processed=0
        self.variant=variant;self.indices=OrderedDict()
        self.stats={'variant':variant,'tool_blocks':0,'constrained_steps':0,'filter_wall_seconds':0.0,
            'token_transfer_and_device_wait_seconds':0.0,'decode_seconds':0.0,
            'parser_seconds':0.0,'mask_enqueue_seconds':0.0,'index_cache_hits':0}

    def __call__(self,input_ids,scores):
        start=time.perf_counter()
        try:
            assert input_ids.shape[0]==1,'One request per constrained decoder'
            tick=time.perf_counter()
            ids=(input_ids[0,self.prompt_length:] if self.variant=='optimized' else input_ids[0]).tolist()
            self.stats['token_transfer_and_device_wait_seconds']+=time.perf_counter()-tick
            text=None
            if self.variant=='reference' or self.active is None:
                tick=time.perf_counter()
                text=self.tokenizer.decode(ids if self.variant=='optimized' else ids[self.prompt_length:],skip_special_tokens=True,clean_up_tokenization_spaces=False)
                self.stats['decode_seconds']+=time.perf_counter()-tick
            tick=time.perf_counter()
            if self.active is None:
                pos=text.find(OPEN,self.processed)
                if pos<0:return scores
                parser=tool_parser(self.names)
                for char in text[pos+len(OPEN):]:
                    if char not in parser.get_allowed_characters():
                        raise ValueError('Invalid prefix inside an emitted tool opener')
                    parser=parser.add_character(char)
                self.active=TokenEnforcer(self.data,parser);self.stats['tool_blocks']+=1
            allowed=self.active.get_allowed_tokens(ids).allowed_tokens
            state=self.active.prefix_states[tuple(ids)].parser
            self.stats['parser_seconds']+=time.perf_counter()-tick
            if state.can_end():
                if text is None:
                    tick=time.perf_counter()
                    text=self.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    self.stats['decode_seconds']+=time.perf_counter()-tick
                self.active=None;self.processed=len(text)
                return scores
            import torch
            tick=time.perf_counter()
            mask=torch.full_like(scores,float('-inf'))
            if self.variant=='reference':
                mask[:,allowed]=scores[:,allowed]
            else:
                # Keep the list alive: object identity is safe only while retained.
                key=(id(allowed),str(scores.device));cached=self.indices.get(key)
                if cached is None:
                    index=torch.tensor(allowed,dtype=torch.long,device=scores.device)
                    self.indices[key]=(allowed,index)
                    if len(self.indices)>32:self.indices.popitem(last=False)
                else:
                    index=cached[1];self.indices.move_to_end(key);self.stats['index_cache_hits']+=1
                mask.index_copy_(1,index,scores.index_select(1,index))
            self.stats['mask_enqueue_seconds']+=time.perf_counter()-tick
            self.stats['constrained_steps']+=1
            return mask
        finally:self.stats['filter_wall_seconds']+=time.perf_counter()-start
