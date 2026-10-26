"""Common public tool approved for all experiment arms; not a learning rule or a Session planner."""
import asyncio
import copy
import fcntl
import json
import re
import time
import yaml
from PhyAgentOS.agent.tools.filesystem import _FsTool
from PhyAgentOS.runtime.state_io.atomic_file import atomic_write_text
from PhyAgentOS.runtime.state_io.markdown_yaml import _YAML_BLOCK_RE


class AppendYamlItemTool(_FsTool):
    @property
    def name(self):
        return 'append_yaml_item'

    @property
    def description(self):
        return ('Append one object to an existing top-level list in a Markdown YAML document. '
                'Supply only the NEW item; the tool preserves all existing records and other document content. '
                'It does not generate items, interpret tasks or alter existing records. Duplicate session_id values in SESSIONS.md are rejected; edit the existing draft instead. '
                'For readability, place a scalar-valued property last in item, after its nested mappings. JSON key order does not affect values. '
                'Return complete JSON: close every nested object, the item, arguments, and the outer call. '
                'Generic syntax example (not a task template): '+json.dumps({
                    'name':'append_yaml_item','arguments':{'path':'notes.md','list_key':'entries',
                    'item':{'details':{'value':1},'label':'new'}}},separators=(',',':')))

    @property
    def parameters(self):
        return {'type':'object','properties':{
            'path':{'type':'string'},'list_key':{'type':'string','pattern':'^[A-Za-z0-9_-]+$'},
            'item':{'type':'object'}},'required':['path','list_key','item'],'additionalProperties':False}

    async def execute(self,path,list_key,item):
        try:
            fp=self._resolve(path)
            if not isinstance(item,dict) or not re.fullmatch(r'[A-Za-z0-9_-]+',list_key):
                return 'Error: supply a top-level list key and one object'
            item=json.loads(json.dumps(item,allow_nan=False))
            lock=fp.with_suffix(fp.suffix+'.lock')
            with lock.open('a+') as handle:
                deadline=time.monotonic()+5
                while True:
                    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                    except BlockingIOError:
                        if time.monotonic()>deadline:return 'Error: timed out waiting for file lock'
                        await asyncio.sleep(.05)
                try:
                    text=fp.read_text();matches=list(_YAML_BLOCK_RE.finditer(text))
                    if len(matches)!=1:return 'Error: expected exactly one YAML block'
                    match=matches[0];data=yaml.safe_load(match['body'])
                    if not isinstance(data,dict) or not isinstance(data.get(list_key),list):
                        return 'Error: selected top-level list does not exist'
                    if fp.name=='SESSIONS.md' and list_key=='sessions':
                        sid=item.get('session_id')
                        if not isinstance(sid,str) or not sid.strip():return 'Error: a Session requires a nonempty session_id'
                        if any(isinstance(row,dict) and row.get('session_id')==sid for row in data[list_key]):
                            return 'Error: session_id already exists. No file change. Read and edit the existing draft; do not append it again.'
                    before=copy.deepcopy(data);data[list_key].append(item)
                    serialized=yaml.safe_dump(data,sort_keys=False,allow_unicode=True)
                    verified=yaml.safe_load(serialized)
                    assert verified[list_key][:-1]==before[list_key]
                    assert {k:v for k,v in verified.items() if k!=list_key}=={k:v for k,v in before.items() if k!=list_key}
                    # The body regex excludes the newline immediately before the fence.
                    updated=text[:match.start('body')]+serialized.rstrip('\n')+text[match.end('body'):]
                    atomic_write_text(fp,updated)
                    return json.dumps({'status':'appended','old_count':len(before[list_key]),'new_count':len(data[list_key])})
                finally:fcntl.flock(handle,fcntl.LOCK_UN)
        except (OSError,ValueError,TypeError,yaml.YAMLError) as exc:
            return 'Error: '+str(exc)
