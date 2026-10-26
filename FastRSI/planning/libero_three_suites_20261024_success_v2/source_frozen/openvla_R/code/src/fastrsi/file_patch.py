"""Generic atomic update of one uniquely identified YAML-list object; no planning."""
import asyncio,copy,fcntl,json,re,time,yaml
from PhyAgentOS.agent.tools.filesystem import _FsTool
from PhyAgentOS.runtime.state_io.atomic_file import atomic_write_text
from PhyAgentOS.runtime.state_io.markdown_yaml import _YAML_BLOCK_RE

class PatchYamlItemTool(_FsTool):
    @property
    def name(self):return 'patch_yaml_item'
    @property
    def description(self):
        return ('Update selected fields in exactly one existing object of a Markdown YAML list, matched by an identity key and value. '
                'Nested mappings merge recursively; scalar/list values replace. No creation, no identity change, no task interpretation. '
                'For an existing Session draft use path=SESSIONS.md, list_key=sessions, match_key=session_id, match_value=its ID, '
                'and patch containing only the fields you intend to correct, such as benchmark, execution or timeouts. '
                'Other records and unrelated fields are preserved. This does not dispatch or approve a Session.')
    @property
    def parameters(self):
        return {'type':'object','properties':{'path':{'type':'string'},'list_key':{'type':'string'},'match_key':{'type':'string'},'match_value':{'type':'string'},'patch':{'type':'object'}},
                'required':['path','list_key','match_key','match_value','patch'],'additionalProperties':False}
    async def execute(self,path,list_key,match_key,match_value,patch):
        try:
            fp=self._resolve(path)
            if not all(re.fullmatch(r'[A-Za-z0-9_-]+',k) for k in [list_key,match_key]) or not isinstance(patch,dict):return 'Error: valid keys and an object patch required'
            patch=json.loads(json.dumps(patch,allow_nan=False))
            if match_key in patch and patch[match_key]!=match_value:return 'Error: identity changes are forbidden'
            with fp.with_suffix(fp.suffix+'.lock').open('a+') as handle:
                until=time.monotonic()+5
                while True:
                    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                    except BlockingIOError:
                        if time.monotonic()>until:return 'Error: file lock timeout'
                        await asyncio.sleep(.05)
                try:
                    text=fp.read_text();matches=list(_YAML_BLOCK_RE.finditer(text))
                    if len(matches)!=1:return 'Error: expected one YAML block'
                    m=matches[0];data=yaml.safe_load(m['body'])
                    if not isinstance(data,dict) or not isinstance(data.get(list_key),list):return 'Error: list not found'
                    indices=[i for i,v in enumerate(data[list_key]) if isinstance(v,dict) and v.get(match_key)==match_value]
                    if len(indices)!=1:return 'Error: identity must match exactly one existing item; no file change'
                    before=copy.deepcopy(data);i=indices[0]
                    def merge(dst,src):
                        for k,v in src.items():
                            if isinstance(v,dict) and isinstance(dst.get(k),dict):merge(dst[k],v)
                            else:dst[k]=copy.deepcopy(v)
                    merge(data[list_key][i],patch)
                    encoded=yaml.safe_dump(data,sort_keys=False,allow_unicode=True);checked=yaml.safe_load(encoded)
                    assert checked[list_key][i][match_key]==match_value
                    assert checked[list_key][:i]+checked[list_key][i+1:]==before[list_key][:i]+before[list_key][i+1:]
                    assert {k:v for k,v in checked.items() if k!=list_key}=={k:v for k,v in before.items() if k!=list_key}
                    atomic_write_text(fp,text[:m.start('body')]+encoded.rstrip('\n')+text[m.end('body'):])
                    return json.dumps({'status':'patched','match_key':match_key,'match_value':match_value,'patched_fields':list(patch)})
                finally:fcntl.flock(handle,fcntl.LOCK_UN)
        except (OSError,ValueError,TypeError,yaml.YAMLError) as exc:return 'Error: '+str(exc)
