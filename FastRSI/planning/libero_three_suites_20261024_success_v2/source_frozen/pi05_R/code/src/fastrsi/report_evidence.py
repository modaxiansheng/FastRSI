"""Lossless value view of current Session nodes and episode JSON; no summarization.

Complete chunks arrays use lossless columns, invariant values and string prefixes. All values are retained.
Canonical files and their raw-byte hashes remain authoritative.
"""
import copy, hashlib, json
from pathlib import Path
VERSION='lossless_report_evidence_v2'
REF='$fastrsi_chunks_ref'
def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def pack_rows(rows):
    if not rows or not all(isinstance(r,dict) and set(r)==set(rows[0]) for r in rows):
        return {'codec':'literal','value':copy.deepcopy(rows)}
    columns=list(rows[0]);constants={};variable=[];prefixes={}
    for k in columns:
        vals=[r[k] for r in rows]
        if all(canonical(v)==canonical(vals[0]) for v in vals):constants[k]=copy.deepcopy(vals[0])
        else:
            variable.append(k)
            if all(isinstance(v,str) for v in vals):
                import os
                prefix=os.path.commonprefix(vals)
                if len(prefix)>8:prefixes[k]=prefix
    table=[[r[k][len(prefixes[k]):] if k in prefixes else copy.deepcopy(r[k]) for k in variable] for r in rows]
    return {'codec':'columns_constants_prefixes_v1','columns':columns,'constants':constants,
            'variable_columns':variable,'string_prefixes':prefixes,'rows':table}
def unpack_rows(table):
    if table['codec']=='literal':return copy.deepcopy(table['value'])
    if table['codec']!='columns_constants_prefixes_v1':raise ValueError('Unknown table codec')
    columns=table['columns'];variable=table['variable_columns'];constants=table['constants'];prefixes=table['string_prefixes']
    if len(columns)!=len(set(columns)) or set(variable)&set(constants) or set(variable)|set(constants)!=set(columns):
        raise ValueError('Invalid table columns')
    rows=[]
    for row in table['rows']:
        if len(row)!=len(variable):raise ValueError('Invalid table width')
        values={**copy.deepcopy(constants),**{k:prefixes[k]+v if k in prefixes else copy.deepcopy(v) for k,v in zip(variable,row)}}
        rows.append({k:values[k] for k in columns})
    return rows
def encode(sources):
    shared={};occurrences=0
    def pack(value,key=None):
        nonlocal occurrences
        if isinstance(value,dict):
            if REF in value:raise ValueError('Reserved reference key in source')
            return {k:pack(v,k) for k,v in value.items()}
        if isinstance(value,list):
            if key=='chunks':
                # Do not use an outcome-dependent filter, drop failures, or cap the array.
                encoded=canonical(value);ref=hashlib.sha256(encoded.encode()).hexdigest()
                shared[ref]=pack_rows(value);occurrences+=1
                return {REF:ref}
            return [pack(v) for v in value]
        return value
    data=pack(copy.deepcopy(sources))
    view={'version':VERSION,'meaning':'Expand shared_chunks tables: each row maps to variable_columns; prepend string_prefixes to those string cells; merge constants; reorder by columns. This reconstructs every original chunk exactly, including all failures. Sources contain all other complete values. No values omitted.','sources':data,'shared_chunks':shared}
    if decode(view)!=sources:raise ValueError('Evidence roundtrip mismatch')
    return view,{'chunk_array_occurrences':occurrences,'unique_chunk_arrays':len(shared),'lossless_value_roundtrip':True}
def decode(view):
    if view['version']!=VERSION:raise ValueError('Unknown evidence version')
    shared={k:unpack_rows(v) for k,v in view['shared_chunks'].items()}
    for k,v in shared.items():
        if hashlib.sha256(canonical(v).encode()).hexdigest()!=k:raise ValueError('Shared evidence hash mismatch')
    def unpack(v):
        if isinstance(v,dict):
            if REF in v:
                if set(v)!={REF} or v[REF] not in shared:raise ValueError('Invalid reference')
                return copy.deepcopy(shared[v[REF]])
            return {k:unpack(x) for k,x in v.items()}
        if isinstance(v,list):return [unpack(x) for x in v]
        return v
    return unpack(view['sources'])
def create(workspace,selected,canonical_paths):
    w=Path(workspace);sources=[]
    for rel in canonical_paths:
        raw=(w/rel).read_bytes()
        sources.append({'path':rel,'raw_sha256':hashlib.sha256(raw).hexdigest(),
                        'selection':'complete current Session nodes' if rel=='SESSIONS.md' else 'complete episode',
                        'data':selected if rel=='SESSIONS.md' else json.loads(raw)})
    view,receipt=encode(sources)
    rel='reports/evidence/current-'+hashlib.sha256(canonical(sources).encode()).hexdigest()+'.json'
    p=w/rel;p.parent.mkdir(parents=True,exist_ok=True)
    # Compact JSON reduces formatting only; the decoder reconstructs all original values.
    body=json.dumps(view,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
    if p.exists() and p.read_text()!=body:raise ValueError('Existing evidence view differs')
    p.write_text(body)
    return rel,{**receipt,'view_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'view_bytes':len(body.encode()),'canonical_value_bytes':len(canonical(sources).encode()),'source_sha256':{s['path']:s['raw_sha256'] for s in sources}}
