"""Summarize spans without adding overlapping/parent timings as total cost."""
import argparse
import collections
import json
from pathlib import Path


def interval_union(intervals):
    total=0
    edge=None
    for start,end in sorted(intervals):
        if end<start:raise ValueError('Negative span duration')
        if edge is None or start>=edge:
            total+=end-start
        elif end>edge:
            total+=end-edge
        edge=end if edge is None else max(edge,end)
    return total


def summarize(paths):
    begins={}
    ends={}
    native=[]
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row=json.loads(line)
            if row['event']=='native_observation':native.append(row)
            if row['event'] not in ('begin','end'):continue
            key=(row['process'],row['span_id'])
            table=begins if row['event']=='begin' else ends
            if key in table:raise ValueError('Duplicate span event')
            table[key]=row
    groups=collections.defaultdict(list)
    total=[]
    for key,end in ends.items():
        if key not in begins:continue
        start=begins[key]
        assert start['name']==end['name']
        pair=(start['mono_ns'],end['mono_ns'])
        groups[start['name']].append(pair)
        total.append(pair)
    result={'scope':'measured intervals only; setup/load and uninstrumented time require separate receipts',
        'clock_assumption':'all files from one Linux host sharing the monotonic clock',
        'unclosed_spans':len(begins.keys()-ends.keys()),
        'orphan_end_spans':len(ends.keys()-begins.keys()),
        'observed_envelope_seconds':(max(b for _,b in total)-min(a for a,_ in total))/1e9 if total else None,
        'observed_union_seconds':interval_union(total)/1e9,'categories':{}}
    for name,intervals in groups.items():
        result['categories'][name]={'calls':len(intervals),
            'service_seconds':sum(b-a for a,b in intervals)/1e9,
            'union_seconds':interval_union(intervals)/1e9}
    result['categories_are_additive']=False
    last={}
    for row in sorted(native,key=lambda v:v['wall_ns']):last[row['session_id']]=row
    result['native_terminal_by_session']=last
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('files',nargs='+');args=parser.parse_args()
    print(json.dumps(summarize(args.files),ensure_ascii=False,indent=2))
