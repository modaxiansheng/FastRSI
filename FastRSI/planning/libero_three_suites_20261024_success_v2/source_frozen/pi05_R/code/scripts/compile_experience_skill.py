"""Create a versioned candidate only from real, replay-verified source success."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.experience_compiler import compile_success,validate_compiled
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--stream',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 report=a.stream/'RESULT.json';x=json.loads(report.read_text());assert len(x['cases'])==1;row=x['cases'][0]
 trace=a.stream/'stage_trace'/row['case']['id']/'TRACE.jsonl';frames=[json.loads(s) for s in trace.read_text().splitlines()]
 skill=compile_success(row,frames,json.loads(a.replay.read_text()),provenance=dict(result_sha256=sha(report),trace_sha256=sha(trace),replay_sha256=sha(a.replay)))
 validate_compiled(skill);a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(skill,f,indent=2,allow_nan=False)
 print(json.dumps(dict(status='CANDIDATE_COMPILED',steps=len(skill['waypoints']),sha256=sha(a.output),transfer_validated=False)))
if __name__=='__main__':main()
