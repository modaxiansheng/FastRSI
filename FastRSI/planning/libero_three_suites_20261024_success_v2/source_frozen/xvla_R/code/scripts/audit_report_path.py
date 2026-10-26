"""Source-bound role of reporting and duplicate chunk representations."""
import hashlib,json,re,sys
from pathlib import Path
import yaml
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from fastrsi.report_evidence import encode,decode,canonical
sources={};rows=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,pair,arm,case in [('task_window_v1_20261003','pair0','N','net_t4_i5'),('task_window_v1_20261003','pair0','F','net_t4_i5'),('feedback_window_known_v1_20261003','pair0','N','net_t4_i31')]:
 w=R/'runs'/name/pair/arm/'stream'/case/'workspace';sp=w/'SESSIONS.md';ep=w/'artifacts/runtime'/case/'episode.json'
 doc=yaml.safe_load(re.search(r'```yaml\n(.*?)\n```',sp.read_text(),re.S)[1]);selected=[s for s in doc['sessions'] if s['session_id']==case or s['session_id'].startswith(case+'_replan')]
 records=[{'path':'SESSIONS.md','data':selected},{'path':str(ep.relative_to(w)),'data':json.loads(ep.read_text())}]
 view,receipt=encode(records);assert decode(view)==records
 compact=len(canonical(records).encode());view_bytes=len(canonical(view).encode())
 rows.append({'source':str(w.relative_to(R)),'original_value_bytes':compact,'view_value_bytes':view_bytes,'byte_reduction_fraction':1-view_bytes/compact,**receipt})
 for p in (sp,ep):sources[str(p.relative_to(R))]=sha(p)
for rel in ('third_party/PhyAgentOS-core/PhyAgentOS/skills/benchmarking/SKILL.md','third_party/PhyAgentOS-core/PhyAgentOS/agent/loop.py','src/fastrsi/terminal_results.py','src/fastrsi/native_binding.py','src/fastrsi/experience_admission.py','scripts/run_native_baseline.py'):
 sources[rel]=sha(R/rel)
report={'scope':'Observed controlled OS report path; not all native deployments','findings':['Verifier runs before consume_terminal; reporting does not invoke verifier.','Native benchmarking Results requests terminal state, manifest/summary and report references. Full current-node and full episode line coverage is a local integration safeguard, not an explicit native requirement.','Reporting is a native process_direct turn saved to task conversation; original token-driven consolidation hooks run normally. finish_agent drains and clears in-memory cache, does not forcibly call /new.','Whole-workspace admission copies reports only after runtime/semantic agreement. Inclusion in snapshot is not proof a report was read or learned in a later Session.','Repeated chunk arrays occur inside canonical Session, episode benchmark and final_status metadata. All are retained in raw files; optional view interns identical arrays once with invertible references.','Old full lifecycle times remain valid observed configured-entry results; extra read requirement must be disclosed and common repair cannot be called learning gain.'],'rows':rows,'sources':sources,'limitations':'Value-level roundtrip, not byte-format reconstruction of YAML comments. Byte hashes preserve original sources. Byte reduction is not a measured speedup.'}
out=R/'planning/report_evidence_v1_20261003';out.mkdir(parents=True,exist_ok=True);(out/'REPORT_PATH_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(rows,indent=2))
