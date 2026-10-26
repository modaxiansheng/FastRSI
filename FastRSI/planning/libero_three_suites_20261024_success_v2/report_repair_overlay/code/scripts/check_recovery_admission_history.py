"""Read-only before/after admission comparison on all 40 bound executions."""
import hashlib
import importlib.util
import json
from pathlib import Path
from fastrsi.experience_admission import assess


def main():
    root=Path(__file__).resolve().parents[1]
    old_path=root/'runs/completion_learning_v1_20261004/code/src/fastrsi/experience_admission.py'
    spec=importlib.util.spec_from_file_location('old_admission',old_path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    inputs=root/'planning/budget_opportunity_v3_20261004/INPUTS.json'
    rows=[]
    for pair in json.loads(inputs.read_text()):
        for rel,digest in pair['sources'].items():
            if not rel.endswith('/stream/RESULT.json'):continue
            p=root/rel;data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==digest
            for row in json.loads(data)['cases']:
                w=p.parent/row['case']['id']/'workspace'
                before=old.assess(w,row);after=assess(w,row)
                keys=('eligible','reasons','attempts')
                same=all(before[k]==after[k] for k in keys)
                rows.append(dict(source=rel,source_sha256=digest,case=row['case']['id'],
                    old_eligible=before['eligible'],new_eligible=after['eligible'],unchanged=same))
    assert len(rows)==40
    out=root/'planning/recovery_dispatch_v1_20261004/HISTORY_REGRESSION.json'
    result=dict(status='PASSED' if all(r['unchanged'] for r in rows) else 'FAILED',
        scope='Read-only assessment; no historical receipt or experience rewrite',executions=len(rows),
        old_code_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
        new_code_sha256=hashlib.sha256((root/'src/fastrsi/experience_admission.py').read_bytes()).hexdigest(),rows=rows)
    out.write_text(json.dumps(result,indent=2));print(result['status'],len(rows))
    assert result['status']=='PASSED'


if __name__=='__main__':main()
