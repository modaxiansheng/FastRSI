"""Shared input/output checks, independent of fixed or learned routing."""
import json
import re


def json_object_text(text):
    """Accept valid JSON or one complete JSON code fence; never repair values."""
    candidate=text.strip()
    if candidate.startswith('```'):
        match=re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```',candidate,re.DOTALL)
        if match is None:
            raise ValueError('Incomplete or mixed-content JSON fence')
        candidate=match.group(1).strip()
    obj=json.loads(candidate)
    if not isinstance(obj,dict):
        raise ValueError('Expected a JSON object')
    return candidate


def reuse_admissible(candidate_id, rules, evidence_version):
    """A model choice cannot create a missing or unvalidated recovery rule."""
    if not candidate_id:
        return False,'missing_candidate'
    rule=rules.get(candidate_id)
    if rule is None:
        return False,'unknown_candidate'
    if rule.get('validation_status')!='accepted':
        return False,'unvalidated_candidate'
    if rule.get('evidence_version')!=evidence_version or not evidence_version:
        return False,'stale_or_missing_evidence'
    if rule.get('applicable') is not True:
        return False,'applicability_unconfirmed'
    return True,'admissible'
