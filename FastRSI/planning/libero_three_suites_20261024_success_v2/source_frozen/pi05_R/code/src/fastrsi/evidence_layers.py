"""Report protocol completion separately from semantic correctness.

This is post-run audit metadata. It never changes native feedback or supplies
benchmark labels to a decision model. Aligned observations are not skills.
"""
def evidence_layers(native_success, outcomes, chain_status):
    codes=[o.get('code') for o in outcomes]
    verdicts=[o.get('verdict') for o in outcomes]
    last=verdicts[-1] if verdicts else None
    protocol_ok=bool(outcomes) and all(o.get('ok') is True and o.get('code')=='verified' for o in outcomes)
    agreement=None
    if type(native_success) is bool and last in ('success','failure'):
        agreement=native_success==(last=='success')
    if chain_status!='PASSED' or not protocol_ok:
        label='quarantine_incomplete_chain'
    elif agreement is False:
        label='quarantine_native_semantic_conflict'
    elif agreement is None:
        label='quarantine_unresolved_label'
    else:
        label='aligned_observation_only_not_validated_skill'
    return {'native_success':native_success,'chain_status':chain_status,
        'verification_protocol_completed':protocol_ok,'verification_codes':codes,
        'semantic_verdicts':verdicts,'semantic_session_statuses':[o.get('status') for o in outcomes],
        'native_vs_final_semantic_agreement':agreement,'learning_label_audit':label,
        'usage':'post-run audit only; preserve native behavior and raw feedback'}
