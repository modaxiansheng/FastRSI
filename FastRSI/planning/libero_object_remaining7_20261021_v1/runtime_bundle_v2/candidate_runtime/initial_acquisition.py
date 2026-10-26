"""Respect frozen resident ownership: first acquisition is not an old-owner handoff."""
def acquire_boundary(client, prior_inferences, request_id):
    if prior_inferences == 0:
        status = client.status()
        active = status.get('active') or {}
        assert active.get('session_id') == client.session_id
        assert active.get('inferences') == 0 and active.get('owner') is None
        assert status.get('owner_connected') is False
        return dict(operation='FIRST_INFERENCE_OWNER_ACQUISITION_AFTER_SKILL', prior_inferences=0,
                    no_administrative_mutation=True, status_before_first_inference=status,
                    old_policy_actions_existed=False, rng_reset=False,
                    evidence='Begin receipt and zero inference ledger; no previous owner or action queue exists')
    receipt = client.handoff(request_id)
    assert receipt['receipt']['rng_reset'] is False
    assert receipt['receipt']['rng_before_hashes'] == receipt['receipt']['rng_after_hashes']
    return receipt
