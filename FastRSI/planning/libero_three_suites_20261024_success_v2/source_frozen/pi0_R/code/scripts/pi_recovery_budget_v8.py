"""Registered model-wide recovery budgets, without task/outcome special cases."""


def install(verifier_class, model, arm):
    if model not in ('pi0', 'pi05') or arm not in ('P', 'F', 'R'):
        raise ValueError('Unregistered model/arm')
    if model != 'pi0' or arm != 'R':
        return {'changed': False, 'reason': 'native verifier budget retained'}
    original = verifier_class.__init__
    def construct(self, *args, **kwargs):
        kwargs['max_replans'] = 0
        return original(self, *args, **kwargs)
    verifier_class.__init__ = construct
    return {'changed': True, 'max_replans': 0,
            'rule': 'Every PI0 R case: one native physical attempt; preserve original semantic verdict and canonical failure. No task-ID special case.'}
