"""Transport ownership only: canonical scientific identities never change."""
MODES = ('ordinary', 'subgoal')


def session_id(identity, mode):
    if not isinstance(identity, str) or not identity.isalnum() or mode not in MODES:
        raise ValueError('invalid canonical identity or branch')
    return identity + '-' + mode + '-fourth_ordinary'


def resolve_session(value, rows):
    # Exact registration, not permissive splitting or prefix matching.
    matches = [(identity, mode) for identity in rows for mode in MODES
               if value == session_id(identity, mode)]
    if len(matches) != 1:
        raise ValueError('unregistered branch owner')
    identity, mode = matches[0]
    return identity, 'ordinary'  # One native sample for BOTH experimental branches.
