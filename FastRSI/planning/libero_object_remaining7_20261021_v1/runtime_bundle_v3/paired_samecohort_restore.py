"""Exact actual baseline-end pickle restore; no actor, sampler or model changes.

Derived from the accepted paired_restore_v2 transport owner. Checkpoints are
trusted only after the host planner has byte-pinned their original same-cohort
producer. This module never reconstructs RNG with synthetic noise draws.
"""
from pathlib import Path
import hashlib
import json
import pickle
import time
from paired_alias_resident_frozen import CheckpointResident as FrozenAliasResident
from fastrsi.resident_policy import RNGSnapshot


class CheckpointResident(FrozenAliasResident):
    def __init__(self, native, **kwargs):
        super().__init__(native, **kwargs)
        pilot = Path('/pilot').resolve()
        plan = json.loads((pilot/'RESTORE_PLAN.json').read_text())
        assert plan['schema'] == 'fastrsi_object7_actual_samecohort_rng_restore_v1'
        start = RNGSnapshot()
        assert start.hashes == self.rng.hashes == plan['old_startup_rng_hashes'], 'Postload RNG differs'
        assert self.active is None and not self.used_sessions
        rows = []
        seen = set()
        try:
            for item in plan['checkpoints']:
                canonical = item['condition']
                assert canonical not in seen and canonical in self.allowed and canonical not in self.aliases
                seen.add(canonical)
                relative = Path(item['relative_path'])
                assert not relative.is_absolute() and '..' not in relative.parts
                path = (pilot/relative).resolve()
                assert path.is_relative_to(pilot)
                raw = path.read_bytes()
                assert hashlib.sha256(raw).hexdigest() == item['sha256']
                value = pickle.loads(raw)
                assert set(value) == {'python','numpy','cpu','cuda','cuda_count','hashes'}
                snapshot = RNGSnapshot()
                for key in ['python','numpy','cpu','cuda','cuda_count','hashes']:
                    setattr(snapshot, key, value[key])
                assert snapshot.cuda_count == self.rng.cuda_count == 1
                assert snapshot.hashes == item['rng_hashes']
                snapshot.restore()
                assert RNGSnapshot().hashes == item['rng_hashes']
                self.checkpoints[canonical] = snapshot
                aliases = [alias for alias, source in self.aliases.items() if source == canonical]
                for alias in aliases:
                    self.checkpoints[alias] = snapshot
                rows.append(dict(condition=canonical, aliases=aliases,
                    checkpoint_sha256=item['sha256'], rng_hashes=snapshot.hashes, roundtrip_verified=True))
        finally:
            start.restore()
        assert RNGSnapshot().hashes == plan['old_startup_rng_hashes']
        with (self.receipt_dir/'EXACT_PRIOR_BASELINE_RESTORE.json').open('x') as handle:
            json.dump(dict(time=time.time(), schema=plan['schema'], original_scope_sha256=plan['original_scope_sha256'],
                plan_sha256=hashlib.sha256((pilot/'RESTORE_PLAN.json').read_bytes()).hexdigest(),
                postload_rng_match=True, rows=rows, neural_forwards=0, environment_actions=0,
                noise_draw_reconstruction=0, old_results_modified=False), handle, indent=2)
