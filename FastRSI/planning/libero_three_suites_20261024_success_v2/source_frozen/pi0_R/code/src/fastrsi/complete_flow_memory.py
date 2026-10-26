"""Cheap retrieval baseline for whole, versioned execution-flow experiences.

Exact legal-observation reuse is deliberately narrow. It does not claim unseen
scene generalization, parameter distillation, or intermediate action optimality.
"""
import hashlib

LEGAL_KEYS = ('observation/image', 'observation/wrist_image', 'observation/state', 'prompt')


def legal_key(observation, packb):
    return hashlib.sha256(packb({k: observation[k] for k in LEGAL_KEYS})).hexdigest()


class CompleteFlowMemory:
    def __init__(self, registry, model_sha256, server_sha256):
        if registry['teacher_model_sha256'] != model_sha256 or registry['teacher_server_sha256'] != server_sha256:
            raise ValueError('Whole-flow evidence belongs to a different executable version')
        self.entries = registry['entries']
        self.sessions = {}

    def choose(self, session_id, step, observation, packb):
        if session_id not in self.sessions:
            if step != 0:
                raise ValueError('Cannot install an initial-state flow choice halfway through execution')
            key = legal_key(observation, packb)
            entry = self.entries.get(key)
            self.sessions[session_id] = dict(mode=entry['mode'] if entry else 'N',
                legal_input_sha256=key, evidence_read=bool(entry),
                scope='exact observed scene; future-noise transfer requires real testing')
        return self.sessions[session_id].copy()
