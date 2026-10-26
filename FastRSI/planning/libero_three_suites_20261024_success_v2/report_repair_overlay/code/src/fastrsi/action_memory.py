"""Bounded observed-action memory. Similarity is a pilot heuristic, not safety proof."""
import hashlib
import json
import numpy as np

GOAL = 'stack the right bowl on the left bowl and place them in the tray'
LIMITS = dict(position=.01, rotation=.10, gripper=.004,
              image_mean=.02, image_patch=.20, wrist_mean=.02, wrist_patch=.20)


def feature(image):
    a = np.asarray(image)
    if a.dtype != np.uint8 or a.shape != (256, 256, 3):
        raise ValueError('Expected the original 256x256 RGB observation')
    return a.astype(np.float32).reshape(16,16,16,16,3).mean(axis=(1,3)) / 255.


def state(value):
    a = np.asarray(value, dtype=np.float32)
    if a.shape != (8,) or not np.isfinite(a).all():
        raise ValueError('Invalid visible proprioception')
    return a


def open_motion(s, actions):
    """Restrict the first experiment to elevated, open-gripper motion segments.

    This checks available commands/state, not hidden contact or object truth.
    """
    a = np.asarray(actions)
    return bool(a.shape == (10,7) and np.isfinite(a).all() and
                np.all(a[:,-1] <= -.9) and s[2] >= .55 and np.all(np.abs(s[6:]) >= .03))


class ActionMemory:
    def __init__(self, artifact):
        if (artifact.get('version') != 1 or artifact.get('goal') != GOAL or
                artifact.get('limits') != LIMITS or not 1 <= len(artifact.get('entries', [])) <= 40):
            raise ValueError('Unbound action memory')
        self.entries = []
        previous = -1
        for entry in artifact['entries']:
            s = state(entry['state']); actions = np.asarray(entry['actions'], dtype=np.float32)
            images = [np.asarray(entry[k], dtype=np.float32) for k in ('image','wrist')]
            if (type(entry['source_step']) is not int or entry['source_step'] <= previous or
                    not open_motion(s, actions) or any(a.shape != (16,16,3) or
                    not np.isfinite(a).all() or np.any((a<0)|(a>1)) for a in images)):
                raise ValueError('Malformed source motion')
            previous = entry['source_step']
            self.entries.append((entry['source_step'], s, actions, *images))
        self.last_source = -1

    def choose(self, observation):
        if observation.get('prompt') != GOAL:
            return None, dict(reason='goal_mismatch')
        s = state(observation['observation/state'])
        if s[2] < .55 or np.any(np.abs(s[6:]) < .03):
            return None, dict(reason='outside_open_motion')
        images = [feature(observation[k]) for k in ('observation/image','observation/wrist_image')]
        best = None
        for step, ref, actions, front, wrist in self.entries:
            if step <= self.last_source:
                continue
            metrics = dict(position=float(np.linalg.norm(s[:3]-ref[:3])),
                           rotation=float(np.linalg.norm(s[3:6]-ref[3:6])),
                           gripper=float(np.max(np.abs(s[6:]-ref[6:]))))
            for name, current, saved in zip(('image','wrist'), images, (front,wrist)):
                d = np.abs(current-saved)
                metrics[name+'_mean'] = float(d.mean())
                metrics[name+'_patch'] = float(d.mean(axis=-1).max())
            score = max(metrics[k]/LIMITS[k] for k in LIMITS)
            if best is None or score < best[0]:
                best = (score, step, actions, metrics)
        if best is None:
            return None, dict(reason='memory_exhausted')
        score, step, actions, metrics = best
        receipt = dict(reason='similarity_rejected', source_step=step, score=score, metrics=metrics)
        if score > 1:
            return None, receipt
        self.last_source = step
        return actions.copy(), {**receipt, 'reason':'observed_motion_reused'}


def load_bound(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Memory artifact changed')
    value = json.loads(raw)
    ActionMemory(value)
    return value
