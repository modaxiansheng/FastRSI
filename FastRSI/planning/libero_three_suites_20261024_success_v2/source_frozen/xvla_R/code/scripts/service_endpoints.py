"""One explicit endpoint binding shared by native consumers and services.

Absent the environment binding, callers retain their historical defaults (even
in old per-arm source snapshots). New workers pass the complete binding instead
of editing Python source. No model, task or feedback setting lives here.
"""
import json
import os
import re

ENV = 'FASTRSI_SERVICE_PORTS'
ROLES = ('policy', 'target', 'qwen', 'verifier')
DEFAULTS = (49500, 49502, 49510, 49520)


def validate_ports(ports):
    if not isinstance(ports, dict) or set(ports) != set(ROLES):
        raise ValueError('Service binding requires policy, target, qwen and verifier')
    if any(type(p) is not int or not 1 <= p <= 65535 for p in ports.values()):
        raise ValueError('Invalid TCP port')
    if len(set(ports.values())) != len(ROLES):
        raise ValueError('Services must use distinct ports')
    return dict(ports)


def service_ports(defaults=DEFAULTS, environ=None):
    env = os.environ if environ is None else environ
    if ENV in env:
        return validate_ports(json.loads(env[ENV]))
    if len(defaults) != len(ROLES):
        raise ValueError('Incomplete legacy port defaults')
    return validate_ports(dict(zip(ROLES, defaults)))


def shifted_ports(shift):
    if type(shift) is not int:
        raise ValueError('Port offset must be an integer')
    return validate_ports(dict(zip(ROLES, (p + shift for p in DEFAULTS))))


def docker_port_env(ports):
    return ['-e', ENV + '=' + json.dumps(validate_ports(ports), separators=(',', ':'))]


def relocate_service_command(argv, shift, gpu_base):
    """Relocate legacy service launch templates, never Python source.

    Substitute each port once: overlapping old/new values must not cascade.
    Explicit JSON endpoint bindings are already resolved. Device allocation is
    independent of ports, including the zero-offset arm.
    """
    ports = shifted_ports(shift)
    if type(gpu_base) is not int or gpu_base < 0:
        raise ValueError('GPU base must be a nonnegative integer')
    mapping = dict(zip(map(str, DEFAULTS), map(str, ports.values())))
    pattern = re.compile(r'(?<!\d)(?:' + '|'.join(mapping) + r')(?!\d)')
    out = []
    for value in argv:
        if not value.startswith(ENV + '='):
            value = pattern.sub(lambda m: mapping[m.group()], value)
        for prefix in ('HIP_VISIBLE_DEVICES=', 'CUDA_VISIBLE_DEVICES='):
            if value in (prefix + '0', prefix + '1'):
                value = prefix + str(gpu_base + int(value[-1]))
                break
        out.append(value)
    return out
