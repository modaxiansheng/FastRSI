"""Command descriptions only. No launch, admission, model/environment import.

The canonical accepted launch-parent bytes are copied into the NEW bundle.
Only explicit project/suite/checkpoint routing is configured here. The actor,
native loader, sampling, controller and visual model remain immutable.
"""
from pathlib import Path
import importlib.util
import json

M = Path(__file__).resolve().parent
B = Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI/planning')
POLICY = 'sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81'
ENV = 'sha256:3171fda2362d773433482f79c4fd53f6f428413efbf41c3fdfb43d8c799d6c95'
OPENVLA = {'libero_spatial': 'openvla_spatial', 'libero_object': 'openvla_object',
           'libero_goal': 'openvla_goal', 'libero_10': 'openvla_libero10'}


def generated_root():
    import hashlib
    binding = json.loads((M / 'STATE_OUTPUT_BINDING.json').read_text())
    assert binding['status'] == 'INDEPENDENT_FROZEN_ROSTER_BOUND_FOR_RUNTIME'
    root = Path(binding['generated_host_root'])
    for name, digest in binding['files'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    verification = Path(binding['verification_host_root'])
    assert verification != root
    for name, digest in binding['verification_files'].items():
        assert hashlib.sha256((verification / name).read_bytes()).hexdigest() == digest, name
    return root


def parent():
    import hashlib
    manifest = json.loads((M / 'BUILD_INPUT_MANIFEST.json').read_text())
    declaration = next(v for v in manifest['files']
                       if v['destination'] == 'launch_parent_frozen.py')
    path = M / 'launch_parent_frozen.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == declaration['sha256']
    spec = importlib.util.spec_from_file_location('capability_trial_launch_parent', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.M, module.B = M, B
    module.P = B / 'fastrsi_fixed_retry32_20261016_v1'
    module.O = B / 'liberopro_eval_20261012_v1'
    module.H = B / 'fastrsi_visual_grasp_place_20261017_v1'
    return module


def dependencies(argv, image):
    result = argv.copy()
    index = result.index(image)
    for i, value in enumerate(result[:index]):
        if value.startswith('PYTHONPATH='):
            result[i] = value + ':/runtime-deps'
    result[index:index] = ['-v', str(M / 'supplemental_runtime_deps') + ':/runtime-deps:ro']
    return result


def policy_command(root, node, gpu, model, conditions, suite):
    assert suite in OPENVLA
    # A service group is homogeneous in base suite: normalization is process-local.
    a = parent().policy_command(root, node, gpu, model, conditions)
    a = ['LIBEROPRO_BASE_SUITE=' + suite if v == 'LIBEROPRO_BASE_SUITE=libero_object'
         else v for v in a]
    if model == 'openvla':
        old = '/weights/openvla_object:/model:ro'
        replacement = '/weights/' + OPENVLA[suite] + ':/model:ro'
        assert sum(old in v for v in a) == 1
        a = [v.replace(old, replacement) for v in a]
    return dependencies(a, POLICY)


def environment_command(root, lane, node, gpu, row, kind, branch=None):
    a = parent().environment_command(root, lane, node, gpu, row, kind, branch)
    a = [v.replace(str(M / 'confirmation_generated') + ':/eval/generated:ro',
                   str(generated_root()) + ':/eval/generated:ro') for v in a]
    if branch == 'frozen':
        i = a.index(ENV)
        a[i:i] = ['-v', str(M / 'previous_runtime') + ':/eval/runtime:ro']
    if branch == 'candidate':
        a = [v.replace('PYTHONPATH=/eval/runtime_vendor:/eval/runtime:',
                       'PYTHONPATH=/eval/runtime:/eval/runtime_vendor:') for v in a]
    return dependencies(a, ENV)


def vision_command(root, node, branch, row):
    assert branch in ('frozen', 'candidate')
    assert row['skill'] is not None, 'No unused vision service on unsupported controls'
    a = parent().vision_command(root, node, branch)
    if branch == 'frozen':
        a = [v.replace(str(M / 'runtime') + ':/code:ro',
                       str(M / 'previous_runtime') + ':/code:ro') for v in a]
    a = dependencies(a, ENV)
    assert '--device' not in a and '--gpus' not in a, 'Frozen CPU neural vision'
    return a


def wire_command(root, node, gpu, model):
    return dependencies(parent().wire_command(root, node, gpu, model), ENV)
