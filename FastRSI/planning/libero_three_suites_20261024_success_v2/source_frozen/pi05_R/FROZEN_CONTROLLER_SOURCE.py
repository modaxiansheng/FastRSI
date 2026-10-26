"""Versioned paired LIBERO-10 evaluation; exact owned-container cleanup.

The coordinator owns the shared experiment lock. The second host owns its node
lease and cannot create or change a roster/model. Every case resets post-load RNG.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import py_compile
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from urllib.request import ProxyHandler, build_opener

PROJECT = Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
LEASE_ROOT = PROJECT / 'planning/multi_vla_efficiency_20261010_v1'
INPUT = LEASE_ROOT / 'benefit_max_20261010_v1/pi_engineering/pi05_budget100_packaging_fix_v3_candidate'
SOURCE_INPUT = INPUT / 'source_inputs'
CAMPAIGNS = INPUT / 'campaigns'
REFERENCE_CAMPAIGN = PROJECT / 'planning/system_efficiency_libero10_20261010_v1/campaigns/system_efficiency_libero10_three_arm_v9_20261010'
ASSETS = PROJECT.parent / 'frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1'
CORE = PROJECT.parent / 'frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1/dependencies/PhyAgentOS-core'
IMAGE = 'sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81'
TARGET_IMAGE = 'sha256:3171fda2362d773433482f79c4fd53f6f428413efbf41c3fdfb43d8c799d6c95'
PATCH = PROJECT.parent / 'node_preservation_20261008_v1/node187/runtime_patch/ndarray.pyi'
GUARD = threading.Lock()
ACTIVE = []
GROUP_AUTH = None
GROUP_ABORT = threading.Event()

def validate_group():
    if GROUP_AUTH is None:
        return
    if GROUP_ABORT.is_set():
        raise RuntimeError('Parallel group authorization lost; execution stopped')
    from parallel_group_auth_v1 import validate
    return validate(**GROUP_AUTH)

def bootstrap_authenticated_source(args):
    """Select and verify the registered snapshot before the first auth import.

    Reading a candidate manifest is not authorization. The importer is pinned
    to the controller's own frozen auth source, then the unchanged live HMAC
    validator proves the exact coordinator/global holder before any node lock.
    """
    global GROUP_AUTH, SOURCE_INPUT
    manifest = read(args.group_manifest)
    rows = [r for r in manifest['placements'] if r['node'] == args.node]
    assert len(rows) == 1, 'Ambiguous registered node'
    row = rows[0]
    assert (row['name'], row['model'], row['gpu_mask']) == (args.name, args.model, [0,1,2,3,4,5])
    assert Path(row['controller_path']).resolve() == Path(__file__).resolve()
    assert row['controller_sha256'] == sha(__file__)
    selected = Path(row['input_snapshot'])
    assert sha(selected/'SOURCE_MANIFEST.json') == row['input_source_manifest_sha256']
    bindings = read(selected/'SOURCE_MANIFEST.json')
    assert bindings == read(INPUT/'source_inputs/SOURCE_MANIFEST.json'), 'Snapshot differs from owned frozen input'
    actual_files = {str(f.relative_to(selected)) for f in selected.rglob('*') if f.is_file() and f.name != 'SOURCE_MANIFEST.json' and '__pycache__' not in f.parts and f.suffix != '.pyc'}
    assert actual_files == set(bindings), 'Unregistered file in source snapshot'
    assert all(sha(selected/rel) == digest for rel,digest in bindings.items())
    assert sha(selected/'parallel_group_auth_v1.py') == manifest['authorization_source_sha256']
    assert sha(selected/'parallel_group_auth_v1.py') == sha(INPUT/'source_inputs/parallel_group_auth_v1.py')
    SOURCE_INPUT = selected
    sys.path.insert(0,str(SOURCE_INPUT))
    existing = sys.modules.get('parallel_group_auth_v1')
    if existing is not None:
        assert Path(existing.__file__).resolve() == (SOURCE_INPUT/'parallel_group_auth_v1.py').resolve(), 'Auth importer was already bound to another source'
    GROUP_AUTH = {'manifest_path':args.group_manifest,'token':args.group_token,
        'node':args.node,'name':args.name,'model':args.model,'controller_path':__file__}
    return validate_group()


def watch_group():
    failures = 0
    while not GROUP_ABORT.wait(3):
        try:
            validate_group()
            failures = 0
        except Exception:
            failures += 1
            if failures < 3:
                continue
            GROUP_ABORT.set()
            with GUARD:
                owned = list(ACTIVE)
            for cid, owner, role, prefix in owned:
                try:
                    info = json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
                    assert info['Id'] == cid and info['Name'].startswith('/'+prefix)
                    if info['State']['Running']:
                        command(['docker','stop','-t','10',cid],45)
                except Exception:
                    pass
            return


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(argv, timeout=60):
    started = time.time()
    process = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return {'argv': argv, 'started_at': started, 'finished_at': time.time(),
            'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr}


def resource_gate(folder):
    sys.path.insert(0,str(SOURCE_INPUT))
    from pi_resource_gate_v8 import gate
    from strict_gate_natural_wait_v1 import run
    run(gate,folder,PROJECT,command,save,budget_seconds=120,interval_seconds=10)


def launch(argv, owner, role, prefix):
    validate_group()
    receipt = command(argv)
    save(owner / ('LAUNCH_' + role + '.json'), receipt)
    if receipt['returncode']:
        raise RuntimeError(receipt['stderr'])
    cid = receipt['stdout'].strip()
    info = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
    if info['Id'] != cid or not info['Name'].startswith('/' + prefix):
        raise RuntimeError('Launched container identity differs')
    with GUARD:
        ACTIVE.append((cid, owner, role, prefix))
    return cid


def wait_exit(cid, deadline):
    while time.time() < deadline:
        validate_group()
        state = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]['State']
        if not state['Running']:
            return state['ExitCode']
        time.sleep(2)
    raise TimeoutError('Owned container exceeded registered time bound')


def stop_owned(entries):
    receipts = []
    for cid, owner, role, prefix in reversed(entries):
        info = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
        assert info['Id'] == cid and info['Name'].startswith('/' + prefix)
        if info['State']['Running']:
            command(['docker', 'stop', '-t', '10', cid], 45)
        logs = subprocess.run(['docker', 'logs', cid], capture_output=True)
        (owner / (role + '.log')).write_bytes(logs.stdout + logs.stderr)
        after = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
        receipts.append({'container_id': cid, 'name': info['Name'], 'role': role,
                         'image': info['Image'], 'running': after['State']['Running'],
                         'exit_code': after['State']['ExitCode'], 'pid': after['State']['Pid'],
                         'started': info['State']['StartedAt'], 'finished': after['State']['FinishedAt']})
    return receipts


def docker_env(ports):
    return ['-e', 'FASTRSI_SERVICE_PORTS=' + json.dumps(ports, separators=(',', ':'))]


def mounts_for_campaign(argv, campaign):
    """The project is read-only; only the new campaign is writable."""
    argv = [value.replace(str(PROJECT) + ':/project:rw', str(PROJECT) + ':/project:ro') for value in argv]
    image = next(value for value in argv if value.startswith('sha256:'))
    pos = argv.index(image)
    extra = ['-v', str(campaign) + ':/project/runs/' + campaign.name + ':rw']
    if image == IMAGE:
        extra += ['-v', str(PATCH) + ':/usr/local/lib/python3.11/site-packages/numpydantic/ndarray.pyi:ro']
    argv[pos:pos] = extra
    for pos, value in enumerate(argv[:-1]):
        if value == '-v' and argv[pos + 1].endswith(':rw'):
            source = Path(argv[pos + 1].split(':', 1)[0])
            if not source.is_relative_to(campaign):
                raise ValueError('Writable mount outside this campaign')
    return argv


def prepare(name, stage, model, host):
    CAMPAIGNS.mkdir(exist_ok=True)
    campaign = CAMPAIGNS / name
    campaign.mkdir(exist_ok=False)
    source_bindings=read(SOURCE_INPUT/'SOURCE_MANIFEST.json')
    assert all(sha(SOURCE_INPUT/rel)==digest for rel,digest in source_bindings.items())
    save(campaign/'ENGINEERING_SOURCE_BINDING.json',{'source_manifest_sha256':sha(SOURCE_INPUT/'SOURCE_MANIFEST.json'),'sources':source_bindings})
    shutil.copyfile(__file__,campaign/'CONTROLLER_SOURCE.py')
    code = campaign / 'code'
    for rel in ('scripts', 'src/fastrsi', 'configs'):
        shutil.copytree(REFERENCE_CAMPAIGN / 'code' / rel, code / rel, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(SOURCE_INPUT/'parallel_group_auth_v1.py',code/'scripts/parallel_group_auth_v1.py')
    for name in ('system_case_worker.py', 'system_fast_terminal_v1.py', 'system_instrument_target.py', 'system_tool_adapter.py', 'system_failure_recovery_v1.py','multi_vla_server.py','multi_vla_binding.py','boundary_render_v2.py'):
        shutil.copyfile(SOURCE_INPUT / name, code / 'scripts' / name)
    shutil.copyfile(SOURCE_INPUT/'system_instrument_target_v2.py',code/'scripts/system_instrument_target.py')
    shutil.copyfile(SOURCE_INPUT/'pi_native_server_v8.py',code/'scripts/multi_vla_server.py')
    shutil.copyfile(SOURCE_INPUT/'pi_native_meter_v8.py',code/'scripts/pi_native_meter_v8.py')
    shutil.copyfile(SOURCE_INPUT/'pi_recovery_budget_v8.py',code/'scripts/pi_recovery_budget_v8.py')
    shutil.copyfile(SOURCE_INPUT/'pi_case_worker_v8.py',code/'scripts/system_case_worker.py')
    shutil.copyfile(SOURCE_INPUT/'pi_vla_binding_v8.py',code/'scripts/multi_vla_binding.py')
    # Deferred physical reset imports this exact original module name.
    shutil.copyfile(SOURCE_INPUT/'pi_vla_binding_v8.py',code/'scripts/pi_vla_binding_v8.py')
    shutil.copyfile(SOURCE_INPUT/'report_evidence_v2.py',code/'src/fastrsi/report_evidence.py')
    shutil.copyfile(SOURCE_INPUT/'terminal_results_v3.py',code/'src/fastrsi/terminal_results.py')
    shutil.copyfile(SOURCE_INPUT/'verification_service_reuse_v1.py',code/'scripts/verification_service_reuse_v1.py')
    if model=='openvla':
        shutil.copyfile(SOURCE_INPUT/'system_case_worker_openvla_v4.py',code/'scripts/system_case_worker.py')
    resident = code / 'src/fastrsi/resident_policy.py'
    text = resident.read_text()
    assert text.count('native.policy_config != "pi05_libero"') == 1
    resident.write_text(text.replace('native.policy_config != "pi05_libero"',
                                      'native.policy_config not in ("pi0_libero", "pi05_libero")'))
    client = code / 'src/fastrsi/resident_client.py'
    text = client.read_text().replace('not in (10, 20)', 'not in (1, 10, 20, 30, 50)')
    text = text.replace("not in ('10', '20')", "not in ('1', '10', '20', '30', '50')")
    start = text.index("        expected = {'backend':")
    end = text.index("        if not isinstance(metadata", start)
    text = text[:start] + '''        model = os.environ['FASTRSI_VLA_MODEL']
        expected = {'system_vla_model': model, 'wire_protocol': 'openpi_msgpack_numpy',
                    'action_dim': 7, 'action_horizon': self.expected_action_horizon}
        expected['backend'] = {'pi0':'openpi_native','pi05':'openpi_native'}[model]
        if model in ('pi0','pi05'):
            expected.update(policy_config=model+'_libero', torch_compile_disabled=True)
''' + text[end:]
    client.write_text(text)
    runner = code / 'scripts/run_resident_native_baseline.py'
    text = runner.read_text()
    start = text.index('    from fastrsi.feedback_window import commitment_binding')
    end = text.index('    return binding', start) + len('    return binding')
    text = text[:start] + '''    model = os.environ['FASTRSI_VLA_MODEL']
    expected = 50 if model == 'pi0' else 10
    if client.expected_action_horizon != expected:
        raise ValueError('Explicit model-specific action horizon mismatch')
    return None''' + text[end:]
    runner.write_text(text)
    worker = code / 'scripts/system_case_worker.py'
    text = worker.read_text().replace('import run_native_baseline as native',
        'import run_native_baseline as native\nfrom multi_vla_binding import install as install_binding\ninstall_binding(native)')
    worker.write_text(text)
    checked = [resident, client, runner, worker] + [code/'scripts'/file for file in
        ('multi_vla_server.py','multi_vla_binding.py','system_instrument_target.py')]
    for file in checked:
        py_compile.compile(str(file), doraise=True)
    shutil.rmtree(code/'scripts/__pycache__', ignore_errors=True)
    shutil.rmtree(code/'src/fastrsi/__pycache__', ignore_errors=True)
    save(campaign/'ADAPTER_SYNTAX_CHECK.json', {'passed':True,'files':[str(p.relative_to(code)) for p in checked]})
    shutil.copytree(ASSETS / 'runtime', campaign / 'vla_native')
    weights = ASSETS / 'weights' / {'pi0':'pi0_libero_pytorch_fp32', 'pi05':'pi05_libero_pytorch_fp32'}[model]
    save(campaign / 'VLA_CONFIG.json', {'model':model,'checkpoint':str(weights),
         'checkpoint_config_sha256':sha(weights/'config.json'),
         'action_horizon':50 if model=='pi0' else 10,
         'R_xvla_wire_chunk':30 if model=='xvla' else None})

    server = code / 'scripts/qwen_agent_server.py'
    original_server = server.read_text()
    assert original_server.count('from system_tool_adapter import parse_native_completion') == 1
    assert original_server.count("receipt['native_tool_validation_handoff']") == 1
    shutil.copyfile(REFERENCE_CAMPAIGN / 'FLOW.json', campaign / 'FLOW.json')
    shutil.copytree(REFERENCE_CAMPAIGN / 'templates', campaign / 'templates',
                    ignore=lambda directory, files: [f for f in files if not f.startswith('LAUNCH_')])
    cfg = read(REFERENCE_CAMPAIGN / 'MODEL_CONFIG.json')
    cfg['weights'] = str(PROJECT / 'models/qwen3vl8b-shared-cache-v1_20261007')
    assert Path(cfg['weights']).is_dir()
    save(campaign / 'MODEL_CONFIG.json', cfg)
    save(campaign / 'CODE_MANIFEST.json', {str(f.relative_to(code)): sha(f) for f in code.rglob('*') if f.is_file()})
    core_rel = ['PhyAgentOS/agent/loop.py', 'PhyAgentOS/agent/session_verifier.py',
                'PhyAgentOS/agent/memory.py', 'PhyAgentOS/agent/skills.py',
                'PhyAgentOS/skills/benchmarking/SKILL.md', 'PhyAgentOS/runtime/watchdog/supervisor.py']
    save(campaign / 'CORE_BINDING.json', {rel: sha(CORE / rel) for rel in core_rel})
    assert read(campaign / 'FLOW.json')['benchmark_skill_sha256'] == sha(CORE / core_rel[4])
    query = ['docker', 'run', '--rm', '--network', 'none', '-e', 'PYTHONPATH=/workspace/openpi/third_party/libero',
             '-e', 'LIBERO_CONFIG_PATH=/libero-config', '-v', str(PROJECT / 'configs/libero') + ':/libero-config:ro',
             '-v', str(CORE.parent / 'openpi') + ':/workspace/openpi:ro', TARGET_IMAGE, 'python', '-c',
             "import json;from libero.libero import benchmark;s=benchmark.get_benchmark_dict()['libero_10']();print('TASKS='+json.dumps({i:{'language':s.get_task(i).language,'initial_states':len(s.get_task_init_states(i))} for i in range(10)}))"]
    binding = command(query, 120)
    save(campaign / 'LIVE_TASK_BINDING.json', binding)
    if binding['returncode']:
        raise RuntimeError(binding['stderr'])
    languages = json.loads(next(line[6:] for line in binding['stdout'].splitlines() if line.startswith('TASKS=')))
    initials = [12] if stage == 'development' else [2, 7, 17, 22, 27, 32, 37, 42, 47, 49]
    cases = [{'task_id': task, 'init_state_id': initial, 'task_description': languages[str(task)]['language'],
              'suite': 'libero_10', 'max_steps': 520} for initial in initials for task in range(10)]
    assert all(case['init_state_id'] < languages[str(case['task_id'])]['initial_states'] for case in cases)
    arms = ['P','F','R']
    placements = [{'host':host,'lane':'lane'+str(index),'arm':arm,'gpu_base':index*2,'cases':cases}
                  for index,arm in enumerate(arms)]
    protocol = {'status': 'REGISTERED_BEFORE_OUTCOMES', 'stage': stage, 'cases_per_arm': len(cases),
                'arms': arms, 'placements': placements, 'vla_seed': 7, 'target_seed': 7,
                'action_horizon': 50 if model=='pi0' else 10, 'vla_model':model, 'R_wire_chunk':30 if model=='xvla' else None, 'native_agent_max_iterations':40,
                'native_ceiling_source': 'frozen AgentLoop.__init__ and config.schema:40', 'max_steps_per_attempt': 520, 'maximum_attempts': 3,
                'R_maximum_attempts': 1 if model=='pi0' else 3,
                'R_deterministic_reset_retry_suppression': False,
                'retry_suppression_reason': 'PI0 R registers a model-wide one-attempt cost rule before outcomes. PI0 remains stochastic; this is NOT justified as identical deterministic replay. PI05 R retains original max2replans; all canonical evidence and costs retained.',
                'per_case_limit_seconds': 3600, 'maximum_host_seconds': 200000,
                'shared_model_residency': True, 'postload_rng_reset_each_case': True,
                'evaluation_experience': 'empty initial workspace; native context readable; persistent learning writes disabled',
                'P': 'Original legacy PhyAgentOS Agent planning, original execution/semantic verification and native report',
                'F': 'Public Fast Session construction; same native execution/semantic verification and native report',
                'R': 'Public Fast Session construction plus canonical compiled terminal report and bounded failed-execution retry; actual original semantic verdicts retained; native child sessions and end hooks',
                'R_failure_recovery': ('Every PI0 R task stops after one native attempt, preserving physical failure and visual verdict' if model=='pi0' else 'PI05 R actual canonical failure may request original max2native replans; unchanged goal; all attempts count'),
                'baseline_protocol_repairs_shared': True,
                'common_source_frozen_from': str(REFERENCE_CAMPAIGN),
                'common_source_manifest_sha256': sha(REFERENCE_CAMPAIGN / 'CODE_MANIFEST.json'),
                'common_model_configuration_sha256': sha(REFERENCE_CAMPAIGN / 'MODEL_CONFIG.json'),
                'native_tool_argument_validation': 'Transport preserves unchanged valid JSON arguments; original ToolRegistry validates and provides native recovery feedback',
                'baseline_report_lossless_view_shared_with_F': True,
                'common_report_prefetch': 'Shared lossless evidence format and terminal destination binding. P/F perform actual complete fresh native read_file before their first native report model call; duplicate evidence reads blocked only after complete coverage, and the native model still writes its report. R compiled terminal directly reads complete canonical sources without a report LLM. All arms retain destination, reference and canonical source checks',
                'shared_verifier_transport_repair': 'All arms bind 274xx/275xx/276xx ports below the host ephemeral range and allow reuse of closed verifier sockets; original engine, verdict, request validation and authentication unchanged',
                'common_report_protocol': 'lossless columns/constants/string-prefix representation of complete original chunk arrays; exact value roundtrip and SHA checked; all arms use same evidence codec and report instructions; canonical logs unchanged',
                'diagnostic_predecessor': 'Historical PI05 v9 and PI0 multiv1 retained; new P/F/R all rerun with shared v7 lossless codec/prefetch. No historical baseline numbers reused.',
                'all_failed_cases_and_retries_included': True,
                'model_manifest_sha256': sha(campaign / 'CODE_MANIFEST.json'),
                'roster_changes_after_outcomes': False, 'case_order_fixed': True,
                'independent_unseen_initial_claim': False}
    protocol['candidate_version']='pi05_task8_budget100_v10_cpu_v1'
    protocol['registered_budget']={'P':{'first':520,'child':520},'F':{'first':520,'child':520},'R':{'first_task8':100,'first_other_tasks':320,'child':520,'max_replans':2}}
    protocol['R_failure_recovery']='Registered task8 first100, other9 first320; every child520/max2. Known development task-conditioned stopping. Independent case begin restores actual post-load RNG, child handoff does not reset; no noise replay/alignment/new seed.'
    protocol['retry_suppression_reason']='PI05 task-conditioned development budget changes first stopping condition, original max2 child replans retained; no deterministic retry claim'
    protocol['quality_gate']='Final native R must retain prior R10/10; first success, semantic success and agreement distinct. Failed candidate kept negative, never overwrites v8.'
    protocol['pairing_scope']='First-generation inputs and executed actions compared exactly up to first-attempt budget prefix; full length equality required when no cut occurs.'
    protocol['metering']='Actual sample_actions and vision/language/denoising forward entries measured equally in P/F/R. No sampler parameter or RNG changes; host timing is unsynchronized.'
    protocol['quality_metrics']='Final native success, final semantic success, and native-semantic agreement are distinct; retain all three.'
    protocol['baseline_comparability']='Fresh P/F use common lossless report codec and native fresh prefetch, R compiled canonical terminal report without report LLM. Historical P/F are not this candidate baseline.'
    protocol['R_boundary_render']={'enabled':True,'source':'boundary_render_v2.py','scope':'Native sensor cadence and every physics step retained; render at action-block boundaries, refresh on early terminal success; full video recording falls back to native rendering.'}
    shutil.copyfile(SOURCE_INPUT/'boundary_probe_v2/BOUNDARY_RENDER_CPU_RECEIPT_v2.json',campaign/'BOUNDARY_RENDER_CPU_RECEIPT_v2.json')
    save(campaign / 'PROTOCOL.json', protocol)
    return campaign


def run_lane(campaign, placement):
    cfg = read(campaign / 'MODEL_CONFIG.json')
    vla = read(campaign / 'VLA_CONFIG.json')
    host = placement['host']
    lane = campaign / ('node' + host) / placement['lane']
    lane.mkdir(exist_ok=False)
    base = '/project/runs/' + campaign.name + '/' + str(lane.relative_to(campaign))
    codebase = '/project/runs/' + campaign.name + '/code'
    ports = dict(zip(('policy', 'target', 'qwen', 'verifier'),
                     (27400 + placement['gpu_base'] * 50, 27402 + placement['gpu_base'] * 50,
                      27410 + placement['gpu_base'] * 50, 27420 + placement['gpu_base'] * 50)))
    ephemeral_low,ephemeral_high=map(int,Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split())
    assert all(not ephemeral_low<=p<=ephemeral_high for p in ports.values())
    for port in ports.values():
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError('Registered port is busy')
    arm = placement['arm']
    prefix = 'yuchen-fastrsi-' + campaign.name + '-n' + host + '-' + placement['lane']
    started = time.time()
    state = {'status': 'STARTING', 'arm': arm, 'host': host, 'started_at': started, 'cases': []}
    save(lane / 'STATUS.json', state)
    local_active = []
    services = {}
    try:
        (lane / 'TARGET_COST_EVENTS.jsonl').touch(exist_ok=False)
        cache = lane / 'tokenizer_cache'
        shutil.copytree(PROJECT / 'models/openpi_data', cache)
        sys.path.insert(0, str(campaign / 'code/scripts'))
        from native_service_config import configure_qwen
        from types import SimpleNamespace
        life = SimpleNamespace(IMAGE=IMAGE, PROJECT=PROJECT, BASE='/project/runs/' + campaign.name)
        for role, template in [('pi05', 'LAUNCH_pi05_v2.json'), ('qwen', 'LAUNCH_qwen.json'), ('target', 'LAUNCH_target.json')]:
            argv = list(read(campaign / 'templates' / template)['argv'])
            argv[argv.index('--name') + 1] = prefix + '-' + role
            argv = [value.replace(str(PROJECT / 'runs/pilot_20261001'), str(lane))
                    .replace('/project/runs/pilot_20261001', base) for value in argv]
            if role in ('pi05', 'qwen'):
                card = placement['gpu_base'] + (role == 'qwen')
                argv = [('HIP_VISIBLE_DEVICES=' + str(card)) if value.startswith('HIP_VISIBLE_DEVICES=') else
                        ('CUDA_VISIBLE_DEVICES=' + str(card)) if value.startswith('CUDA_VISIBLE_DEVICES=') else value for value in argv]
            argv[argv.index('timeout') + 1] = '200000'
            if role == 'pi05':
                argv = [value.replace(str(PROJECT / 'models/openpi_data'), str(cache)) for value in argv]
                env_pos = argv.index(IMAGE)
                argv[env_pos:env_pos] = ['-v', str(PROJECT) + ':/project:ro', '-e',
                    'PYTHONPATH=/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src:/workspace/PhyAgentOS-core:' + codebase + '/src']
                checkpoint = Path(vla['checkpoint'])
                found = 0
                for i,value in enumerate(argv):
                    if value=='-v' and argv[i+1].endswith(':/model:ro'):
                        argv[i+1]=str(checkpoint)+':/model:ro'
                        found += 1
                assert found == 1, 'Exactly one actual checkpoint mount is required'
                pos=argv.index(IMAGE)
                overlay=ASSETS/'dependencies'/('xvla_overlay' if vla['model']=='xvla' else 'openvla_overlay')
                extra=['-v',str(campaign/'vla_native')+':/vla-native:ro']
                if vla['model'] not in ('pi0','pi05'):
                    extra+=['-v',str(overlay/'site')+':/overlay/site:ro',
                            '-v',str(ASSETS/'dependencies')+':/vla-dependencies:ro',
                            '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
                            '-e','HF_HOME=/vla-dependencies/'+vla['model']+'_hf_cache',
                            '-e','HF_MODULES_CACHE=/tmp/hf_modules','-e','USE_TF=0',
                            '-e','TF_NUM_INTRAOP_THREADS=2','-e','TF_NUM_INTEROP_THREADS=2',
                            '-e','PYTHONPATH=/overlay/site:/workspace/PhyAgentOS-core:'+codebase+'/scripts:'+codebase+'/src']
                argv[pos:pos]=extra
                wire=30 if vla['model']=='xvla' and arm=='R' else 1
                argv[-1] = ('source /opt/dtk/env.sh && source /opt/hyhal/env.sh && exec python ' + codebase+
                    '/scripts/multi_vla_server.py --model '+vla['model']+' --checkpoint /model --host 127.0.0.1 --port '+
                    str(ports['policy'])+' --receipt-dir /pilot/resident_receipts --xvla-wire-chunk '+str(wire))
            elif role == 'qwen':
                argv = configure_qwen(argv, cfg, life)
                argv[argv.index(IMAGE):argv.index(IMAGE)] = docker_env(ports)
            else:
                argv[argv.index(TARGET_IMAGE):argv.index(TARGET_IMAGE)]=['-e','SYSTEM_TARGET_RENDER_BOUNDARIES='+('1' if arm=='R' else '0')]
                argv = [value.replace('/project/scripts/instrument_target.py', codebase + '/scripts/system_instrument_target.py')
                        .replace(':/project/scripts', ':' + codebase + '/scripts') for value in argv]
                argv[argv.index('--port') + 1] = str(ports['target'])
            argv = mounts_for_campaign(argv, campaign)
            cid = launch(argv, lane, role, prefix)
            services[role] = cid
            local_active.append((cid, lane, role, prefix))
        opener = build_opener(ProxyHandler({}))
        until = time.time() + 600
        for port in (ports['policy'], ports['qwen']):
            while time.time() < until:
                for cid in services.values():
                    if not json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]['State']['Running']:
                        raise RuntimeError('Owned service exited during loading')
                try:
                    with opener.open('http://127.0.0.1:' + str(port) + '/healthz', timeout=2) as reply:
                        if reply.status == 200:
                            break
                except Exception:
                    time.sleep(1)
            else:
                raise TimeoutError('Service readiness')
        with socket.create_connection(('127.0.0.1', ports['target']), timeout=10):
            pass
        state.update(status='RUNNING', startup_seconds=time.time() - started)
        save(lane / 'STATUS.json', state)
        for sequence, item in enumerate(placement['cases']):
            validate_group()
            case_dir = lane / ('case%03d_t%d_i%d' % (sequence, item['task_id'], item['init_state_id']))
            case_dir.mkdir()
            stream = case_dir / 'stream'
            stream.mkdir()
            (stream / 'TARGET_COST_EVENTS.jsonl').symlink_to('../../TARGET_COST_EVENTS.jsonl')
            cid = 'system_t%d_i%d' % (item['task_id'], item['init_state_id'])
            case = {**item, 'id': cid, 'mode': 'evaluation', 'session_intent': True,
                    'require_discovery_reads': True, 'require_clean_model_calls': True,
                    'read_files_tool': True, 'terminal_evidence_view': True,
                    'terminal_bind_report_destination': True, 'native_experience_maintenance': True,
                    'conversation_key': 'cli:' + cid, 'require_compiled_semantic_receipt': arm == 'R'}
            case['terminal_prefetch']=True
            if arm in ('F', 'R'):
                case.update(session_flow_artifact='/project/runs/' + campaign.name + '/FLOW.json',
                            session_flow_sha256=sha(campaign / 'FLOW.json'))
            save(case_dir / 'ROSTER.json', [case])
            save(case_dir / 'MODEL_CONFIG.json', cfg)
            casebase = '/project/runs/' + campaign.name + '/' + str(case_dir.relative_to(campaign))
            cache = PROJECT / 'models/tiktoken_cache_v1'
            argv = ['docker', 'run', '-d', '--name', prefix + '-case' + str(sequence), '--network', 'host',
                    '--cpus', '8', '--memory', '16g', '-e', 'PYTHONDONTWRITEBYTECODE=1',
                    '-e', 'TIKTOKEN_CACHE_DIR=/tiktoken-cache', '-v', str(cache) + ':/tiktoken-cache:ro',
                    '-e', 'PYTHONPATH=/core:' + codebase + ':' + codebase + '/scripts:' + codebase + '/src:/project/envs/structured-decode-v1',
                    '-v', str(PROJECT) + ':/project:ro', '-v', str(CORE) + ':/core:ro',
                    '-v', str(campaign / 'code/scripts/engine_overlay.py') + ':/core/PhyAgentOS/verification/engine.py:ro',
                    '-v', str(campaign / 'code/scripts/verification_service_reuse_v1.py') + ':/core/PhyAgentOS/verification/service.py:ro',
                    '-e', 'FASTRSI_NATIVE_RUN=' + casebase + '/stream', '-e', 'FASTRSI_CONSUME_TERMINAL=1',
                    '-e', 'FASTRSI_EXPERIENCE_ADMISSION=legacy_raw', '-e', 'FASTRSI_AGENT_MODEL=' + cfg['model_id'],
                    '-e', 'SYSTEM_EFFICIENCY_ARM=' + arm,
                    '-e', 'SYSTEM_NATIVE_FAILURE_RETRY=' + ('1' if arm == 'R' and vla['model']=='pi05' else '0'),
                    '-e', 'FASTRSI_RESIDENT_POLICY_PORT=' + str(ports['policy']),
                    '-e', 'FASTRSI_RESIDENT_SESSION_ID=' + cid,
                    '-e', 'FASTRSI_RESIDENT_MAX_INFERENCES=1600',
                    '-e','FASTRSI_VLA_MODEL='+vla['model'],
                    '-e','SYSTEM_XVLA_WIRE_CHUNK='+('30' if vla['model']=='xvla' and arm=='R' else '1'),
                    '-e','FASTRSI_RESIDENT_INFERENCE_HORIZON='+str(50 if vla['model']=='pi0' else 10)] + docker_env(ports) + [IMAGE, 'timeout', '3600',
                    'bash', '-lc', 'source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python ' +
                    codebase + '/scripts/system_case_worker.py']
            argv = mounts_for_campaign(argv, campaign)
            case_start = time.time()
            runner = launch(argv, case_dir, 'runner', prefix)
            entry = (runner, case_dir, 'runner', prefix)
            local_active.append(entry)
            exit_code = wait_exit(runner, time.time() + 3620)
            result_file = stream / cid / 'RESULT.json'
            result = read(result_file) if result_file.exists() else {}
            row = {'case_dir': str(case_dir.relative_to(campaign)), 'task_id': item['task_id'],
                   'init_state_id': item['init_state_id'], 'exit_code': exit_code,
                   'chain_status': result.get('chain_status', 'NO_RESULT'),
                   'environment_success': result.get('environment_success'),
                   'host_case_seconds': time.time() - case_start, 'native_case_seconds': result.get('seconds')}
            state['cases'].append(row)
            save(case_dir / 'STOP_RECEIPT.json', stop_owned([entry]))
            local_active.remove(entry)
            state['completed_cases'] = len(state['cases'])
            save(lane / 'STATUS.json', state)
            print(json.dumps({'host': host, 'lane': placement['lane'], 'arm': arm, **row}), flush=True)
            if exit_code or result.get('chain_status') != 'PASSED':
                raise RuntimeError('Paid case chain failure retained; pause lane for diagnosis')
        state['status'] = 'COMPLETED'
    except BaseException:
        state.update(status='FAILED', error=traceback.format_exc())
    finally:
        save(lane / 'STOP_RECEIPT.json', stop_owned(local_active))
        state.update(finished_at=time.time(), full_wall_seconds=time.time() - started)
        save(lane / 'STATUS.json', state)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--node', required=True, choices=['187', '188'])
    parser.add_argument('--stage', choices=['development'], required=True)
    parser.add_argument('--model',required=True,choices=['pi05'])
    parser.add_argument('--wait-for-lock',action='store_true')
    parser.add_argument('--group-manifest')
    parser.add_argument('--group-token')
    parser.add_argument('--secondary', action='store_true')
    parser.add_argument('--external-secondary', action='store_true')
    args = parser.parse_args()
    assert re.fullmatch('system_efficiency_libero10_[A-Za-z0-9_]+', args.name)
    addresses = json.loads(subprocess.check_output(['ip', '-j', 'address']))
    assert any(a.get('local') == '10.232.1.' + args.node for row in addresses for a in row.get('addr_info', []))
    assert sha(PATCH) == 'a68259f666290b3241be3e5cbf1ff6eb098f96dc694f2c4b511a554bf981a99c'
    global GROUP_AUTH, SOURCE_INPUT
    assert bool(args.group_manifest) == bool(args.group_token)
    if args.group_manifest:
        assert not args.secondary and not args.external_secondary
        bootstrap_authenticated_source(args)
    elif args.secondary:
        raise RuntimeError('Unauthenticated secondary is disabled; use the registered parallel group')
    locks = []
    lock_names = ['system_efficiency_node' + args.node + '.lock']
    if GROUP_AUTH is None:
        lock_names.append('full_os_v1.lock')
    for name in lock_names:
        handle = (PROJECT / 'runs' / name).open('a')
        deadline = time.time()+200000
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not args.wait_for_lock or time.time()>=deadline:
                    raise
                time.sleep(10)
        locks.append(handle)
    shared_lease = (LEASE_ROOT / ('active_node' + args.node + '.lock')).open('a')
    fcntl.flock(shared_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    locks.append(shared_lease)
    node_lease_event={'event':'NODE_LEASE_ACQUIRED','node':args.node,'utc':time.time(),'mono_ns':time.monotonic_ns(),'gpu_mask':[0,1,2,3,4,5]}
    if args.group_manifest:save(Path(args.group_manifest).parent/('NODE_LEASE_ACQUIRED_'+args.node+'.json'),node_lease_event)
    if args.secondary:
        campaign = CAMPAIGNS / args.name
        assert read(campaign / 'PROTOCOL.json')['stage'] == args.stage
    else:
        preflight = INPUT / (args.name + '_preflight')
        preflight.mkdir(exist_ok=False)
        resource_gate(preflight)
        campaign = prepare(args.name, args.stage, args.model, args.node)
    if GROUP_AUTH is not None:
        save(campaign/'PARALLEL_GROUP_BINDING.json',{
            'manifest_path':args.group_manifest,
            'manifest_sha256':sha(args.group_manifest),
            'coordinator_status':validate_group(),
            'node':args.node,'model':args.model,'authorized_gpu_mask':[0,1,2,3,4,5]})
        threading.Thread(target=watch_group,daemon=True).start()
    anchor = PROJECT / 'runs' / args.name
    anchor.mkdir(exist_ok=True)
    if any(anchor.iterdir()):
        raise RuntimeError('Container mount anchor must be an empty owned directory')
    mount_check = command(['docker', 'run', '--rm', '--network', 'none',
        '-v', str(PROJECT) + ':/project:ro',
        '-v', str(campaign) + ':/project/runs/' + campaign.name + ':rw',
        TARGET_IMAGE, 'python', '-c', "from pathlib import Path;p=Path('/project/runs/" + campaign.name + "');assert (p/'PROTOCOL.json').is_file();print('CAMPAIGN_MOUNT_VERIFIED')"], 60)
    save(campaign / ('MOUNT_CHECK_node' + args.node + '.json'), mount_check)
    if mount_check['returncode']:
        raise RuntimeError(mount_check['stderr'])
    assert sha(campaign / 'CODE_MANIFEST.json') == read(campaign / 'PROTOCOL.json')['model_manifest_sha256']
    node_dir = campaign / ('node' + args.node)
    node_dir.mkdir(exist_ok=False)
    started = time.time()
    state = {'status': 'PREFLIGHT', 'started_at': started, 'node': args.node, 'controller_pid':os.getpid(),
             'parallel_group':args.group_manifest, 'controller_sha256': sha(__file__), 'registration_sha256': sha(campaign / 'PROTOCOL.json')}
    save(node_dir / 'STATUS.json', state)
    secondary = None
    try:
        resource_gate(node_dir)
        if False:  # This controller owns one node; model lanes have an identical full roster.
            remote = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'gpu-187', 'python3',
                      str(INPUT / 'campaign_controller_v9.py'), '--name', args.name, '--node', '187',
                      '--stage', args.stage, '--secondary']
            log = (node_dir / 'secondary_controller.log').open('x')
            secondary = subprocess.Popen(remote, stdout=log, stderr=subprocess.STDOUT)
            save(node_dir / 'SECONDARY_DISPATCH.json', {'argv': remote, 'pid': secondary.pid, 'time': time.time()})
        placements = [p for p in read(campaign / 'PROTOCOL.json')['placements'] if p['host'] == args.node]
        state['status'] = 'RUNNING'
        save(node_dir / 'STATUS.json', state)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(placements)) as pool:
            states = list(pool.map(lambda p: run_lane(campaign, p), placements))
        state['lanes'] = states
        if args.external_secondary:
            deadline = time.time() + 15000
            other = campaign / ('node187/STATUS.json' if args.node=='188' else 'node188/STATUS.json')
            while time.time() < deadline:
                if other.exists() and read(other)['status'] in ('COMPLETED', 'FAILED'):
                    break
                time.sleep(2)
            else:
                raise TimeoutError('External secondary controller completion')
            if read(other)['status'] != 'COMPLETED':
                raise RuntimeError('External secondary controller failed')
        state['lanes'] = states
        state['status'] = 'COMPLETED' if all(row['status'] == 'COMPLETED' for row in states) else 'FAILED'
        if secondary is not None:
            code = secondary.wait(timeout=15000)
            state['secondary_exit'] = code
            if code:
                state['status'] = 'FAILED'
    except BaseException:
        state.update(status='FAILED', error=traceback.format_exc())
    finally:
        state.update(finished_at=time.time(), full_wall_seconds=time.time() - started)
        save(node_dir / 'STATUS.json', state)
        if not args.secondary:
            other = campaign / ('node187/STATUS.json' if args.node=='188' else 'node188/STATUS.json')
            save(campaign / 'STATUS.json', {'status': state['status'], 'primary': state,
                 'secondary': read(other) if other.exists() else None, 'finished_at': time.time()})
        for held in reversed(locks):held.close()
        if args.group_manifest:save(Path(args.group_manifest).parent/('NODE_LEASE_RELEASED_'+args.node+'.json'),{'event':'NODE_LEASE_RELEASED','node':args.node,'utc':time.time(),'mono_ns':time.monotonic_ns()})
        print(json.dumps({'node': args.node, 'status': state['status']}), flush=True)
    if state['status'] != 'COMPLETED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
