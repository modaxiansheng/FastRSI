"""Reuse the complete native Agent lifecycle for one fixed local motion skill."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import orchestrate_native_sampler_agent_arm_v1 as native

FILES = ["src/fastrsi/local_bowl_transition_v1.py",
         "src/fastrsi/public_translation_gate_v1.py",
         "scripts/run_local_bowl_transition_native_v1.py",
         "scripts/instrument_local_bowl_transition_target_v1.py",
         "scripts/orchestrate_local_bowl_transition_v1.py",
         "tests/test_local_bowl_transition_v1.py",
         "tests/test_native_stage_continuation_v1.py"]
RECIPE = dict(version="local_bowl_transition_v1",max_local_actions=40,chunk_steps=5,
    hover_z=.62,position_tolerance=.01,max_normalized_translation=.4,open_finger_min=.03,
    search_radius_pixels=24,minimum_match_score=.8,minimum_peak_gap=.05)


def prepare(project,campaign,arm,draw_seed=None,resource_slot=None,sampler_input=None,candidate_snapshot=None):
    project,campaign=Path(project).resolve(),Path(campaign).resolve()
    source=project/"runs/native_successor_operation_B_640715_v3_20261007/SOURCE_BINDINGS.json"
    bindings=native.read(source)
    if sampler_input is not None:
        root=Path(sampler_input).resolve();root.relative_to(project/"planning")
        reg=root/"REGISTRATION.json";manifest=root/"outputs/samples/MANIFEST.json"
        bindings["sampler_registration"]=dict(path=str(reg),sha256=native.sha(reg))
        bindings["sampler_manifest"]=dict(path=str(manifest),sha256=native.sha(manifest))
    if draw_seed is not None:
        # The native preparer still validates membership in the unchanged
        # development registration and binds the exact saved sampler state.
        registration=native.read(Path(bindings["sampler_registration"]["path"]))
        native.validate_sampler_partition(registration,bindings["mode"],draw_seed)
        bindings["draw_seed"]=draw_seed
    # Preserve the tested older runtime files. Its old closure lacks scripts
    # referenced by today's freezer; supplement missing files in a NEW bundle.
    old_code=Path(bindings["runtime_code_manifest"]["path"]).parent/"code"
    runtime_base=campaign.parent/(campaign.name+"_runtime_base")
    if runtime_base.exists():
        raise FileExistsError("Existing runtime base must remain immutable")
    shutil.copytree(old_code,runtime_base/"code",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    for subtree in ("scripts","src/fastrsi"):
        for path in (project/subtree).rglob("*.py"):
            destination=runtime_base/"code"/path.relative_to(project)
            if not destination.exists():
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,destination)
    native.save(runtime_base/"CODE_MANIFEST.json",native.manifest(runtime_base/"code"))
    bindings["runtime_code_manifest"]=dict(path=str(runtime_base/"CODE_MANIFEST.json"),sha256=native.sha(runtime_base/"CODE_MANIFEST.json"))
    slot=(0 if arm=="native" else 1) if resource_slot is None else resource_slot
    if type(slot) is not int or slot not in range(4):
        raise ValueError("Explicit independent resource slot 0..3")
    gpu_pair=[slot*2,slot*2+1]
    buses={0:"09",1:"36",2:"55",3:"77",4:"85",5:"b5",6:"d5",7:"f5"}
    port=53500+100*slot
    bindings["resources"]=dict(gpus=gpu_pair,device_map={str(i):dict(pci_bus="0000:"+buses[i]+":00.0",
        render_node="/dev/dri/renderD"+str(128+i)) for i in gpu_pair},
        ports=dict(policy=port,target=port+2,qwen=port+10,verifier=port+20))
    staging=campaign.parent/(campaign.name+"_source_bindings.json")
    if staging.exists():
        raise FileExistsError("Existing preparation must be inspected, never overwritten")
    native.save(staging,bindings)
    native.prepare(project,campaign,staging)
    for relative in FILES:
        destination=campaign/"code"/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(project/relative,destination)
        if native.sha(destination)!=native.sha(project/relative):
            raise ValueError("Changed code during new candidate freeze")
    if sampler_input is not None:
        relative="src/fastrsi/native_sampler_restore_v1.py"
        shutil.copyfile(project/relative,campaign/"code"/relative)
    if candidate_snapshot is not None:
        root=Path(candidate_snapshot).resolve();root.relative_to(project/"runs")
        # Exact previously executed operation/runner/tests; no gate or new rule.
        for relative in ("src/fastrsi/local_bowl_transition_v1.py",
                         "scripts/run_local_bowl_transition_native_v1.py",
                         "tests/test_local_bowl_transition_v1.py"):
            shutil.copyfile(root/"code"/relative,campaign/"code"/relative)
    case=native.read(campaign/"ROSTER.json")[0]
    base="/project/"+campaign.relative_to(project).as_posix()
    refs={}
    for key,name in (("source_rgb","SOURCE.png"),("calibration","CALIBRATION.json")):
        path=campaign/"initial_experience/configs/source_successor_v1"/name
        refs[key]=dict(path=base+"/initial_experience/configs/source_successor_v1/"+name,sha256=native.sha(path))
    cfg=dict(enabled=arm!="native",recipe=RECIPE,source_rois=[[166,126,209,157],[83,122,127,155]],**refs)
    if arm=="gated":
        cfg["eligibility_gate"]="public_translation_history_v1"
    native.save(campaign/"LOCAL_TRANSITION_CONFIG.json",cfg)
    case["local_bowl_transition"]=dict(path=base+"/LOCAL_TRANSITION_CONFIG.json",sha256=native.sha(campaign/"LOCAL_TRANSITION_CONFIG.json"))
    native.save(campaign/"ROSTER.json",[case])
    target=native.read(campaign/"TARGET_CONFIG.json")
    target["local_transition_recipe"]=RECIPE
    native.save(campaign/"TARGET_CONFIG.json",target)
    reg=native.read(campaign/"REGISTRATION.json")
    reg.update(local_bowl_transition_v1=cfg,operation_arm=arm,runner_timeout_seconds=1500,
        service_timeout_seconds=1800,controller_timeout_seconds=1800,
        scope="Complete original Agent developmental operation: public feedback lift/approach; not learned/heldout gain")
    native.save(campaign/"CODE_MANIFEST.json",native.manifest(campaign/"code"))
    reg["code_manifest_sha256"]=native.sha(campaign/"CODE_MANIFEST.json")
    native.save(campaign/"REGISTRATION.json",reg)
    inputs=native.manifest(campaign)
    inputs.pop("MANIFEST.json",None)
    native.save(campaign/"MANIFEST.json",inputs)
    return dict(status="PREPARED_NOT_DISPATCHED",campaign=str(campaign),manifest_sha256=native.sha(campaign/"MANIFEST.json"))


original_launch=native.launch_argv
def launch(project,campaign,role):
    argv=original_launch(project,campaign,role)
    old,new=("instrument_native_sampler_draw_target_v1.py","instrument_local_bowl_transition_target_v1.py") if role=="target" else (
        ("run_native_sampler_goal_component_v1.py","run_local_bowl_transition_native_v1.py") if role=="runner" else
        ("orchestrate_native_sampler_agent_arm_v1.py","orchestrate_local_bowl_transition_v1.py"))
    if role in ("target","runner","cpu"):
        argv=[item.replace(old,new) for item in argv]
    return argv


def cpu_check(project,campaign,manifest):
    native.check(project,campaign,manifest)
    campaign=Path(campaign)
    tests=campaign/"sdk_local_transition"
    tests.mkdir(exist_ok=False)
    started=time.perf_counter()
    env=__import__("os").environ.copy()
    env["PYTHONPATH"]="%s:%s:%s:/core:%s"%(campaign/"code/tests",campaign/"code/src",campaign/"code/scripts",env.get("PYTHONPATH",""))
    result=subprocess.run([sys.executable,"-m","unittest","test_local_bowl_transition_v1","-v"],capture_output=True,text=True,env=env,timeout=90)
    (tests/"stdout.log").write_text(result.stdout)
    (tests/"stderr.log").write_text(result.stderr)
    native.save(tests/"RESULT.json",dict(returncode=result.returncode,seconds=time.perf_counter()-started,
        manifest_sha256=manifest,models=0,robot_actions=0,scope="Actual native SDK with synthetic target and protocol checks"))
    if result.returncode:
        raise RuntimeError(result.stderr)
    return native.cpu_composition(project,campaign,manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    for option in ("prepare","cpu-composition","execute"):
        group.add_argument("--"+option,action="store_true")
    parser.add_argument("--project",type=Path,required=True)
    parser.add_argument("--campaign",type=Path,required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--arm",choices=("native","transition","gated"))
    parser.add_argument("--draw-seed",type=int)
    parser.add_argument("--resource-slot",type=int,choices=range(4))
    parser.add_argument("--sampler-input",type=Path)
    parser.add_argument("--candidate-snapshot",type=Path)
    parser.add_argument("--allow-real-physical-calls",action="store_true")
    args=parser.parse_args()
    if args.prepare:
        if args.arm is None:parser.error("Preparation requires explicit arm")
        result=prepare(args.project,args.campaign,args.arm,args.draw_seed,args.resource_slot,args.sampler_input,args.candidate_snapshot)
    else:
        if not args.manifest_sha256:parser.error("Exact snapshot manifest required")
        native.launch_argv=launch
        if args.cpu_composition:
            result=cpu_check(args.project,args.campaign,args.manifest_sha256)
        else:
            if not args.allow_real_physical_calls:parser.error("Explicit developmental dispatch required")
            result=native.execute(args.project,args.campaign,args.manifest_sha256)
    print(json.dumps(result,indent=2))
    if result.get("status","").startswith(("FAILED","CLEANUP_FAILED")):
        raise SystemExit(1)


if __name__=="__main__":main()
