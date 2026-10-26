"""Build an opt-in candidate from a real native sampler development success.

Reads saved public execution records and bound sampling metadata. No environment,
model, hidden array/XML, or prospective outcome is queried. Compilation does not
certify transfer or approve the candidate for release.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
from fastrsi.experience_compiler import compile_live_success, canonical_sha
from fastrsi.native_goal_component_v2 import prepare_component


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False)


def build(campaign, registration, collection, selection, calibration, output):
    start=time.perf_counter()
    campaign,registration,collection,selection,calibration,output=map(Path,(campaign,registration,collection,selection,calibration,output))
    if output.exists():raise ValueError('Fresh successor source package required')
    report=json.loads((campaign/'stream/RESULT.json').read_text())
    if report.get('status')!='COMPLETED' or len(report.get('cases',[]))!=1:raise ValueError('Complete single root Agent report required')
    result=report['cases'][0];case=result['case'];sid=case['id']
    config=json.loads((campaign/'SAMPLER_CONFIG.json').read_text());sample=config['restore']['sample_id']
    if not sample.startswith('sample_'):raise ValueError('Explicit registered sample identity required')
    seed=int(sample[len('sample_'):])
    coverage=json.loads(selection.read_text())
    if coverage['registration_sha256']!=digest(registration) or coverage['collection_manifest_sha256']!=digest(collection/'MANIFEST.json') or seed not in coverage['selected_administrative_seeds']:
        raise ValueError('Source is not one of the preregistered coverage selections')
    selected=next(r for r in coverage['rows'] if r['administrative_seed']==seed)
    rois=selected['matching_diagnostic']['selected_boxes']
    sources=dict(result=campaign/'stream/RESULT.json',trace=campaign/f'stream/stage_trace/{sid}/TRACE.jsonl',
        live_controls=campaign/f'stream/live_controls/{sid}/LIVE_CONTROLS.json',
        controller_jsonl=campaign/f'stream/live_controls/{sid}/CONTROLLER_TARGETS.jsonl',
        target_events=campaign/'stream/TARGET_COST_EVENTS.jsonl',sampler_binding=campaign/'SAMPLER_CONFIG.json',
        sampler_registration=registration,sampler_manifest=collection/'MANIFEST.json',
        sampler_receipt=collection/sample/'COLLECTION_RECEIPT.json',sampler_observation=collection/sample/'public_observation/OBSERVATION.json')
    blobs={k:v.read_bytes() for k,v in sources.items()}
    trace=[json.loads(x) for x in blobs['trace'].splitlines()];live=json.loads(blobs['live_controls'])
    events=[json.loads(x) for x in blobs['target_events'].splitlines()]
    restores=[r for r in events if r.get('event')=='native_sampler_restore' and r.get('session_id')==sid]
    if len(restores)!=1:raise ValueError('Exactly one actual root restoration required')
    initialization=dict(restore_event=restores[0],sampler_binding=blobs['sampler_binding'],registration=blobs['sampler_registration'],
        collection_manifest=blobs['sampler_manifest'],sample_receipt=blobs['sampler_receipt'],public_observation=blobs['sampler_observation'])
    compiled=compile_live_success(result,trace,live,provenance=dict(result_sha256=digest(sources['result']),
        trace_sha256=digest(sources['trace']),controller_trace_sha256=digest(sources['live_controls'])),initialization_records=initialization)
    waypoints=compiled['waypoints'];signs=[w['gripper_command']>0 for w in waypoints]
    transitions=[i for i in range(1,len(signs)) if signs[i]!=signs[i-1]]
    if signs[0] or len(transitions)!=3 or [signs[i] for i in transitions]!=[True,False,True]:
        raise ValueError('Current two-anchor interface needs two complete commanded grasp phases; no truncation or invented annotations')
    close,release,second_close=transitions
    plan=[dict(anchor='first_object',start=0,close=close,release=release,stop=release+1,terminal_kind='observed_open_command'),
          dict(anchor='second_object',start=release+1,close=second_close,release=None,stop=len(signs),terminal_kind='source_native_goal_truncated')]
    output.mkdir(parents=True);package=output/'SOURCE_PACKAGE';package.mkdir()
    prefix='configs/source_successor_v1/'
    refs={}
    for key,blob in blobs.items():
        rel=prefix+key+('.jsonl' if key in ('trace','controller_jsonl','target_events') else '.json')
        path=package/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
        refs[key]=dict(path=rel,sha256=digest(path))
    artifact=package/(prefix+'COMPILED.json');save(artifact,compiled)
    image=package/(prefix+'SOURCE.png');shutil.copyfile(campaign/f'stream/stage_trace/{sid}/0000_agentview_image.png',image)
    camera=package/(prefix+'CALIBRATION.json');shutil.copyfile(calibration,camera)
    declaration=dict(version='native_goal_component_v2',source_artifact=dict(path=artifact.relative_to(package).as_posix(),sha256=digest(artifact)),source_records=refs,
        transform=dict(version='dual_anchor_v2',source_rgb=dict(path=image.relative_to(package).as_posix(),sha256=digest(image)),
            calibration=dict(path=camera.relative_to(package).as_posix(),sha256=digest(camera)),source_rois_xyxy=rois,anchor_plan=plan,max_offset_m=.06),
        bindings=dict(source_goal=case['task_description'],model_config_sha256=case['model_config_sha256'],target_ref='target://libero_real_remote',action_dim=7,robot_state_dim=8))
    qualified=prepare_component(package,declaration,expected_goal=case['task_description'],expected_model_config_sha256=case['model_config_sha256'],expected_target_ref='target://libero_real_remote')
    template=copy.deepcopy(declaration);template['bindings']['model_config_sha256']='$MODEL_CONFIG_SHA256'
    catalogue=package/(prefix+'CATALOGUE.json');save(catalogue,dict(version='native_goal_component_catalogue_v2',candidates={'successor_v1':template}))
    manifest=dict(version='native_goal_component_package_v2',catalogue_path=catalogue.relative_to(package).as_posix(),
        files={p.relative_to(package).as_posix():digest(p) for p in sorted(package.rglob('*')) if p.is_file()})
    save(package/'MANIFEST.json',manifest)
    receipt=dict(status='ACTUAL_ROOT_SUCCESS_COMPILED_UNVALIDATED_TRANSFER',source_campaign=str(campaign),source_session=sid,
        original_record_hashes={k:digest(v) for k,v in sources.items()},code_sha256=digest(__file__),selection_sha256=digest(selection),
        native_success=result['environment_success'],verification_status=result.get('verification_status'),chain_status=result['chain_status'],
        compiled_source_sha256=canonical_sha(compiled),source_completed_step=compiled['source_completed_step'],source_qualification=qualified['derivation_receipt']['source_qualification'],
        anchor_plan=plan,source_rois_xyxy=rois,annotation_scope='Public initial appearance/order boxes and actual commanded-gripper sign changes; not semantic identity or physical release certification',
        package_manifest_sha256=digest(package/'MANIFEST.json'),model_calls=0,resets=0,actions=0,learning_gain_claimed=False,
        seconds=time.perf_counter()-start,cost_scope='Through receipt construction; process startup and final receipt write excluded')
    save(output/'BUILD_RECEIPT.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','registration','collection','selection','calibration','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=build(a.campaign,a.registration,a.collection,a.selection,a.calibration,a.output)
    print(json.dumps({k:r[k] for k in ('status','source_completed_step','compiled_source_sha256','seconds')}))
if __name__=='__main__':main()
