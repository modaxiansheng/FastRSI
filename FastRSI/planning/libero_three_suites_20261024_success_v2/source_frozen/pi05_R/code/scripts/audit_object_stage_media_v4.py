"""CPU-only actual vendor encoding check, including a black-image control."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

NAME = 'object_stage_media_audit_v4_20261006'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def encode_check(project, out):
    started=time.perf_counter();out.mkdir(exist_ok=False)
    prior=project/'runs/object_stage_memory_v2_20261006'
    source=prior/'inputs/INPUTS.json'
    old=read(prior/'model_job/learning/RESULT.json')
    if sha(source)!=old['input_artifact_sha256']['INPUTS.json']:raise ValueError('Changed cached inputs')
    model=project/'models/clef-flash'/old['model_revision']
    manifest=read(model/'DOWNLOAD.json')
    for name in ('joint_schema_model.py','config.json','processor_config.json','tokenizer_config.json','tokenizer.json'):
        if sha(model/name)!=manifest['files'][name]['sha256']:raise ValueError('Changed processor asset')
    from transformers import AutoProcessor
    from PIL import Image
    from fastrsi.object_stage_memory_v1 import encode
    processor=AutoProcessor.from_pretrained(model,local_files_only=True)
    spec=importlib.util.spec_from_file_location('clef_media_audit',model/'joint_schema_model.py')
    vendor=importlib.util.module_from_spec(spec);sys.modules[spec.name]=vendor;spec.loader.exec_module(vendor)
    inputs=read(source);selected=[inputs[0],next(r for r in inputs if r['role']=='probe')]
    results=[]
    for row in selected:
        images=[]
        for ref in row['images']:
            path=project/ref['path']
            if sha(path)!=ref['sha256']:raise ValueError('Changed saved input image')
            with Image.open(path) as image:
                if hashlib.sha256(image.tobytes()).hexdigest()!=ref['pixel_sha256']:raise ValueError('Changed image pixels')
                images.append(image.copy())
        encoded=encode(vendor.encode_record,processor,{**row['request'],'images':images})
        batch=vendor.collate_records([encoded],processor.tokenizer.pad_token_id,'cpu')
        pixels=batch['media']['pixel_values']
        if tuple(batch['media']['image_grid_thw'].shape)!=(4,3):raise ValueError('Images lost during batching')
        original=hashlib.sha256(pixels.numpy().tobytes()).hexdigest()
        black=encode(vendor.encode_record,processor,{**row['request'],
            'images':[Image.new('RGB',image.size) for image in images]})
        control=hashlib.sha256(black.media['pixel_values'].numpy().tobytes()).hexdigest()
        if original==control:raise ValueError('Actual image tensors insensitive to replacing pixels')
        results.append(dict(id=row['id'],input_tokens=len(encoded.input_ids),
            image_grid_thw=batch['media']['image_grid_thw'].tolist(),pixel_shape=list(pixels.shape),
            original_pixel_tensor_sha256=original,black_pixel_tensor_sha256=control,
            pixels_changed=True,actual_batched_images=4))
    report=dict(status='PASSED',rows=results,seconds=time.perf_counter()-started,
        new_model_forward_calls=0,new_robot_episodes=0,gpu_cards_reserved=0,
        source_input_sha256=sha(source),vendor_source_sha256=sha(model/'joint_schema_model.py'),
        scope='Actual CPU encoding/batching only; image tensors differ under pixel replacement. '
              'This does not measure output sensitivity, model accuracy or policy benefits.')
    (out/'RESULT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


def host():
    import fcntl
    import traceback
    import orchestrate_full_os as life
    run=life.PROJECT/'runs'/NAME
    for name,h in read(run/'CODE_MANIFEST.json').items():
        if sha(run/'code'/name)!=h:raise ValueError('Frozen input audit source changed')
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    life.deadline=time.time()+160;out=run/'job';out.mkdir(exist_ok=False)
    state=dict(status='RUNNING',started_at=time.time(),new_model_forward_calls=0,new_robot_episodes=0,gpu_cards_reserved=0)
    try:
        life.gate(out)
        base='/project/runs/'+NAME+'/code'
        argv=['docker','run','-d','--name','yuchen-fastrsi-object-stage-media-v4','--network','none',
            '--cpus','4','--memory','8g','-e','HIP_VISIBLE_DEVICES=-1','-e','CUDA_VISIBLE_DEVICES=-1',
            '-e','HF_HUB_OFFLINE=1','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH='+base+'/src',
            '-v','/opt/hyhal:/opt/hyhal:ro','-v','/etc/hfm:/etc/hfm:ro',
            '-v',str(life.PROJECT)+':/project:ro','-v',str(out)+':/output:rw',life.IMAGE,
            'timeout','120','bash','-lc','source /opt/dtk/env.sh && source /opt/hyhal/env.sh && exec '
            '/project/envs/clef-decision-v1/bin/python '+base+'/scripts/audit_object_stage_media_v4.py '
            'encode --project /project --output /output/encoding']
        cid=life.launch('encode',argv,out);life.wait_exit(cid,130)
        if read(out/'encoding/RESULT.json')['status']!='PASSED':raise ValueError('Media audit failed')
        state['status']='COMPLETED'
    except BaseException:state.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        life.save(out/'STOP_RECEIPT.json',life.stop_all())
        state.update(seconds=time.time()-state['started_at'],finished_at=time.time());life.save(out/'STATUS.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('host','encode'))
    parser.add_argument('--project',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    host() if args.mode=='host' else encode_check(args.project,args.output)
