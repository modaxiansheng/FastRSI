"""Project saved pre-action inputs without promoting outcomes to policy labels."""
import hashlib
import json
import math
import struct
import zlib


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def project_input(event, execution_rule):
    state=event.get('robot_state');bindings=event.get('array_bindings',{})
    missing=[]
    if not isinstance(state,list) or len(state)!=8 or any(type(v) not in (int,float) or not math.isfinite(v) for v in state):
        missing.append('robot_state')
    for key in ('observation/image','observation/wrist_image','observation/state'):
        b=bindings.get(key,{})
        if not isinstance(b.get('sha256'),str) or len(b['sha256'])!=64:missing.append(key+'_binding')
    if not event.get('prompt'):missing.append('goal')
    if missing:return None,missing
    # Whitelist, not a copy: outcome, initial ID, chosen future action, and model
    # explanations never become input even if a source dictionary includes them.
    return dict(goal=event['prompt'],robot_state=list(state),current_execution_rule=execution_rule,
                array_bindings={k:bindings[k] for k in ('observation/image','observation/wrist_image','observation/state')}),[]


def png_pixels(blob):
    """Decode the existing stage_trace RGB PNG writer; reject unsupported encoding."""
    if blob[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Not PNG')
    pos=8;packed=[];shape=None
    while pos<len(blob):
        n=struct.unpack('!I',blob[pos:pos+4])[0];kind=blob[pos+4:pos+8];data=blob[pos+8:pos+8+n]
        if len(data)!=n:raise ValueError('Truncated PNG')
        crc=struct.unpack('!I',blob[pos+8+n:pos+12+n])[0]
        if zlib.crc32(kind+data)&0xffffffff != crc:raise ValueError('PNG CRC mismatch')
        if kind==b'IHDR':
            width,height,bits,color,compression,filtering,interlace=struct.unpack('!2I5B',data)
            if (bits,color,compression,filtering,interlace)!=(8,2,0,0,0):raise ValueError('Unsupported PNG')
            shape=[height,width,3]
        elif kind==b'IDAT':packed.append(data)
        elif kind==b'IEND':break
        pos+=n+12
    if shape is None:raise ValueError('Missing PNG shape')
    raw=zlib.decompress(b''.join(packed));stride=shape[1]*3+1
    if len(raw)!=shape[0]*stride or any(raw[y*stride]!=0 for y in range(shape[0])):
        raise ValueError('Not lossless stage_trace filter-zero format')
    return b''.join(raw[y*stride+1:(y+1)*stride] for y in range(shape[0])),shape


def checked_image(blob, expected_file_sha, binding):
    if hashlib.sha256(blob).hexdigest()!=expected_file_sha:raise ValueError('Image bytes changed')
    pixels,shape=png_pixels(blob)
    if shape!=binding['shape'] or binding['dtype']!='uint8' or hashlib.sha256(pixels).hexdigest()!=binding['sha256']:
        raise ValueError('Image does not match actual policy pixels')
    return hashlib.sha256(pixels).hexdigest()


def observed_relation(left, right):
    """Paired episode outcomes, never the correctness of an intermediate decision."""
    if type(left) is not bool or type(right) is not bool:raise ValueError('Boolean outcomes required')
    if left==right:return 'both_success' if left else 'both_failure'
    return 'left_success_only' if left else 'right_success_only'


def grouped_folds(records):
    """All initial states, repeated runs and adjacent frames of a task stay together."""
    tasks=sorted({r['task_group'] for r in records})
    return [dict(development_task=task, training_groups=[g for g in tasks if g!=task],
                 validation_groups=[task],scope='retrospective grouped development, not unseen final test') for task in tasks]


def training_examples(inputs, decisions):
    """Fail closed on unqualified policy targets; a teacher opinion is insufficient."""
    indexed={r['record_id']:r for r in inputs};result=[]
    for row in decisions:
        if row.get('label_status')!='VALIDATED_DECISION':continue
        if row.get('label') not in ('continue','inspect','escalate'):raise ValueError('Unknown decision label')
        if row.get('evidence_kind') not in ('executed_comparison','verified_invalidation') or not row.get('evidence_sources'):
            raise ValueError('Decision lacks outcome or invalidation evidence')
        source=indexed[row['record_id']]
        if source['model_input'] is None:raise ValueError('Incomplete decision input')
        result.append(dict(model_input=source['model_input'],label=row['label']))
    return result
