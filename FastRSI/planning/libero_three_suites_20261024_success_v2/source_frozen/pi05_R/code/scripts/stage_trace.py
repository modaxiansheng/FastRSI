"""Opt-in diagnostic copy of existing TargetWS images/actions; never controller input."""
import hashlib,json,struct,zlib
from pathlib import Path


def png_bytes(rgb):
    # Encode existing RGB pixels without any flip, resize, colour conversion or quantization.
    height=len(rgb);width=len(rgb[0]);assert height and width
    rows=[]
    fast=getattr(rgb,'dtype',None) is not None and str(rgb.dtype)=='uint8' and tuple(rgb.shape)==(height,width,3)
    for row in rgb:
        assert len(row)==width
        if fast:
            rows.append(b'\0'+row.tobytes(order='C'))
            continue
        pixels=[]
        for pixel in row:
            assert len(pixel)==3
            pixels.append(bytes(int(c) for c in pixel))
        rows.append(b'\0'+b''.join(pixels))
    def chunk(kind,data):return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data)&0xffffffff)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(b''.join(rows)))+chunk(b'IEND',b'')


def capture(runtime, method, args, root):
    folder=Path(root)/runtime.session_id;folder.mkdir(parents=True,exist_ok=True)
    row={'method':method,'step':runtime.step_idx,'observation_id':runtime._last_obs['observation_id'],
         'session_id':runtime.session_id,'images':{},'native_success':bool(runtime.success),
         'scope':'Diagnostic only; exact already-visible policy-oriented pixels, not new controller evidence.'}
    for name in ('agentview_image','robot0_eye_in_hand_image'):
        blob=png_bytes(runtime._last_obs[name]);name=f'{runtime.step_idx:04d}_{name}.png'
        target=folder/name
        with target.open('xb') as f:f.write(blob)
        row['images'][name]=hashlib.sha256(blob).hexdigest()
    if method=='action_chunk':
        action=args[0]['actions'];row['actions']=action.tolist() if hasattr(action,'tolist') else action
    with (folder/'TRACE.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    return row
