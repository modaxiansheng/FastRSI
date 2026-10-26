from visual_kernel_frozen import held_gate, place_plan, rigid
from fastrsi.visual_recovery.gates import initial_pose_gate as _gate
from pathlib import Path
import os,json
def initial_pose_gate(pose,skill):
    source=json.loads(Path(__file__).with_name('STATIC_SOURCE_POSE.json').read_text())
    return _gate(pose,skill,os.environ['LIBEROPRO_VARIANT'],source['calibration'])
