"""Development-only prefix replay at reset, followed by unchanged native target."""
import importlib.util,json,os,hashlib
from pathlib import Path
from cost_trace import emit,span
from stage_trace import capture,png_bytes
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fastrsi.trial_prefix import execute_prefix,signature
spec=importlib.util.spec_from_file_location('frozen_target','/core/PhyAgentOS/runtime/targets/remote/libero/server.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
config=json.loads(Path(os.environ['FASTRSI_TRIAL_SPEC']).read_text())
root=Path(os.environ['FASTRSI_STAGE_TRACE']).parent
used=False
original_reset=module.LiberoRealRuntime.reset
original_action=module.LiberoRealRuntime.action_chunk

def reset(self,*args,**kw):
    global used
    with span('environment_reset',method='reset'):
        out=original_reset(self,*args,**kw)
        if used:raise ValueError('Trial prefix study requires a single root Session')
        used=True
        if config['mode']=='reuse' and not config.get('cache_loop',False):
            trace=Path(config['source_trace']);raw=trace.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=config['source_trace_sha256']:raise ValueError('Paid source trace changed')
            rows=list(map(json.loads,raw.splitlines()));boundary=config['boundary']
            expected=next(r for r in rows if r['step']==boundary)
            for camera in ('agentview_image','robot0_eye_in_hand_image'):
                if hashlib.sha256(png_bytes(out[camera])).hexdigest()!=rows[0]['images']['0000_'+camera+'.png']:raise ValueError('Different initial pixels')
            self._trial_replaying=True
            try:
                if config.get('keep_prefix_rendering',False):
                    from fastrsi.trial_prefix import validate_prefix
                    import time
                    tick=time.perf_counter()
                    for row in validate_prefix(rows,boundary)[1:]:
                        self.action_chunk({'actions':row['actions']})
                        if self.step_idx!=row['step'] or self.done:raise ValueError('Prefix changed termination')
                    receipt=dict(physical_steps_reexecuted=boundary,boundary=boundary,omitted_camera_step_frames=0,seconds=time.perf_counter()-tick)
                else:
                    receipt=execute_prefix(self,rows,boundary)
            finally:self._trial_replaying=False
            for camera in ('agentview_image','robot0_eye_in_hand_image'):
                if hashlib.sha256(png_bytes(self._last_obs[camera])).hexdigest()!=expected['images'][f'{boundary:04d}_{camera}.png']:raise ValueError('Restored boundary differs')
            receipt.update(source_trace_sha256=config['source_trace_sha256'],physics=signature(self.env.env.sim.get_state().flatten()),
                           source_prefix_is_inherited=True,scope='Same development instance only; unchanged prefix physics reexecuted')
            (root/'TRIAL_PREFIX_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
            out=self._last_obs
    record(self,'reset',args)
    return out

def record(self,method,args):
    emit(dict(event='native_observation',session_id=self.session_id,method=method,steps=self.step_idx,
              native_success=bool(self.env.check_success()),runtime_success=bool(self.success)))
    with span('stage_diagnostic_capture',session_id=self.session_id,step=self.step_idx):
        row=capture(self,method,args,os.environ['FASTRSI_STAGE_TRACE'])
    with (root/'TRIAL_PHYSICS.jsonl').open('a') as f:
        f.write(json.dumps(dict(step=self.step_idx,physics=signature(self.env.env.sim.get_state().flatten())))+'\n')

def action(self,*args,**kw):
    if getattr(self,'_trial_replaying',False):return original_action(self,*args,**kw)
    with span('environment_execution',session_id=self.session_id,method='action_chunk'):
        out=original_action(self,*args,**kw)
    record(self,'action_chunk',args)
    return out

module.LiberoRealRuntime.reset=reset
module.LiberoRealRuntime.action_chunk=action
module.main()
