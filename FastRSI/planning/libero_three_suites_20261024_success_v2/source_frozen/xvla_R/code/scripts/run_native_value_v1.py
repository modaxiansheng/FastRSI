"""Run the original full Agent/verification loop with fixed value experience."""
import asyncio
import hashlib
import json
import os
import shutil
from pathlib import Path
import run_native_baseline as baseline
from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter
from fastrsi.native_value_binding_v1 import install
from cost_trace import emit

CONFIG=Path(os.environ['FASTRSI_NATIVE_VALUE_CONFIG'])
binding=json.loads(CONFIG.read_text())
old_setup=baseline.setup_workspace
old_case=baseline.run_case


def setup(folder,case,accepted):
    w,spec,previous=old_setup(folder,case,accepted)
    assert case['suite']=='libero_90' and case['task_id']==64 and case['max_steps']==400
    dest=w/'policies/external_value_v1';dest.mkdir(parents=True,exist_ok=True)
    source=Path(binding['model_path']);sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha==binding['model_sha256']
    target=dest/'model.npz'
    if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==sha
    else:shutil.copyfile(source,target)
    value=dict(version='native_external_value_v1',goal=case['task_description'],max_steps=400,
        extra_candidate_cap=20,model_relative='policies/external_value_v1/model.npz',model_sha256=sha,
        enabled=binding['enabled'],noise_base=binding['noise_bases'][str(case['init_state_id'])],
        qualification='DEVELOPMENT_CANDIDATE_NOT_PRODUCTION',source_feedback=binding['source_feedback'])
    (dest/'CURRENT.json').write_text(json.dumps(value,indent=2)+'\n')
    # Installed before bind_agent's fixed-experience snapshot and evaluation
    # freeze. No labels, parameter changes or registry update during evaluation.
    (folder/'VALUE_INPUT_INSTALL.json').write_text(json.dumps(dict(
        config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),installed_before_agent=True,
        source_feedback=binding['source_feedback'],model_sha256=sha),indent=2)+'\n')
    return w,spec,previous


async def run_case(case,accepted):
    # setup_workspace is called inside the original run_case. Install only
    # after that has created this task's own workspace.
    restore=None
    def with_install(folder,case,accepted):
        nonlocal restore
        result=setup(folder,case,accepted)
        restore=install(OpenPIPi05Adapter,result[0],emit)
        return result
    baseline.setup_workspace=with_install
    try:return await old_case(case,accepted)
    finally:
        if restore is not None:restore()
        baseline.setup_workspace=old_setup


baseline.run_case=run_case
if __name__=='__main__':asyncio.run(baseline.main())
