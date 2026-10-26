"""Adopt a tested common repair without overwriting the original campaign."""
import fcntl,json,shutil,time
from pathlib import Path
import orchestrate_full_os as life
from orchestrate_representative_batch import digest,load,tree

def main():
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    master=life.PROJECT/'runs/representative_native_v1_20261002'
    recovery=life.PROJECT/'runs/report_path_recovery_v1_20261002'
    state=load(master/'STATE.json');protocol=load(master/'PROTOCOL.json')
    assert state['status']=='RECOVERY_COMPLETE_NEEDS_PROTOCOL_FIX'
    assert state['roots_reserved']==3 and state['streams']['0']['offset']==3
    assert load(recovery/'STATUS.json')['status']=='COMPLETED'
    result=load(recovery/'stream/RESULT.json')
    assert result['source_unchanged'] and result['recovered_case']['terminal_report']['status']=='PASSED'
    assert load(master/'batch_00_s0/STATUS.json')['status']=='FAILED'
    assert tree(master/'code')==load(master/'CODE_MANIFEST.json')
    assert time.time()<protocol['campaign_absolute_deadline']
    repairs=['src/fastrsi/terminal_results.py','tests/test_terminal_results.py']
    for rel in repairs:assert digest(life.PROJECT/rel)==digest(recovery/'code'/rel), 'Untested repair'
    rev=master/'protocol_revisions/report_path_v2';rev.mkdir(parents=True,exist_ok=False)
    life.save(rev/'STATE_BEFORE.json',state)
    shutil.copytree(master/'code',rev/'code')
    changes={}
    for rel in repairs+['scripts/orchestrate_representative_batch.py']:
        target=rev/'code'/rel;before=digest(target)
        shutil.copyfile(life.PROJECT/rel,target)
        changes[rel]={'before':before,'after':digest(target)}
    life.save(rev/'CODE_MANIFEST.json',tree(rev/'code'))
    life.save(rev/'BINDING.json',{
      'created_at':time.time(),'revision':'report_path_v2','changes':changes,
      'original_protocol_sha256':digest(master/'PROTOCOL.json'),
      'code_manifest_sha256':digest(rev/'CODE_MANIFEST.json'),
      'verified_by':str(recovery),'verification_status_sha256':digest(recovery/'STATUS.json'),
      'reason':'Explicit allowed report destination and actionable wrong-path rejection; common repair for every arm.',
      'preserved':['Original failed batch and all costs','Native AgentLoop and memory triggers','Original roster, task goals and model configuration','Global start/deadline and root reservations'],
      'comparison':'Do not pool original and repaired timing as a FastRSI effect; no learned method enabled.',
      'next_batch':'Continue frozen stream 0 from offset 3, two roots; no replacement for failed root.'})
    state.update(status='READY',active_protocol_revision='report_path_v2')
    state['elapsed_master_wall_seconds']=time.time()-state['started_at']
    life.save(master/'STATE.json',state)
    print(json.dumps({'status':state['status'],'revision':str(rev),'deadline':protocol['campaign_absolute_deadline']}))

if __name__=='__main__':main()
