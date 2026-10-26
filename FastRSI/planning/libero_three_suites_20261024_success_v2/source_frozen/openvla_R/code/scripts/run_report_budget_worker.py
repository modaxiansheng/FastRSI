"""One isolated full-OS development worker for the report-prefetch budget study."""
import hashlib,json,shutil,socket,sys,time,traceback,signal
import orchestrate_full_os as life
from native_service_config import configure_qwen,model_env

def main():
    name,arm=sys.argv[1:];assert arm in ('N','F')
    label=name.replace('/','-')
    campaign=life.PROJECT/'runs'/name;group=campaign/arm
    life.CAMPAIGN=group;life.BASE='/project/runs/'+name+'/'+arm
    protocol=json.loads((campaign/'PROTOCOL.json').read_text());life.deadline=protocol['absolute_deadline']
    plan=json.loads((campaign/'PLAN.json').read_text());cfg=json.loads((campaign/'MODEL_CONFIG.json').read_text())
    initial=campaign/'initial_experience';flowsha=hashlib.sha256((campaign/'FLOW.json').read_bytes()).hexdigest()
    slot=protocol['slot'];shift=slot*1000+(100 if arm=='F' else 0)
    def port(p):return p+shift
    def relocate(argv,arm):
        if not shift:return argv
        out=[]
        for v in argv:
            for p in (49500,49502,49510,49520):v=v.replace(str(p),str(p+shift))
            for prefix in ('HIP_VISIBLE_DEVICES=','CUDA_VISIBLE_DEVICES='):
                if v==prefix+'0':v=prefix+str(slot*4+(2 if arm=='F' else 0))
                elif v==prefix+'1':v=prefix+str(slot*4+(2 if arm=='F' else 0)+1)
            out.append(v)
        return out
    def interrupt(n,f):raise KeyboardInterrupt('Interrupted worker '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    group=campaign/arm;stream=group/'stream';stream.mkdir(parents=True,exist_ok=False);gstart=time.time();base=str(stream).replace(str(life.PROJECT),'/project')
    shutil.copyfile(campaign/'MODEL_CONFIG.json',group/'MODEL_CONFIG.json')
    cases=[]
    for item in plan['cases']:
        case={**item,'id':f"net_t{item['task_id']}_i{item['init_state_id']}",'session_intent':True,
              'require_discovery_reads':True,'require_clean_model_calls':True,'read_files_tool':True}
        case['terminal_prefetch']=(arm=='F')
        case.update(session_flow_artifact='/project/runs/'+name+'/FLOW.json',session_flow_sha256=flowsha)
        cases.append(case)
    life.save(group/'ROSTER.json',cases);g={'arm':arm,'status':'RUNNING','started_at':gstart};life.save(group/'STATUS.json',g)
    try:
        shutil.copytree(life.PROJECT/'models/openpi_data',stream/'tokenizer_cache')
        for role,receipt in [('pi05','LAUNCH_pi05_v2.json'),('qwen','LAUNCH_qwen.json'),('target','LAUNCH_target.json')]:
            argv=json.loads((life.OLD/receipt).read_text())['argv'];argv[argv.index('--name')+1]='yuchen-fastrsi-'+label+'-'+arm.lower()+'-'+role
            argv=[x.replace(str(life.OLD),str(stream)).replace('/project/runs/pilot_20261001',base) for x in argv]
            if role=='pi05':
                argv=[x.replace(str(life.PROJECT/'models/openpi_data'),str(stream/'tokenizer_cache')) for x in argv];assert '--torch-seed 7' in argv[-1]
            if role=='qwen':argv=configure_qwen(argv,cfg,life)
            if role=='target':argv=[x.replace('/project/scripts/instrument_target.py',life.BASE+'/code/scripts/instrument_target.py').replace(':/project/scripts',':'+life.BASE+'/code/scripts') for x in argv]
            argv=relocate(argv,arm);life.launch(role,argv,stream)
        life.health(port(49510));life.health(port(49500))
        with socket.create_connection(('127.0.0.1',port(49502)),timeout=5):pass
        g['startup_seconds']=time.time()-gstart
        source=str(initial).replace(str(life.PROJECT),'/project')
        remaining=int(life.deadline-time.time()-30);assert remaining>60
        argv=life.cpu_args('yuchen-fastrsi-'+label+'-'+arm.lower()+'-runner',stream)+model_env(cfg)+['-e','FASTRSI_NATIVE_RUN='+base,'-e','FASTRSI_CONSUME_TERMINAL=1','-e','FASTRSI_EXPERIENCE_ADMISSION=guarded_v1','-e','FASTRSI_INITIAL_EXPERIENCE='+source,
             life.IMAGE,'timeout',str(remaining),'bash','-lc','source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/run_native_baseline.py']
        cid=life.launch('runner',argv,stream);life.wait_exit(cid,remaining+5)
        report=json.loads((stream/'RESULT.json').read_text())
        assert report['status']=='COMPLETED' and len(report['cases'])==1
        assert sum(len(row['outcomes']) for row in report['cases'])<=3
        for row in report['cases']:
            assert row['chain_status']=='PASSED'
            if row['case']['mode']=='development':assert 'experience_admission' in row
            assert row['terminal_report']['read_coverage']['complete']
            if row['case']['mode']=='evaluation':assert row['experience_before']==row['experience_after']
        g.update(status='COMPLETED',controller_seconds=report['seconds'],
                 outcomes=[{'case':row['case']['id'],'environment_success':row['environment_success'],
                            'verdicts':[o.get('verdict') for o in row['outcomes']]} for row in report['cases']])
    except BaseException:g.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        life.save(stream/'STOP_RECEIPT.json',life.stop_all());g.update(finished_at=time.time(),full_wall_seconds=time.time()-gstart);life.save(group/'STATUS.json',g);print(json.dumps(g),flush=True)

if __name__=='__main__':main()
