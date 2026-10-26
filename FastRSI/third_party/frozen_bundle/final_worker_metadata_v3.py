"""Full evaluation: real policies, public RGB verifier, live native recovery.

No historical outcomes are reused. No verifier label can override native success.
Action traces and simulator evidence are persisted; offline snapshot restoration
and the paired Continue branch are separate gates, not claimed by this smoke.
"""
import argparse,asyncio,base64,hashlib,importlib.util,io,json,pathlib,pickle,socket,time,traceback
from urllib import request
import numpy as np
from PIL import Image
from resume_cases import completed_case, verify_async
from final_contract import final_record,configure_recovery_horizon
from verdict_metadata import validate_verdict

def put(path,value):
 temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False));temp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def health(plan):
 deadline=time.time()+1200
 port=int(plan['policy_endpoint'].rsplit(':',1)[1])
 while time.time()<deadline:
  try:
   with socket.create_connection(('127.0.0.1',port),timeout=2):pass
   with request.urlopen('http://127.0.0.1:38710/healthz',timeout=3) as response:
    if response.status==200:return
  except (OSError,ValueError):pass
  time.sleep(5)
 raise TimeoutError('policy/verifier readiness timeout')

def verify(plan,episode,folder):
 prompt=('You are a visual verifier for a robot manipulation task. Judge the task using only the supplied initial and final camera images. '
 'The four images are ordered: initial external, initial wrist, final external, final wrist. '
 'Return ONLY a JSON object with keys verdict, evidence, failure_reason, replan_task_description, lesson. '
 'verdict must be success, replan, or failure. If visibly complete use success. If incomplete and further robot actions could help, use replan. '
 'If recovery is not feasible use failure. evidence must be a nonempty list of specific visual observations; lesson must be nonempty. '
 'For replan/failure provide failure_reason; for replan provide a concrete nonempty replan_task_description. '
 'Do not claim success merely because the robot is near an object. Task: '+episode['task_description'])
 content=[{'type':'text','text':prompt}]
 public={'task':episode['task_description'],'images':[]}
 for phase in ('initial','final'):
  obs=episode['evidence'][phase+'_observation']
  for camera in ('agentview_image','robot0_eye_in_hand_image'):
   path=folder/(phase+'_'+camera+'.png');Image.fromarray(obs[camera]).save(path)
   public['images'].append({'path':path.name,'sha256':sha(path)})
   content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()}})
 put(folder/'VERIFIER_PUBLIC_INPUT.json',public)
 body={'model':'qwen25vl7b','messages':[{'role':'user','content':content}],'max_tokens':384,'temperature':0}
 req=request.Request(plan['verifier_endpoint'],data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 started=time.time()
 with request.urlopen(req,timeout=600) as response:reply=json.load(response)
 put(folder/'VERIFIER_RAW.json',reply)
 text=reply['choices'][0]['message']['content'];left=text.find('{');right=text.rfind('}')
 verdict=validate_verdict(json.loads(text[left:right+1]));verdict['elapsed_seconds']=time.time()-started
 put(folder/'VERDICT.json',verdict);return verdict

async def run(args):
 plan=json.loads(args.plan.read_text());args.output.mkdir(parents=True,exist_ok=True)
 if plan.get('first_steps_override') is not None and plan.get('diagnostic_only') is not True:
  raise ValueError('short First budget only allowed in explicitly diagnostic plans')
 put(args.output/'STATUS.json',{'stage':'waiting_for_services'})
 health(plan)
 spec=importlib.util.spec_from_file_location('paper_runtime','/bundle/server.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 class RecordingRuntime(m.LiberoRealRuntime):
  folder=None
  reset_count=0
  def reset(self,ctx):
   self.reset_count+=1
   out=super().reset(ctx)
   # Recovery can exceed the simulator's default 1000-step lifetime.
   full_budget=max(t['max_steps'] for t in plan['tasks'])
   horizon=configure_recovery_horizon(self.env.env,plan.get('first_steps_override',full_budget),full_budget,plan['max_replans'],10)
   if self.folder:
    put(self.folder/'LIFETIME_BUDGET.json',{'simulator_horizon':horizon,'max_replans':plan['max_replans']})
    np.save(self.folder/'initial_sim_state.npy',self.env.get_sim_state())
    (self.folder/'model.xml').write_text(self.env.sim.model.get_xml())
   return out
  def action_chunk(self,chunk):
   before=self.step_idx
   result=super().action_chunk(chunk)
   executed=self.step_idx-before
   with (self.folder/'ACTIONS.jsonl').open('a') as f:
    f.write(json.dumps({'first_step':before+1,'actions':np.asarray(chunk['actions'])[:executed].tolist()})+'\n');f.flush()
   return result
 runtime=RecordingRuntime();records=[]
 put(args.output/'SOURCE_BINDING.json',{'worker_sha256':sha(pathlib.Path(__file__)),'contract_sha256':sha(pathlib.Path('/bundle/final_contract.py')),
      'runtime_sha256':sha(pathlib.Path('/bundle/server.py')),'plan_sha256':sha(args.plan)})
 try:
  async with m._PolicyWsClient(plan['policy_endpoint'],timeout_s=180) as policy:
   put(args.output/'POLICY_METADATA.json',policy.metadata)
   expected_backend={'pi0':'openpi_native','pi05':'openpi_native','xvla':'xvla_lerobot','openvla':'openvla'}[plan['model']]
   if policy.metadata.get('backend')!=expected_backend:raise ValueError('wrong policy backend '+str(policy.metadata))
   for task in plan['tasks']:
    case='%s-t%d-i%d'%(task['suite'],task['task_id'],task['init_state_id']);case_dir=args.output/case
    existing=completed_case(case_dir,task,plan['model'])
    if existing is not None:
     records.append(existing);continue
    case_dir.mkdir(exist_ok=False)
    put(case_dir/'PLAN.json',plan);attempts=[];verdicts=[];case_started=time.time();reset_before=runtime.reset_count
    config={'seed':plan['seed'],'camera_height':256,'camera_width':256,'num_steps_wait':10,'replan_every_steps':5,'session_id':'%s-%s-g%d-%s'%(plan['run_id'],plan['job_id'],plan['gpu'],case)}
    try:
     for index in range(plan['max_replans']+1):
      folder=case_dir/('attempt%d'%index);folder.mkdir();runtime.folder=folder
      start_state=np.asarray(runtime.env.get_sim_state()).copy() if index else None
      put(args.output/'STATUS.json',{'stage':'first' if index==0 else 'recovery','case':case,'attempt':index,'time':time.time()})
      run_task=dict(task)
      if index==0 and plan.get('first_steps_override') is not None:run_task['max_steps']=plan['first_steps_override']
      if index==0 and plan.get('first_replay_source'):
       source=pathlib.Path(plan['first_replay_source'])
       original=m.unpackb((source/'EPISODE.msgpack').read_bytes())
       assert not original.get('error_code')
       assert all(original[k]==task[k] for k in ('suite','task_id','init_state_id'))
       runtime.reset({'session_id':config['session_id'],'libero':{**config,'benchmark_name':task['suite'],'task_id':task['task_id'],
          'init_state_id':task['init_state_id'],'max_steps':task['max_steps'],'control_mode':plan['control_mode']}})
       np.testing.assert_allclose(runtime.env.get_sim_state(),np.load(source/'initial_sim_state.npy'),rtol=0,atol=1e-10)
       for line in (source/'ACTIONS.jsonl').read_text().splitlines():
        chunk=json.loads(line);runtime.action_chunk({'actions':np.asarray(chunk['actions'],dtype=np.float32)})
       np.testing.assert_allclose(runtime.env.get_sim_state(),np.load(source/'terminal_sim_state.npy'),rtol=0,atol=1e-10)
       assert runtime.step_idx==original['num_steps'] and bool(runtime.env.check_success())==bool(original['success'])
       episode=original;episode['evidence']={'initial_observation':runtime._initial_obs,'final_observation':runtime._last_obs}
       put(folder/'REPLAY_VALIDATION.json',{'source':str(source),'source_episode_sha256':sha(source/'EPISODE.msgpack'),
          'initial_and_final_state_match_atol':1e-10,'native_success_matches':True,'steps':runtime.step_idx})
      else:
       episode=await runtime._run_benchmark_episode(policy,**run_task,control_mode=plan['control_mode'],attempt_index=index,evidence_mode='all',config=config,reset_environment=index==0)
      if index:np.save(folder/'recovery_start_state.npy',start_state)
      np.save(folder/'terminal_sim_state.npy',runtime.env.get_sim_state())
      (folder/'numpy_rng.pkl').write_bytes(pickle.dumps(np.random.get_state()))
      native=bool(runtime.env.check_success())
      episode['runtime_success']=episode['success'];episode['success']=native
      (folder/'EPISODE.msgpack').write_bytes(m.packb(episode))
      summary={k:v for k,v in episode.items() if k!='evidence'};put(folder/'EPISODE.json',summary);attempts.append(summary)
      if runtime.reset_count!=reset_before+1:raise AssertionError('environment reset during recovery')
      if episode.get('error_code'):raise RuntimeError(str(episode['error_code'])+': '+str(episode.get('error_message')))
      if native or index==plan['max_replans']:break
      put(args.output/'STATUS.json',{'stage':'verifying','case':case,'attempt':index,'time':time.time()})
      verdict=await verify_async(verify,plan,episode,folder);verdicts.append(verdict)
      if verdict['verdict']!='replan':break
      # Pinned original-instruction recovery. Preserve the live environment;
      # the policy session resets per the existing runtime protocol.
     row={**final_record(attempts,verdicts),'task':task,'model':plan['model'],'elapsed_seconds':time.time()-case_started,
          'attempts':attempts,'verdicts':verdicts,'environment_resets':runtime.reset_count-reset_before,
          'paper_configuration_verified':False,'scope':plan['scope'],'diagnostic_only':plan.get('diagnostic_only',False),
          'first_step_budget':plan.get('first_steps_override',task['max_steps'])}
     put(case_dir/'RESULT.json',row);records.append(row)
     print(json.dumps({k:v for k,v in row.items() if k not in ('attempts','verdicts')}),flush=True)
    except BaseException as exc:
     put(case_dir/'ERROR.json',{'error':str(exc),'traceback':traceback.format_exc(),'attempts':attempts})
     raise
  put(args.output/'COMPLETE.json',{'cases':len(records),'first_successes':sum(r['first_success'] is True for r in records),
       'final_successes':sum(r['final_success'] is True for r in records),'recovery_cases':sum(r['recovery_attempts']>0 for r in records),
       'recovery_attempts':sum(r['recovery_attempts'] for r in records),'status':'COMPLETE_FULL_SHARD',
       'scope':plan['scope'],'all_planned_cases_completed':len(records)==len(plan['tasks'])})
  put(args.output/'STATUS.json',{'stage':'COMPLETE','time':time.time()})
 finally:
  if runtime.env is not None:runtime.env.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
 try:asyncio.run(run(args))
 except BaseException as exc:
  args.output.mkdir(parents=True,exist_ok=True);put(args.output/'STATUS.json',{'stage':'FAILED','error':str(exc),'traceback':traceback.format_exc()});raise
