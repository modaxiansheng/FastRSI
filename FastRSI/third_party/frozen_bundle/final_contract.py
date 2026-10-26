def validate_verdict(v):
 if not isinstance(v,dict) or v.get('verdict') not in ('success','failure','replan'):
  raise ValueError('invalid verdict')
 if not isinstance(v.get('evidence'),list) or not v['evidence'] or not all(isinstance(x,str) and x.strip() for x in v['evidence']):
  raise ValueError('missing visual evidence')
 if not isinstance(v.get('lesson'),str) or not v['lesson'].strip():raise ValueError('missing lesson')
 if v['verdict'] in ('failure','replan') and not v.get('failure_reason'):raise ValueError('missing failure reason')
 if v['verdict']=='replan' and not v.get('replan_task_description'):raise ValueError('missing replan instruction')
 return v

def final_record(attempts,verdicts):
 error=not attempts or any(x.get('error_code') for x in attempts)
 return {'first_success':None if not attempts or attempts[0].get('error_code') else bool(attempts[0]['success']),
         'final_success':None if error else bool(attempts[-1]['success']),
         'execution_status':'runtime_error' if error else 'completed',
         'recovery_attempts':max(0,len(attempts)-1),'success_authority':'libero_native',
         'verifier_calls':len(verdicts)}

def configure_recovery_horizon(env,first_steps,recovery_steps,max_replans,wait_steps):
 # Keep attempt budgets in the runtime; this only prevents the underlying
 # simulator lifetime limit from truncating a valid multi-attempt episode.
 required=int(wait_steps)+int(first_steps)+int(recovery_steps)*int(max_replans)+1
 env.horizon=max(int(env.horizon),required)
 return env.horizon
