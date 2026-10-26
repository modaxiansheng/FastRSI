"""Require step-for-step equivalence before reporting runtime timing differences."""
import argparse
import json
from pathlib import Path
import statistics


def step_content(row):
    return {k:v for k,v in row.items() if k!='seconds'}


def require_equivalent(left,right):
    if left['source_sha256']!=right['source_sha256']:
        raise ValueError('Different action source')
    if left['status']!='EXACT_BOUNDARY_REPLAY' or right['status']!='EXACT_BOUNDARY_REPLAY':
        raise ValueError('Incomplete historical boundary verification')
    if [step_content(s) for s in left['steps']] != [step_content(s) for s in right['steps']]:
        raise ValueError('Physics, observation, reward or done differs at an action step')
    for key in ('native_steps','success','boundaries'):
        if left[key]!=right[key]:raise ValueError('Different terminal/boundary result: '+key)


def analyze(run):
    protocol=json.loads((run/'PROTOCOL.json').read_text())
    status=json.loads((run/'STATUS.json').read_text())
    variable=protocol.get('variable','renderer')
    setting_key='blas_threads' if variable=='blas' else 'threads'
    result_key='openblas_num_threads' if variable=='blas' else 'lp_num_threads'
    if status['status']!='COMPLETED':raise ValueError('Incomplete replay campaign')
    groups={};detail={}
    for job in protocol['jobs']:
        result=json.loads((run/job['id']/'RESULT.json').read_text())
        if result[result_key] != (None if job[setting_key] is None else str(job[setting_key])):
            raise ValueError('Renderer setting differs from protocol')
        launch=json.loads((run/job['id']/'LAUNCH_replay.json').read_text())['argv']
        if launch[launch.index('--cpus')+1]!='8':raise ValueError('CPU quota changed')
        groups.setdefault(job['source'],[]).append((job,result))
        detail[job['id']]={k:result.get(k) for k in ('seconds','timings','process_threads','threadpools','cpu_count','gl_renderer','gl_version','cpu_stat_before','cpu_stat_after')}
    comparisons=[]
    for source,rows in groups.items():
        baseline=next(r for j,r in rows if j[setting_key] is None)
        for _,r in rows:require_equivalent(baseline,r)
        defaults=[r for j,r in rows if j[setting_key] is None]
        ones=[r for j,r in rows if j[setting_key]==1]
        metrics={}
        for metric in ('seconds','action_chunk','env_step_sum','cold_environment_create','reset'):
            get=lambda r:r['seconds'] if metric=='seconds' else r['timings'][metric]
            a=statistics.mean(map(get,defaults));b=statistics.mean(map(get,ones))
            metrics[metric]=dict(default_mean=a,one_mean=b,difference=a-b,reduction_fraction=(a-b)/a)
        comparisons.append(dict(source=source,default_repeats=len(defaults),one_repeats=len(ones),
            native_steps=baseline['native_steps'],checked_steps_including_settling=len(baseline['steps']),
            boundaries=len(baseline['boundaries']),metrics=metrics,all_steps_exact=True))
    return dict(status='EXACT_STEP_EQUIVALENCE',variable=variable,comparisons=comparisons,jobs=detail,
        batch_wall_seconds=status['seconds'],new_model_calls=0,new_policy_episodes=0,
        learning_gain_established=False,full_os_speedup_measured=False,
        scope='Common CPU runtime experiment on saved action trajectories; no skipped images, actions or verification.')


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=analyze(a.run)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='jobs'},indent=2))


if __name__=='__main__':main()
