"""Honor request temperature; retain checkpoint top-p/top-k for sampling."""
import math


def generation_options(body):
    temperature=float(body.get('temperature',0))
    if not math.isfinite(temperature) or not 0<=temperature<=2:
        raise ValueError('Temperature must be finite and between 0 and 2')
    if temperature==0:
        return {'do_sample':False}, None
    seed=body.get('seed',0)
    if not isinstance(seed,int) or isinstance(seed,bool) or not 0<=seed<2**63:
        raise ValueError('Seed must be an integer in [0, 2**63)')
    return {'do_sample':True,'temperature':temperature},seed


def memory_repetition_processor(body,prompt_length,penalty=1.0):
    """Common memory-only anti-loop control, excluding source prompt tokens."""
    penalty=float(penalty)
    if not math.isfinite(penalty) or not 1.0<=penalty<=1.2:
        raise ValueError('Memory repetition penalty must be between 1 and 1.2')
    names={t.get('function',{}).get('name') for t in body.get('tools',[]) or []}
    if names!={'save_memory'} or penalty==1.0:return None
    from transformers import RepetitionPenaltyLogitsProcessor
    return RepetitionPenaltyLogitsProcessor(penalty=penalty,prompt_ignore_length=prompt_length)
