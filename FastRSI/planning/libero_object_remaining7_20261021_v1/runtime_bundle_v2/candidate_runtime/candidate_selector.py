"""Frozen, outcome-blind selector: return an original candidate, never an average."""
import hashlib,math

def seed_for(identity,block,index):
    assert index in (1,2,3)
    return int.from_bytes(hashlib.sha256(f'candidate4-v1|{identity}|{block}|{index}'.encode()).digest()[:8],'big')%(2**63)

def select(candidates,scale):
    x=[c.tolist() if hasattr(c,'tolist') else c for c in candidates];s=list(scale)
    assert len(x)==4 and len(s)==7 and all(math.isfinite(v) and v>0 for v in s)
    assert all(len(c)==10 and all(len(r)==7 and all(math.isfinite(v) for v in r) for r in c) for c in x)
    distance=[[math.fsum(((x[i][t][k]-x[j][t][k])/s[k])**2 for t in range(10) for k in range(7))/70 for j in range(4)] for i in range(4)]
    scores=[math.fsum(r) for r in distance];index=min(range(4),key=lambda i:(scores[i],i))
    return index,dict(scores=scores,distance_matrix=distance)
