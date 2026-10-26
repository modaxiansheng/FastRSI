"""Action-conditioned pair ranker. Scores are NOT success probabilities."""
import hashlib
from pathlib import Path
import numpy as np


def difference(tokens):
    x = np.asarray(tokens, dtype=np.float64)
    assert x.shape == (2,10,1024) and np.isfinite(x).all()
    x = (x-x.mean(-1,keepdims=True))/np.maximum(x.std(-1,keepdims=True),1e-6)
    delta = (x[1]-x[0]).reshape(-1)
    norm = np.linalg.norm(delta)
    return delta/max(norm,1e-12)


class ExpertPairRanker:
    def __init__(self,path):
        self.sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as f:
            self.x = f['expert_training_differences'].copy()
            self.coefficients = f['expert_kernel_coefficients'].copy()
        assert self.x.shape[1] == 10240 and self.coefficients.shape == (len(self.x),)

    def choose(self,tokens):
        d = difference(tokens)
        score = float(d@self.x.T@self.coefficients)
        return int(score>0),dict(pair_preference_score=score,rule='fixed dual ridge sign; zero defaults',
            success_probability=False,training_lambda=1.,online_initial_id_used=False)

