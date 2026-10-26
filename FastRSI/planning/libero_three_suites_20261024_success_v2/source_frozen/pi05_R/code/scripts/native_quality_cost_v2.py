"""Shared score interpretation repair, not a learned capability increment."""
import numpy as np
from native_quality_cost_v1 import QualityCostPredictor

class QualityCostPredictorV2(QualityCostPredictor):
    def score(self,feature,robot,candidates,budget):
        state=self.state(feature,robot,budget)
        action=(np.asarray(candidates,np.float32).reshape(-1,70)-self.a['action_mean'])/self.a['action_std']
        outputs=self.outputs(np.concatenate([np.repeat(state[None],len(action),axis=0),action],axis=1))
        assert outputs.shape==(2,len(action),2) and np.isfinite(outputs).all()
        assert len(action) in (1,2)
        p=outputs[:,:,0];cost=outputs[:,:,1];selected=0;reason='Default or unresolved'
        if len(action)==2:
            # Tiny saturated score differences belong to the cost comparison,
            # rather than winning lexicographically over a slower candidate.
            if p[:,1].min()-p[:,0].max()>.02:
                selected=1;reason='Both-member completion advantage above shared score tolerance'
            elif p.min()>=.5 and np.max(np.abs(p[:,1]-p[:,0]))<=.02 and cost[:,1].max()<cost[:,0].min():
                selected=1;reason='Similar completion scores and both-member action-cost advantage'
        return selected,dict(completion_scores=p.tolist(),conditional_action_cost_scores=cost.tolist(),
            decision_reason=reason,score_tolerance=.02,cost_comparison_completion_score_floor=.5,
            calibrated_probability=False,member_ranges_are_not_certificates=True,
            cost_is_remaining_actions_proxy=True,public_protocol_correction_not_learning=True)
