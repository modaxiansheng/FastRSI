"""Portable small completion/cost head; scores and member ranges are uncalibrated."""
import numpy as np
from phase_iql_value_v1 import Predictor

class QualityCostPredictor(Predictor):
    def outputs(self,inputs):
        values=[]
        for member in range(2):
            h=np.tanh(inputs@self.a[f'h{member}_w0'].T+self.a[f'h{member}_b0'])
            y=h@self.a[f'h{member}_w1'].T+self.a[f'h{member}_b1']
            values.append(1/(1+np.exp(-np.clip(y,-40,40))))
        return np.stack(values)

    def score(self,feature,robot,candidates,budget):
        state=self.state(feature,robot,budget)
        action=(np.asarray(candidates,np.float32).reshape(-1,70)-self.a['action_mean'])/self.a['action_std']
        outputs=self.outputs(np.concatenate([np.repeat(state[None],len(action),axis=0),action],axis=1))
        p=outputs[:,:,0];cost=outputs[:,:,1];selected=0;reason='Default or unresolved'
        assert len(action) in (1,2)
        if len(action)==2:
            if p[:,1].min()>p[:,0].max():selected=1;reason='Both-member completion advantage'
            elif np.max(np.abs(p[:,1]-p[:,0]))<=.02 and cost[:,1].max()<cost[:,0].min():
                selected=1;reason='Similar completion scores and both-member action-cost advantage'
        return selected,dict(completion_scores=p.tolist(),conditional_action_cost_scores=cost.tolist(),
            decision_reason=reason,score_tolerance=.02,calibrated_probability=False,
            member_ranges_are_not_certificates=True,cost_is_remaining_actions_proxy=True)
