"""Position-only endpoint skip around the unchanged matched recurrent predictor."""
import torch
from torch.nn import functional as F

from merit.matched import MatchedPredictor
from merit.inputs import position_to_state,preprocess,cv_filter_history


class EndpointResidual(MatchedPredictor):
    report_age=False
    filter_cv=None

    def __init__(self,row,mean,scale,enabled,*,filter_cv=None,report_age=False):
        if row not in ('direct','projection','gate_analytic') or type(enabled) is not bool:
            raise ValueError('Unregistered endpoint-residual setting')
        super().__init__(row,mean,scale)
        self.endpoint_residual=enabled
        if (filter_cv is not None or report_age) and row!='direct':
            raise ValueError('Input baselines require the direct row')
        self.report_age=report_age
        self.filter_cv=filter_cv
        if filter_cv is not None:
            # The KF is linear with shared covariance. Build its causal operator
            # once in float64, then apply it to each batch on the input device.
            basis=torch.eye(60,dtype=torch.float64).T[:,:,None].expand(-1,-1,2)
            operator=cv_filter_history(basis,*filter_cv)[:,:,0].T
            self.register_buffer('filter_operator',operator.float(),persistent=False)
        if report_age:
            old=self.core.lstm
            self.core.lstm=torch.nn.LSTM(7,old.hidden_size,num_layers=old.num_layers,
                batch_first=True,dropout=old.dropout)

    def forward(self,observed,clean=None):
        if observed.ndim!=3 or observed.shape[1:]!=(60,7 if self.report_age else 6):
            raise ValueError('Expected 60-step history with six states and optional report age')
        mean,scale=self.core._state_mean,self.core._state_scale
        position=observed[:,:,:2]*scale[:2]+mean[:2]
        if self.filter_cv is not None:
            position=self.filter_operator.to(position) @ position
        local=position_to_state(position,mean,scale);aux=observed.new_zeros(())
        if self.row=='gate_analytic':
            pos,projected,blend=self.core.refinement_components(local)
            refined=self.core.apply_refinement_components(local,pos,projected,blend)
            if clean is not None:
                aux=F.mse_loss(refined[:,:,:2],clean[:,:,:2])
                target=self.core.optimal_blend_target(pos,projected,clean[:,:,:2]).detach()
                aux=aux+F.smooth_l1_loss(blend,target)
        else:
            method='prefix' if self.row=='direct' else self.row
            refined=position_to_state(preprocess(position,method),mean,scale)
        recurrent_input=torch.cat((refined,observed[:,:,6:7]),dim=-1) if self.report_age else refined
        encoded,_=self.core.lstm(recurrent_input)
        prediction=self.core.direct_proj(encoded[:,-1]).reshape(-1,30,6)
        if self.endpoint_residual:
            prediction=torch.cat([prediction[:,:,:2]+refined[:,-1:,:2],prediction[:,:,2:]],dim=-1)
        return prediction,aux
