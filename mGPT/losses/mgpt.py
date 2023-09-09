import torch
import torch.nn as nn
from .base import BaseLosses


class CommitLoss(nn.Module):
    """
    Useless Wrapper
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, commit, commit2, **kwargs):
        return commit


class GPTLosses(BaseLosses):
    
    def __init__(self, cfg, stage, num_joints, **kwargs):
        # Save parameters
        self.stage = stage
        recons_loss = cfg.LOSS.ABLATION.RECONS_LOSS

        # Define losses
        losses = []
        params = {}
        if stage == "vae":
            losses.append("recons_feature")
            params['recons_feature'] = cfg.LOSS.LAMBDA_FEATURE

            losses.append("recons_velocity")
            params['recons_velocity'] = cfg.LOSS.LAMBDA_VELOCITY

            losses.append("vq_commit")
            params['vq_commit'] = cfg.LOSS.LAMBDA_COMMIT
        elif stage in ["lm_pretrain", "lm_instruct"]:
            losses.append("gpt_loss")
            params['gpt_loss'] = cfg.LOSS.LAMBDA_CLS

        # Define loss functions & weights
        losses_func = {}
        for loss in losses:
            if loss.split('_')[0] == 'recons':
                if recons_loss == "l1":
                    losses_func[loss] = nn.L1Loss
                elif recons_loss == "l2":
                    losses_func[loss] = nn.MSELoss
                elif recons_loss == "l1_smooth":
                    losses_func[loss] = nn.SmoothL1Loss
            elif loss.split('_')[1] in [
                    'commit', 'loss', 'gpt', 'm2t2m', 't2m2t'
            ]:
                losses_func[loss] = CommitLoss
            elif loss.split('_')[1] in ['cls', 'lm']:
                losses_func[loss] = nn.CrossEntropyLoss
            else:
                raise NotImplementedError(f"Loss {loss} not implemented.")

        super().__init__(cfg, losses, params, losses_func, num_joints,
                         **kwargs)

    def update(self, rs_set):
        '''Update the losses'''
        total: float = 0.0

        if self.stage in ["vae"]:
            total += self._update_loss("recons_feature", rs_set['m_rst'],
                                       rs_set['m_ref'])
            # total += self._update_loss("recons_joints", rs_set['joints_rst'], rs_set['joints_ref'])
            nfeats = rs_set['m_rst'].shape[-1]
            if nfeats in [263, 135 + 263]:
                if nfeats == 135 + 263:
                    vel_start = 135 + 4
                elif nfeats == 263:
                    vel_start = 4
                total += self._update_loss(
                    "recons_velocity",
                    rs_set['m_rst'][..., vel_start:(self.num_joints - 1) * 3 +
                                    vel_start],
                    rs_set['m_ref'][..., vel_start:(self.num_joints - 1) * 3 +
                                    vel_start])
            else:
                if self._params['recons_velocity'] != 0.0:
                    raise NotImplementedError(
                        "Velocity not implemented for nfeats = {})".format(nfeats))
            total += self._update_loss("vq_commit", rs_set['loss_commit'],
                                       rs_set['loss_commit'])

        if self.stage in ["lm_pretrain", "lm_instruct"]:
            total += self._update_loss("gpt_loss", rs_set['outputs'].loss,
                                       rs_set['outputs'].loss)

        # Update the total loss
        self.total += total.detach()
        self.count += 1

        return total
