import torch
import torch.nn as nn

class BaseLosses(nn.Module):
    def __init__(self, cfg, losses, params, losses_func, num_joints, **kwargs):
        super().__init__()
        
        # Save parameters
        self.num_joints = num_joints
        self._params = params
        
        # Add total indicator
        losses.append("total") if "total" not in losses else None
        
        # Register losses
        for loss in losses:
            self.register_buffer(loss, torch.tensor(0.0))
        self.register_buffer("count", torch.tensor(0.0))
        self.losses = losses
        
        # Instantiate loss functions
        self._losses_func = {}
        for loss in losses[:-1]:
            self._losses_func[loss] = losses_func[loss](reduction='mean')
            
    def _update_loss(self, loss: str, outputs, inputs):
        '''Update the loss and return the weighted loss.'''
        # Update the loss
        val = self._losses_func[loss](outputs, inputs)
        # self.losses_values[loss] += val.detach()
        getattr(self, loss).add_(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        return weighted_loss
            
    def reset(self):
        '''Reset the losses to 0.'''
        for loss in self.losses:
            setattr(self, loss, torch.tensor(0.0, device=getattr(self, loss).device))
        setattr(self, "count", torch.tensor(0.0, device=getattr(self, "count").device))

    def compute(self, split):
        '''Compute the losses and return a dictionary with the losses.'''
        count = self.count
        # Loss dictionary
        loss_dict = {loss: getattr(self, loss)/count for loss in self.losses}
        # Format the losses for logging
        log_dict = { self.loss2logname(loss, split): value.item() 
            for loss, value in loss_dict.items() if not torch.isnan(value)}
        # Reset the losses
        self.reset()
        return log_dict

    def loss2logname(self, loss: str, split: str):
        '''Convert the loss name to a log name.'''
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name
