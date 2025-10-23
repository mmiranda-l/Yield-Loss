import torch.nn as nn
import torch

class PhysicsLoss:
    def __init__(self, seq_weights=None, loss_type="mse", reduction="mean"):
        super(PhysicsLoss, self).__init__()
        self.seq_weights = seq_weights
        self.reduction = reduction
        if self.seq_weights is not None:
            self.seq_weights /= self.seq_weights.mean()
        if loss_type == "mse": 
            self.core_call = self.core_call_mse
        else: raise Exception("Physics loss not implemented")

    def __call__(self, prediction, target):

        return self.core_call(prediction, target)

    def core_call_mse(self, prediction, target, l=1.0):

        upper_bound = torch.where(prediction > target, 
                            (prediction - target) ** 2, 
                            torch.zeros_like(prediction))
        
        within_bounds = torch.where((prediction >= 0) & (prediction <= target), 
                                    (prediction - target) ** 2, 
                                    torch.zeros_like(prediction))
        
        lower_bound = torch.where(prediction < 0, prediction ** 2, torch.zeros_like(prediction))

        out = upper_bound + within_bounds + lower_bound        
        if self.seq_weights is not None:
            out *= self.seq_weights
        return l * out.mean()

class WeightedMSE:
    def __init__(self, ignored_target_value=-1, seq_weights=None):
        self.ignored_target_value = ignored_target_value
        self.seq_weights = seq_weights
        if self.seq_weights is not None:
            self.seq_weights /= self.seq_weights.mean()

        self.core_call = self.core_call_ignore

    def __call__(self, prediction, target):
        out = self.core_call(prediction, target) 
        return 100 * out

    def core_call_ignore(self, prediction, target):
        mask = target == self.ignored_target_value
        out = (prediction - target) ** 2
        if self.seq_weights is not None:
            out *= self.seq_weights
        return out[~mask].mean()
