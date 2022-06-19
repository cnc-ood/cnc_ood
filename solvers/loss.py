import torch.nn as nn
import torch

class CrossEntropyWithSoftTargets(nn.Module):
    def __init__(self, dim=-1, **kwargs):
        super(CrossEntropyWithSoftTargets, self).__init__()
        self.dim = dim
        self.cls = torch.nn.CrossEntropyLoss()

    def forward(self, pred, target):
        # import pdb; pdb.set_trace()
        if pred.shape == target.shape:
            pred = pred.log_softmax(dim=self.dim)
            return torch.mean(torch.sum(-target * pred, dim=self.dim))
        else:
            # return the normal cross-entropy
            assert pred.shape[0] == target.shape[0]
            return self.cls(pred, target)