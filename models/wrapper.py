import torch
import torch.nn as nn
import timm
from torch.distributions.uniform import Uniform

class TimmModelWrapper(nn.Module):
    def __init__(self, model, mixup_factor=0.5):
        super(TimmModelWrapper, self).__init__()

        # Load the pre-trained Timm model
        self.model = model

        # mixup factor for mixed-up feed forward
        self.mixup_factor = mixup_factor

    def forward(self, input1, input2=None):
        if input2 is None:
            return self.model(input1)
        else:
            # Forward pass for the first input
            features1 = self.model.forward_features(input1)

            # Forward pass for the second input
            features2 = self.model.forward_features(input2)

            # mix the features
            batch_size = features2.shape[0] # type: ignore
            mixup_coeff = Uniform(self.mixup_factor,1).sample([batch_size]).cuda() # type: ignore
            feats = (features1.T * mixup_coeff).T + (features2.T * (1-mixup_coeff)).T 

            # Forward pass through the modified classifier
            output = self.model.forward_head(feats)
        return output
    def forward_features(self, x):
        return self.model.forward_features(x)
    def forward_head(self, x):
        return self.model.forward_head(x)
