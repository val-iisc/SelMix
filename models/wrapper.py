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
        self.classifier = self.model.get_classifier()

    def forward(self, input1, input2=None):
        if input2 is None:
            return self.model(input1)
        else:
            # Forward pass for the first input
            features1 = self.pre_logits(input1)

            # Forward pass for the second input
            features2 = self.pre_logits(input2)

            # mix the features
            batch_size = features2.shape[0] # type: ignore
            mixup_coeff = Uniform(self.mixup_factor,1).sample([batch_size]).cuda() # type: ignore
            feats = (features1.T * mixup_coeff).T + (features2.T * (1-mixup_coeff)).T 

            # Forward pass through the modified classifier
            output = self.classify_prelogits(feats)
        return output
    def pre_logits(self, x):
        return self.model.forward_head(self.model.forward_features(x), pre_logits=True)
    def classify_prelogits(self, x):
        return self.classifier(x)