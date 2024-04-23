## Usage Guidelines with Custom Models

### Using non-timm models
For custom models such as wide_resnet_28_2, we recommend adding additional functions that would allow you to extract backbone features for inputs to allow performing muixups.

```python
    
class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, bn_momentum=0.1, leaky_slope=0.0, dropRate=0.0):
        super(WideResNet, self).__init__()
        # declare the layers of the model, same as that defined before

    def forward_head(self, x, pre_logits=False):
        # feeds the backbone features to the linear classifier head
        # similar to timm, which allows you to add a linear layer or directly extract
        # backbone features
        return x if pre_logits else self.fc(x)

    def forward_features(self, x):
        '''
            Args:
                x: input image
            Return:
                out, the backbone features 
        '''
        return out

    def classifier(self, x):
        # feeds the backbone features to the linear classifier head
        return self.fc(x)
    
    def get_classifier(self):
        # returns the linear layer of the model
        return self.fc
```


### Wrapping your timm-like model 
```python
import torch
import timm
from models.wrapper import TimmModelWrapper
from MetricOptimisation import MinRecall
from dataloaders import FastJointSampler

# define your model and load the pret-trained checkpoint
model = timm.create_model('resnet32', pretrained=False)
model.load_state_dict(torch.load('/path/to/checkpoint'))
model.cuda()

# wrap your model into our timm model wrapper 
# to allow selective mixup finetuning
mixup_factor = 0.6 # to sample from uniform randomly from [0.6, 1]
model = TimmModelWrapper(model=model, mixup_factor=mixup_factor)

# sample batch of inputs
B, C, H, W = 128, 3, 224, 224
x1 = torch.randn(torch.randn(B, C, H, W))
x2 = torch.randn(torch.randn(B, C, H, W))

# a simple feedforward to get logits after mixup of features
# λ ~ U[mixup_factor, 1]
# if logits for x is  classifier(features(x)) then their mixup logits are
# classifier(λ features(x1) + (1-λ) features(x2))

mixup_logits = model(x1, x2)

# inference 
y_ = model(x1)

# extract features 
features = model.forward_features(x)

# classify the extracted features
logits = model.forward_head(features)
```
## Other timm model functions
```python
wrapper_model = TimmModelWrapper(timm_model, mixup_factor)

# the original timm model is a member of the wrapper class
# hence the member functions func() can still be accesed as
func_return = wrapper_model.model.func(args)

```
