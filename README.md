# SelMix
code for our paper selmix
## Introduction

## How it works and eq. 

## installation

## Usage

### Wrapping your timm model 
```python
import torch
import timm
from models.wrapper import TimmModelWrapper

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

```

## results

## citation