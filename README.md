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
## Training loop
Overview of the training pipeling

```python 
## your datasets for mixup, for supervised case they are the same dataset
dataset1 = None
dataset2 = None 

optimizer = None

lagrange_multipliers = None


for epochs in range(num_epochs):
    confusion_matrix, prototypes = validation(valset, model)
    objective = MinRecall(confusion_matrix, prototypes, lagrange_multipliers)
    lagrange_multipliers = objective.lambdas
    
    P_selmix = objective.P
    SelMix_dataloader = FastJointSampler(dataset1, dataset2, model, P_selmix)

    for steps in range(num_steps_per_epoch):
        (x1, y1), (x2, y2) = SelMix_dataloader.get_batch()
        logits = model(x1, x2)
        loss = F.cross_entropy(logits, y1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

```
## results

## citation