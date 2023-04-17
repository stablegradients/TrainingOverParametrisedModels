# TrainingOverParametrisedModels
pytorch implementation of the paper "Training Over-parameterised Models with Non-Decomposable Objectives" Harikrishan Narasimhan et al.

```python
import numpy as np
from losses import MinRecall

# prior  : a list of prior
# val_lr : validation learning rate
# device : str, "cuda:x" where x is device id
criterion = MinRecall(prior, val_lr, device='cuda:0')

for epoch in range(num_epochs):
  # generate your model metrics
  # CM :confusion_matrix consider a 10 class classification
  criterion.update(CM)
  for x, targets in dataloader:
    # N steps of SGD
    preds = model(x)
    loss = criterion(preds, targets, redution)
    loss.backward()
    model.zero_grad()
```

