# PyTorch Trainer

A module for handling the training logic for PyTorch models. The Trainer classes act as the [information expert](https://en.wikipedia.org/wiki/GRASP_(object-oriented_design)#Information_expert) for the training data and metrics for running model training. Essentially all information needed to fullfill training and generated via training will be handled by these classes. This helps to reduce repeated code and produce easily traceable ML code.

The trainers are meant to be used with base PyTorch. PyTorch already has simple and flexible methods for implementing neural networks, data loaders, and various training components. What's missing is a unified API for running training for common tasks.

```python
from pytorchtrainer.trainers import SingleOutputTrainer

net = torchvision.models.resnet18()

# set up our trainer
trainer = SingleOutputTrainer(
    net = net,
    train_loader = train_loader,
    valid_loader = valid_loader,
    crit = nn.CrossEntropyLoss(),
    device = torch.device('cuda'),
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001),
    epochs = 100,
)

# train the network
trainer.train_network()
```