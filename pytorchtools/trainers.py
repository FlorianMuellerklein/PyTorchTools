from torch import Tensor

from pytorchtrainer.base import BaseTrainer

class SingleOutputTrainer(BaseTrainer):
    '''
    Single task trainer for PyTorch models. Applieds n loss functions to a model
    that produces a single output which corresponds to a single target.
    '''
    def __init__(self, *args, **kwargs):
        super(SingleOutputTrainer, self).__init__(*args, **kwargs)

    def get_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        # calculate loss
        loss = 0.
        for crit_idx, crit in enumerate(self.crit):
            loss += self.crit_lambdas[crit_idx] * crit(preds, targets)

        return loss

class MultiOutputTrainer(BaseTrainer):
    '''
    Multi output trainer for PyTorch models. Applies n loss functions to a
    model that produces n outputs which correspond to n targets.
    '''
    def __init__(self, *args, **kwargs):
        super(MultiOutputTrainer, self).__init__(*args, **kwargs)

    def get_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        # calculate loss
        loss = 0.
        for crit_idx, crit in enumerate(self.crit):
            loss += self.crit_lambdas[crit_idx] * crit(preds[crit_idx], targets[crit_idx])

        return loss