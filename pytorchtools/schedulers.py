import numpy as np

class AddWarmup():
    '''
        Wraps a standard scheduler in a warmup scheduler

        Parameters
        ----------
            scheduler: torch.nn.Module
                A pytorch model that produces predictions on the data
                and whose weights are updated via the optimizer

            starting_lr: float
                The initial value to start the learning rate ramp up from

            ending_lr: float
                The final learning rate value to achieve after rampup

            warmup_dur: int
                How many steps to go from starting_lr to ending_lr

        Attributes
        ----------
            lr_schedule: np.array
                the schedule for linear learning rate ramp up

        Methods
        -------
            _set_lr
                Changes the learning rate for the optimizer

        '''
    def __init__(
            self,
            scheduler,
            starting_lr: float = 0.0,
            ending_lr: float = 0.1,
            warmup_dur: int = 5,
        ):
        self.scheduler = scheduler
        self.warmup_dur = warmup_dur
        self.starting_lr = starting_lr
        self.ending_lr = ending_lr
        self.steps = 0

        self.lr_schedule = np.linspace(starting_lr, ending_lr, warmup_dur+1)
        self._set_lr()

    def step(self):
        if self.steps < self.warmup_dur:
            self.steps += 1
            self._set_lr()
        else:
            self.scheduler.step()

    def _set_lr(self):
        print()
        print('Setting LR: ', self.lr_schedule[self.steps])
        for g in self.scheduler.optimizer.param_groups:
            g['lr'] = self.lr_schedule[self.steps]

