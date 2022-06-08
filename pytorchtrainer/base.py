import os
import sys
import time

import torch

from typing import List, Optional, Iterable

class BaseTrainer:
    '''

    '''
    def __init__(
        self,
        net: torch.nn.Module = None,

        # data / dataloaders
        train_loader: Iterable = None,
        valid_loader: Optional[Iterable] = None,

        # loss functions
        crit: List[torch.nn.Module] = None,
        crit_lambdas: Optional[List[float]] = None,

        # metrics
        metrics: Optional[List] = None,
        metric_names: Optional[List] = None,

        # weight updates
        epochs: int = 100,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        mixed_precision: bool = False,

        # accelerator
        device: torch.device = None,

        # model saving
        checkpoint_every: int = 100,
        checkpoint_dir: Optional[str] = './',
        model_name: Optional[str] = 'model'
    ):
        '''

        '''
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.net = net
        self.crit = crit if type(crit) is list else [crit]
        if crit_lambdas is None:
            self.crit_lambdas = [1. for _ in range(len(self.crit))]
        else:
            self.crit_lambdas = crit_lambdas

        self.metrics = metrics

        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled = mixed_precision)

        self.epochs = epochs
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name + '.pth'

        # initialize update step counter
        self.iterations = 0

        # loss and metrics tracking
        self.losses = {'train': [], 'valid': []}
        self.metric_names = metric_names
        if self.metric_names is None:
            self.metric_names = [m.__name__.capitalize()  for m in metrics]
        if metrics is not None:
            self.metric_tracking = {mn: {'train': [], 'valid': []} for mn in self.metric_names}


    def train_network(self):
        '''
        Train the network using the train and valid functions.

        Record the losses and saves model weights periodically

        Allows keyboard interrupt to stop training at any time
        '''
        try:
            print()
            print('-----------------------------------------')
            print('Training ...')
            print('-----------------------------------------')
            print()

            for e in range(self.epochs):

                print('\n' + 'Iter {}/{}'.format(e + 1, self.epochs))
                start = time.time()
                self.run_epoch(mode = 'train')

                if self.valid_loader is not None:
                    with torch.no_grad():
                        self.run_epoch(mode = 'valid')

                if self.scheduler is not None:
                    self.scheduler.step()

                print('Time: {}'.format(time.time() - start))

                if e % self.checkpoint_every == 0:
                    model_path = os.path.join(self.checkpoint_dir, self.model_name)
                    print(f'Saving model to {model_path}')
                    torch.save(self.net.state_dict(), model_path)

        except KeyboardInterrupt:
            pass

    def run_epoch(self, mode: str):
        '''
        Uses the data loader to grab a batch of images

        Pushes images through network and gathers predictions

        Updates network weights by evaluating the loss functions
        '''

        running_loss = 0.
        if self.metrics is not None:
            running_metrics = [0. for _ in range(len(self.metrics))]

        if mode == 'train':
            # zero gradients
            self.net.zero_grad(set_to_none = True)
            self.net.train(True)

            iterator = enumerate(self.train_loader)
            n_batches = len(self.train_loader)
        else:
            self.net.eval()
            iterator = enumerate(self.valid_loader)
            n_batches = len(self.valid_loader)

        for i, (inputs, targets) in iterator:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                # get predictions
                preds = self.net(inputs)

                loss = self.get_loss(preds, targets)
                running_loss += loss.item()

            if mode == 'train':

                self.scaler.scale(loss).backward()

                # update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # zero gradients for next run
                self.net.zero_grad(set_to_none=True)
                # count grad updates
                self.iterations += 1

            # track metrics
            if self.metrics is not None:
                metric_vals = [m(preds, targets) for m in self.metrics]
                for m_idx, mv in enumerate(metric_vals):
                    self.metric_tracking[self.metric_names[m_idx]][mode].append(mv)
                    running_metrics[m_idx] += mv

            # display running statistics
            sys.stdout.write('\r')
            sys.stdout.write('{} B: {:>3}/{:<3} | Loss: {:.4}'.format(
                mode.capitalize(),
                i+1,
                n_batches,
                loss.item(),
            ))
            if self.metrics is not None:
                for m_idx, mn in enumerate(self.metric_names):
                    sys.stdout.write(' {}: {:.4}'.format(mn, metric_vals[m_idx]))
            sys.stdout.flush()

        # display batch statistics
        print(
            '\n' + 'Avg Loss: {:.4}'.format(
                running_loss / n_batches,
            )
        )

        if self.metrics is not None:
            for m_idx, rm in enumerate(running_metrics):
                print('Avg {}: {:.4}'.format(
                    self.metric_names[m_idx], rm / n_batches
                ))

        self.losses[mode].append(running_loss / n_batches)