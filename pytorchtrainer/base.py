import os
import sys
import time

import torch

from typing import List, Optional, Iterable

class BaseTrainer:
    '''
    Base class for PyTorch training
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

        # artifact saving
        checkpoint_every: int = 100,
        checkpoint_dir: Optional[str] = './',
        model_name: Optional[str] = 'model',

        tb_writer: torch.utils.tensorboard.SummaryWriter = None,
    ):
        '''
        Initializes the trainer.

        Arguments:
            net: A torch.nn.Module that produces predictions on the images
                and whose weights are updated via the optimizer

            train_loader: An iteratble that returns batches of (inputs, targets)

            valid_loader: An iteratble that returns batches of (inputs, targets)

            crit: A single or list of loss functions to apply to the output of the network

            crit_lambdas: A list of loss function scaling lambdas to apply to each loss

            metrics: A list of optional metrics to use to track learning progress outside of the loss function.
                Must accept inputs in the form of metric(y_true, y_score), following sklearn convention for compatibility

            metric_names: An optional list of metrics names to be used during tracking and saving. If not supplied
                will use the function names for each metrics

            epochs: How many epochs to use for training

            optimizer: The torch optimizer function, can be any supplied by PyTorch

            scheduler: Optional PyTorch learning rate scheduler

            mixed_precision: Whether to use mixed precision training or not

            device: Whether to use any supported accelerator, for example a GPU


            checkpoint_every: How often to save model weights

            checkpoint_dir: Directory to save model weights

            model_name: The name of the model for saving weights

            tb_writer: An optional tensorboard writer for logging

        Attributes:
            iterations: tracks the total number of gradient updates

            losses: tracks the loss values for each batch

            metric_tracking: trackes the metric values for each batch
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
            print(self.metric_tracking)

        self.tb_writer = tb_writer


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

                metric_vals = [m(targets, preds) for m in self.metrics]
                for m_idx, mv in enumerate(metric_vals):
                    self.metric_tracking[self.metric_names[m_idx]][mode].append(mv)
                    running_metrics[m_idx] += mv

            if self.tb_writer is not None:
                self.writer.add_scalar(f'Loss/{mode}', loss.item(), self.iterations)

                for metric_name in self.metric_tracking:
                    self.writer.add_scalaer(f'{metric_name}/{mode}', self.metric_tracking[metric_name][mode][-1], self.iterations)

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

    def get_loss(self, inputs, targets):
        '''
        Abstract class for calculating the loss. This part can be customized for custom loaders. This
        follows PyTorch convention in accepting (input, target) rather than (target, input) like sklearn
        '''
        pass