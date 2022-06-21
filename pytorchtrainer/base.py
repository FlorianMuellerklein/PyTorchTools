import os
import sys
import time

import torch
from torch.utils.data import Dataset, DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from typing import List, Optional, Iterable

class BaseTrainer:
    '''
    Base class for PyTorch training
    '''
    def __init__(
        self,
        net: torch.nn.Module = None,

        # data / dataloaders
        train_dataset: Optional[Dataset] = None,
        valid_dataset: Optional[Dataset] = None,

        train_loader: Optional[DataLoader] = None,
        valid_loader: Optional[DataLoader] = None,

        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,

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
        device_ids: List[int] = None,
        ddp: bool = False,

        # artifact saving
        checkpoint_every: int = 100,
        checkpoint_dir: Optional[str] = './',
        model_name: Optional[str] = 'model',

        tb_writer = None,
    ):
        '''
        Initializes the trainer.

        Parameters
        ----------
            net: torch.nn.Module
                A pytorch model that produces predictions on the data
                and whose weights are updated via the optimizer

            train_loader: iterable
                An iteratble that returns batches of (inputs, targets)

            valid_loader: iterable
                An iteratble that returns batches of (inputs, targets)

            crit: torch.nn.Module or List[torch.nn.Module]
                A single or list of loss functions to apply to the output of the network

            crit_lambdas: float or List[float]
                A list of loss function scaling lambdas to apply to each loss

            metrics: function or List[function]
                A list of optional metrics to use to track learning progress outside of the loss function.
                Must accept inputs in the form of metric(y_true, y_score), following sklearn convention for compatibility

            metric_names: str or List[str]
                An optional list of metrics names to be used during tracking and saving. If not supplied
                will use the function names for each metrics

            epochs: int
                How many epochs to use for training

            optimizer: torch.optim.Optimizer
                The torch optimizer function, can be any supplied by PyTorch

            scheduler: torch.optim.lr_scheduler
                Optional PyTorch learning rate scheduler

            mixed_precision: bool
                Whether to use mixed precision training or not

            device_ids: List[int]
                List of device ids to use for training

            ddp: bool
                Whether to use distributed data parallel training

            checkpoint_every: int
                How often to save model weights

            checkpoint_dir: str
                Directory to save model weights

            model_name: str
                The name of the model for saving weights

            tb_writer: SummaryWriter
                An optional tensorboard writer for logging

        Attributes
        ----------
            iterations: int
                tracks the total number of gradient updates

            losses: dict
                tracks the loss values for each batch

            metric_tracking: dict
                trackes the metric values for each batch
        '''
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.net = net
        self.crit = crit if type(crit) is list else [crit]
        if crit_lambdas is None:
            self.crit_lambdas = [1. for _ in range(len(self.crit))]
        else:
            self.crit_lambdas = crit_lambdas
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.metrics = metrics

        self.ddp = ddp
        self.device_ids = device_ids
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
        if self.metric_names is None and metrics is not None:
            self.metric_names = [m.__name__.capitalize()  for m in metrics]
        if metrics is not None:
            self.metric_tracking = {mn: {'train': [], 'valid': []} for mn in self.metric_names}

        self.tb_writer = tb_writer

        if ddp:
            # initialize distributed data parallel modules
            print('DDP setup ...')
            # set up the processing group
            self.init_ddp()
            self.world_size = len(self.device_ids)
        else:
            self.device = torch.device('cuda:{}'.format(device_ids[0]) if torch.cuda.is_available() else 'cpu')

    def fit(self):
        if self.ddp:
            # first input to this is automatically rank or device
            mp.spawn(self.train_network, args=(self.world_size,), nprocs = self.world_size, join=True)
        else:
            self.train_network()

    def train_network(self, device = None, world_size = None):
        '''
        Train the network using the train and valid functions.

        Record the losses and saves model weights periodically

        Allows keyboard interrupt to stop training at any time
        '''
        if self.ddp:
            dist.init_process_group("gloo", rank = device, world_size = world_size)
            net, optimizer = self.setup_ddp_modules(device, self.net)
        else:
            device = self.device
            net = self.net.to(device)
            optimizer = self.optimizer

        try:
            print()
            print('-----------------------------------------')
            print('Training ...')
            print('-----------------------------------------')
            print()

            for e in range(self.epochs):

                if (self.ddp and device == 0) or (not self.ddp):
                    print('\n' + 'Iter {}/{}'.format(e + 1, self.epochs))

                start = time.time()
                self.run_epoch(net, optimizer, device, mode = 'train')

                if self.valid_loader is not None:
                    with torch.no_grad():
                        self.run_epoch(net, optimizer, device, mode = 'valid')

                if self.scheduler is not None:
                    self.scheduler.step()

                if (self.ddp and device == 0) or (not self.ddp):
                    print('Time: {}'.format(time.time() - start))

                if e % self.checkpoint_every == 0:
                    model_path = os.path.join(self.checkpoint_dir, self.model_name)
                    print(f'Saving model to {model_path}')
                    torch.save(self.net.state_dict(), model_path)

        except KeyboardInterrupt:
            pass

        if self.ddp:
            dist.destroy_process_group()

    def run_epoch(self, net, optimizer, device, mode: str):
        '''
        Uses the data loader to grab a batch of data

        Pushes data through network and gathers predictions

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
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                # get predictions
                preds = net(inputs)

                loss = self.get_loss(preds, targets)
                running_loss += loss.item()

            if mode == 'train':

                self.scaler.scale(loss).backward()

                # update weights
                self.scaler.step(optimizer)
                self.scaler.update()
                # zero gradients for next run
                self.net.zero_grad(set_to_none=True)
                # count grad updates
                self.iterations += 1

            # track metrics
            if self.metrics is not None:

                metric_vals = [m(targets, preds).item() for m in self.metrics]
                for m_idx, mv in enumerate(metric_vals):
                    self.metric_tracking[self.metric_names[m_idx]][mode].append(mv)
                    running_metrics[m_idx] += mv

            if self.tb_writer is not None:
                self.writer.add_scalar(f'Loss/{mode}', loss.item(), self.iterations)

                for metric_name in self.metric_tracking:
                    self.writer.add_scalaer(f'{metric_name}/{mode}', self.metric_tracking[metric_name][mode][-1], self.iterations)

            if (self.ddp and device == 0) or (not self.ddp):
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

        if (self.ddp and device == 0) or (not self.ddp):
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

    def init_ddp(self):
        '''
        Standard PyTorch setup for distributed dataparallel

        Returns
        -------
            None
        '''
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

    def prep_ddp_loaders(self, dataset: Iterable, rank: int, world_size: int):
        '''
        Distributes a dataloader across world size.

        Returns
        -------
            loader:
                The dataloader provided to the function with the distributed dataset and sampler
        '''

        sampler = DistributedSampler(
            dataset,
            rank = rank,
            num_replicas = world_size,
            shuffle = True,
            drop_last = True
        )

        loader = DataLoader(
            dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            sampler = sampler,
            drop_last = True,
        )

        return loader

    def setup_ddp_modules(self, rank, net):

        print('Splitting data loader ...')
        # split the dataloader
        self.train_loader = self.prep_ddp_loaders(self.train_dataset, rank, self.world_size)
        self.valid_loader = self.prep_ddp_loaders(self.valid_dataset, rank, self.world_size)

        print('Distributing model ...')
        # wrap the model
        net = DDP(net.to(rank), device_ids = [rank], output_device = rank)

        # wrap the optimizer
        optimizer = self.optimizer
        optimizer.parameters = self.net.parameters

        return net, optimizer


    def get_loss(self, inputs: torch.Tensor, targets: torch.Tensor):
        '''
        Abstract class for calculating the loss. This part can be customized for custom loaders. This
        follows PyTorch convention in accepting (input, target) rather than (target, input) like sklearn

        Returns
        -------
            torch.Tensor of loss value
        '''
        pass