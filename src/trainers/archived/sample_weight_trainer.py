"""
Trainer for training sample weight prediction model jointly with main prediction model.
"""
import math
import time

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch import nn

from transformers.trainer_pt_utils import nested_concat, nested_detach, nested_numpify, nested_truncate, \
    get_parameter_names
from transformers.optimization import AdamW, get_scheduler

from common.utils import set_seed

class SampleWeightTrainer:
    def __init__(self, pred_models, pred_model_trainer, sw_model, num_epochs, train_dataset, train_collator,
                 val_datasets, val_collators, batch_size, weight_decay, adam_beta1, adam_beta2, adam_epsilon,
                 warmup_steps, warmup_ratio, learning_rate, lr_scheduler_type, seed, drop_last=True, num_workers=4,
                 pin_memory=True):
        self.pred_models = pred_models
        self.pred_model_trainer = pred_model_trainer
        self.sw_model = sw_model
        self.num_epochs = num_epochs
        self.train_dataset = train_dataset
        self.val_dataset = val_datasets
        self.train_collator = train_collator
        self.val_collators = val_collators
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler, self.optimizer = None, None
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Seed must be set before instantiating the model when using model
        self.seed = seed
        set_seed(self.seed)

        # setup model
        self.sw_model = nn.DataParallel(sw_model)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.sw_model = self.sw_model.to(self.device)

    def train(self):
        train_dataloader = self.get_train_dataloader()
        num_steps_per_epoch = len(train_dataloader)
        max_steps = math.ceil(num_steps_per_epoch * self.num_epochs)
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.create_optimizer_and_scheduler(num_training_steps=self.num_epochs)
        print("***** Running training (OUTER LOOP) *****")
        print(f"  Num examples = {len(self.train_dataset)}")
        print(f"  Sample Weight Num Epochs = {self.num_epochs}")
        print(f"  Sample Weight Batch Size = {self.batch_size }")
        print(f"  Sample Weight Total optimization steps = {max_steps}")
        start_time = time.time()
        self.sw_model.zero_grad()
        for epoch in range(self.num_epochs):
            for step, inputs in enumerate(train_dataloader):
                # TODO: Fill in


    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=RandomSampler(self.train_dataset),
            collate_fn=self.train_collator,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method (or :obj:`create_optimizer`
        and/or :obj:`create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        self.create_scheduler(num_training_steps)

    def create_optimizer(self):
        decay_parameters = get_parameter_names(self.sw_model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.sw_model.named_parameters() if n in decay_parameters],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.sw_model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        optimizer_kwargs = dict(
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_epsilon,
        )
        optimizer_kwargs["lr"] = self.learning_rate
        self.optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

    def create_scheduler(self, num_training_steps: int):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up before this method is called.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        self.lr_scheduler = get_scheduler(
            self.lr_scheduler_type,
            self.optimizer,
            num_warmup_steps=self.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps


