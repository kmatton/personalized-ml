import inspect
import json
import math
import os
import time

import higher
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, Adam
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler, WeightedRandomSampler
from transformers import EvalPrediction
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import nested_concat, nested_detach, nested_numpify, nested_truncate, \
    get_parameter_names
from transformers.trainer_utils import denumpify_detensorize, speed_metrics, TrainOutput
from IPython import embed

from common.utils import set_seed
from trainers.trainer_callbacks import EarlyStopping
from trainers.trainer_state import TrainerState


class BasicTrainer:
    """
    Basic Trainer for training NN model
    Notes on things I could change:
    - enable outputting/saving predictions for individual examples
    - allowing for metric_for_best_model to be something other than validation loss --> actually maybe that's why the
      Amazon Wild's perf is better --> are they selecting params based on accuracy + also by param search
    - saving more than just the best model as a checkpoint
    - output training progress
    - implement multi-process parallelism
    - implement removing data columns
    - implement gradient accumulation steps
    - resuming from checkpoint
    - saving checkpoints
    - reporting integration calbacks
    - logging + saving more information about training (e.g., flos, runtime, learning rate)
    """

    def __init__(self, model, loss_fn, seed, train_dataset, train_collator, val_dataset, val_collator,
                 data_parallel, optimizer_type, lr_scheduler_type, weight_decay, adam_beta1, adam_beta2, adam_epsilon,
                 learning_rate, warmup_steps, warmup_ratio, batch_size_per_device, num_train_epochs, compute_metrics,
                 logging_steps, eval_steps, output_dir, early_stopping=False, es_patience=3, es_threshold=0.001,
                 load_best_model_at_end=True, deterministic_mode=False, drop_last=False, num_workers=4, pin_memory=True,
                 max_grad_norm=1.0, verbose=True):

        self.verbose = verbose

        # setup optimization
        self.loss_fn = loss_fn
        self.lr_scheduler_type = lr_scheduler_type
        self.optimizer_type = optimizer_type
        self.optimizer, self.lr_scheduler = None, None
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.early_stopping = early_stopping
        self.es_patience = es_patience
        self.es_threshold = es_threshold
        self.es_callback = None
        self.load_best_model_at_end = load_best_model_at_end
        if self.early_stopping:
            self.es_callback = EarlyStopping(self.es_patience, self.es_threshold)

        # general model args
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps

        # data
        self.deterministic_mode = deterministic_mode
        self.train_dataset = train_dataset
        self.train_collator = train_collator
        self.val_dataset = val_dataset
        self.val_collator = val_collator
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # add sample idx to be able to determine what samples were selected
        sample_idx = [i for i in range(len(self.train_dataset))]
        if len(self.train_dataset) != len(self.train_dataset._data):
            # pyarrow datasets are weird so if dataset has been filtered, need to do this
            sample_idx += [-1] * (len(self.train_dataset._data) - len(self.train_dataset))
        self.train_dataset = self.train_dataset.add_column(name="index",
                                                           column=sample_idx)

        # metrics and progress tracking
        # fixing these for now, but may want to change later
        self.greater_is_better = False
        self.metric_to_check = "eval_loss"
        self.state = TrainerState()
        self.compute_metrics = compute_metrics
        self.output_dir = output_dir

        # Seed must be set before instantiating the model when using model
        self.seed = seed
        set_seed(self.seed)

        # setup model
        self.model = model
        self.batch_size_per_device = batch_size_per_device
        self.num_train_epochs = num_train_epochs
        model_signature = inspect.signature(self.model.forward)
        self.model_args = list(model_signature.parameters.keys())
        self.data_parallel = data_parallel
        if self.data_parallel:
            self.model = nn.DataParallel(self.model)
        # save model initial state
        self.model_init_dir = os.path.join(self.output_dir, "model_init")
        self.save_model(self.model_init_dir)
        self.init_weight_path = os.path.join(self.model_init_dir, "pytorch_model.bin")
        # create optimizer
        self.create_optimizer()

        if data_parallel:
            loss_fn_signature = inspect.signature(self.loss_fn.module.forward)
        else:
            loss_fn_signature = inspect.signature(self.loss_fn.forward)
        self.loss_fn_args = list(loss_fn_signature.parameters.keys())

        # place models on device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.batch_size_total = self.batch_size_per_device * self.n_gpu
        self.model = self.model.to(self.device)

        # check validity of args
        assert not(self.load_best_model_at_end and self.eval_steps is None), "need eval steps if loading best model"

    def train(self, sample_select_weights=None, sample_loss_weights=None, num_steps_per_epoch=None,
              save_selected_samples=False):
        """
        :param sample_select_weights: weights to use for dataloader sample selection
        :param sample_loss_weights: weights to use when weighting the contribution of each sample to the loss
        :param num_steps_per_epoch: number of training steps per epoch
        :param save_selected_samples: if applying weighted sample selection and set to true, return selected samples
        """
        num_samples = len(self.train_dataset)
        if num_steps_per_epoch is not None:
            num_samples = num_steps_per_epoch * self.batch_size_total
        train_dataloader = self.get_train_dataloader(sample_select_weights, num_samples)
        if num_steps_per_epoch is None:
            num_steps_per_epoch = len(train_dataloader)
        else:
            assert num_steps_per_epoch == len(train_dataloader), "error num_steps_per_epoch != len(train_dataloader)"
        max_steps = math.ceil(num_steps_per_epoch * self.num_train_epochs)
        self.create_scheduler(num_training_steps=max_steps)
        if self.verbose:
            print("***** Running training *****")
            print(f"  Num examples = {num_samples}")
            print(f"  Num Epochs = {self.num_train_epochs}")
            print(f"  Instantaneous batch size per device = {self.batch_size_per_device}")
            print(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.batch_size_total }")
            print(f" Steps per epoch = {num_steps_per_epoch}")
            print(f"  Total optimization steps = {max_steps}")
        start_time = time.time()
        self.model.zero_grad()
        total_train_sample_count = self.num_train_epochs * num_samples
        selected_samples = None
        if save_selected_samples:
            selected_samples = torch.zeros(total_train_sample_count, dtype=torch.int).to(device=self.device)
        self._train_loop(train_dataloader, sample_loss_weights, save_selected_samples, selected_samples)
        metrics = speed_metrics("train", start_time, num_samples=total_train_sample_count, num_steps=max_steps)
        self.state.log_history.append({**metrics, **{"step": self.state.global_step}})
        print("Training completed in {} steps.".format(self.state.global_step))
        if self.load_best_model_at_end:
            if self.verbose:
                print("loading best model")
            best_model_path = os.path.join(self.output_dir, "pytorch_model.bin")
            state_dict = torch.load(best_model_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=True)
        train_loss = self.state.total_train_loss / self.state.global_step
        if save_selected_samples:
            return TrainOutput(self.state.global_step, train_loss, metrics), selected_samples
        return TrainOutput(self.state.global_step, train_loss, metrics)

    def meta_train(self, fmodel, diffopt, sample_select_weights=None, sample_loss_weights=None,
                   num_steps_per_epoch=None):
        """
        :param fmodel: model wrapped in higher inner loop context
        :param diffopt: differentiable optimizer
        :param sample_loss_weights: weights to use when weighting the contribution of each sample to the loss
        :param num_steps_per_epoch: number of training steps per epoch
        """
        num_samples = len(self.train_dataset)
        if num_steps_per_epoch is not None:
            num_samples = num_steps_per_epoch * self.batch_size_total
        train_dataloader = self.get_train_dataloader(sample_select_weights, num_samples)
        if num_steps_per_epoch is None:
            num_steps_per_epoch = len(train_dataloader)
        else:
            assert num_steps_per_epoch == len(train_dataloader), "error num_steps_per_epoch != len(train_dataloader)"
        max_steps = math.ceil(num_steps_per_epoch * self.num_train_epochs)
        if self.verbose:
            print("***** Running training *****")
            print(f"  Num examples = {num_samples}")
            print(f"  Num Epochs = {self.num_train_epochs}")
            print(f"  Instantaneous batch size per device = {self.batch_size_per_device}")
            print(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.batch_size_total }")
            print(f" Steps per epoch = {num_steps_per_epoch}")
            print(f"  Total optimization steps = {max_steps}")
        start_time = time.time()
        fmodel.zero_grad()
        total_train_sample_count = self.num_train_epochs * num_samples
        self._meta_train_loop(fmodel, diffopt, train_dataloader, sample_loss_weights)
        metrics = speed_metrics("train", start_time, num_samples=total_train_sample_count, num_steps=max_steps)
        self.state.log_history.append({**metrics, **{"step": self.state.global_step}})
        print("Training completed in {} steps.".format(self.state.global_step))
        if self.load_best_model_at_end:
            if self.verbose:
                print("loading best model")
            best_model_path = os.path.join(self.output_dir, "pytorch_model.bin")
            state_dict = torch.load(best_model_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=True)
        train_loss = self.state.total_train_loss / self.state.global_step
        # get validation performance
        print("***** Running Meta-Evaluation *****")
        eval_dataloader = self.get_eval_dataloader(self.val_dataset, self.val_collator)
        losses = torch.zeros(len(self.val_dataset))
        for step, inputs in enumerate(eval_dataloader):
            loss = self.meta_eval_step(inputs)
            step_losses = loss.repeat(len(inputs["embeddings"]))
            start_idx = step * self.batch_size_total
            end_idx = start_idx + len(inputs["embeddings"])
            losses[start_idx:end_idx] = step_losses
        loss = losses.mean()  # this is what we want to to take the gradients of with respect to the weights
        loss.backward(retain_graph=True)
        return loss.detach().item()

    def _train_loop(self, train_dataloader, sample_loss_weights, save_selected_samples, selected_samples):
        for epoch in range(self.num_train_epochs):
            for step, inputs in enumerate(train_dataloader):
                if save_selected_samples:
                    start_idx = step * self.batch_size_total
                    end_idx = start_idx + self.batch_size_total
                    selected_samples[start_idx:end_idx] = inputs["index"]
                loss = self.training_step(inputs, sample_loss_weights)
                self.state.total_train_loss += loss.item()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.model.zero_grad()
                self.state.global_step += 1
                stop_training = self._maybe_log_save_evaluate(loss, epoch)
                if stop_training:
                    return

    def _meta_train_loop(self, fmodel, diffopt, train_dataloader, sample_loss_weights):
        for epoch in range(self.num_train_epochs):
            for step, inputs in enumerate(train_dataloader):
                loss = self.meta_training_step(fmodel, inputs, sample_loss_weights)
                self.state.total_train_loss += loss.item()
                diffopt.step(loss)
                self.state.global_step += 1
                stop_training = self._maybe_log_save_evaluate(loss, epoch)
                if stop_training:
                    return

    def evaluate(self, eval_dataset=None, eval_collator=None, metric_key_prefix="eval", print_loss=False):
        if eval_dataset is None:
            eval_dataset = self.val_dataset
        if eval_collator is None:
            eval_collator = self.val_collator
        eval_dataloader = self.get_eval_dataloader(eval_dataset, eval_collator)
        if self.verbose:
            print(f"***** Running Evaluation *****")
            print(f"  Num examples = {len(eval_dataset)}")
            print(f"  Batch size = {self.batch_size_total}")
        start_time = time.time()
        self.model.eval()

        # initialize containers on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None

        for step, inputs in enumerate(eval_dataloader):
            loss, logits, labels = self.prediction_step(inputs)
            losses = loss.repeat(self.batch_size_total)
            losses = nested_numpify(losses)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
            if logits is not None:
                logits = nested_numpify(logits)
                all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
            if labels is not None:
                labels = nested_numpify(labels)
                all_labels = (
                    labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                )

        num_samples = len(eval_dataset)
        # Number of losses has been rounded to a multiple of batch_size so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # compute metrics
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}
        metrics = denumpify_detensorize(metrics)  # needs to be JSON-serializable

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=num_samples,
                num_steps=math.ceil(num_samples / self.batch_size_total)
            )
        )
        if print_loss:
            print("{} loss: {}".format(metric_key_prefix, metrics[f"{metric_key_prefix}_loss"]))
        return metrics

    def training_step(self, inputs, sample_loss_weights):
        self.model.train()
        model_inputs = {k: v for k, v in inputs.items() if k in self.model_args and k != "labels"}
        model_inputs = self._prepare_inputs(model_inputs)
        labels = inputs.pop("labels")
        labels = labels.to(device=self.device)
        outputs = self.model(**model_inputs)
        logits = outputs['logits']
        index = inputs["sample_index"]
        if "weights" in self.loss_fn_args:
            weights = sample_loss_weights[index]
            weights = weights[:, None]
            loss = self.loss_fn(logits=logits, labels=labels, weights=weights)
            if self.n_gpu > 1:
                loss = loss.sum()
        else:
            loss = self.loss_fn(logits=logits, labels=labels)
            if self.n_gpu > 1:
                loss = loss.mean()
        loss.backward()
        return loss.detach()

    def meta_training_step(self, fmodel, inputs, sample_loss_weights):
        fmodel.train()
        model_inputs = {k: v for k, v in inputs.items() if k in self.model_args and k != "labels"}
        model_inputs = self._prepare_inputs(model_inputs)
        labels = inputs.pop("labels")
        labels = labels.to(device=self.device)
        outputs = fmodel(**model_inputs)
        logits = outputs['logits']
        index = inputs["sample_index"]
        if "weights" in self.loss_fn_args:
            weights = sample_loss_weights[index]
            weights = weights[:, None]
            loss = self.loss_fn(logits=logits, labels=labels, weights=weights)
            if self.n_gpu > 1:
                loss = loss.sum()
        else:
            loss = self.loss_fn(logits=logits, labels=labels)
            if self.n_gpu > 1:
                loss = loss.mean()
        return loss

    def prediction_step(self, inputs):
        model_inputs = {k: v for k, v in inputs.items() if k in self.model_args and k != "labels"}
        model_inputs = self._prepare_inputs(model_inputs)
        labels = inputs.pop("labels")
        labels = labels.to(device=self.device)
        with torch.no_grad():
            outputs = self.model(**model_inputs)
            logits = outputs['logits']
            loss = self.loss_fn(logits=logits, labels=labels)
            loss = loss.mean().detach()
        if self.compute_metrics is None:  # won't need to compute metrics so just return loss
            return loss, None, None
        logits = nested_detach(logits)
        return loss, logits, labels

    def meta_eval_step(self, inputs):
        model_inputs = {k: v for k, v in inputs.items() if k in self.model_args and k != "labels"}
        model_inputs = self._prepare_inputs(model_inputs)
        labels = inputs.pop("labels")
        labels = labels.to(device=self.device)
        outputs = self.model(**model_inputs)
        logits = outputs['logits']
        loss = self.loss_fn(logits=logits, labels=labels)
        loss = loss.mean()
        return loss

    def get_train_dataloader(self, sample_weights=None, num_samples=None):
        train_sampler = self._get_train_sampler(sample_weights, num_samples)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_total,
            sampler=train_sampler,
            collate_fn=self.train_collator,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def get_eval_dataloader(self, eval_dataset, eval_collator):
        sampler = SequentialSampler(eval_dataset)
        return DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.batch_size_total,
            collate_fn=eval_collator,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def save_model(self, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if self.verbose:
            print(f"Saving model checkpoint to {output_dir}")
        state_dict = self.model.state_dict()
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                if self.verbose:
                    print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)

    def re_init_model(self, model_weight_dir):
        """
        Reset model to its initial state
        # TODO: need to figure out how this works with nn.DataParallel
        """
        if model_weight_dir is None:
            model_weight_dir = self.model_init_dir
        loaded_weights = torch.load(os.path.join(model_weight_dir, "pytorch_model.bin"))
        self.model.load_state_dict(loaded_weights)

    def create_optimizer(self):
        if self.optimizer_type == "AdamW":
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_kwargs = dict(
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
            )
            optimizer_kwargs["lr"] = self.learning_rate
            self.optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
        elif self.optimizer_type == "Adam":
            self.optimizer = Adam(self.model.parameters())
        else:
            print("Unrecognized optimizer type {}".format(self.optimizer_type))
            print("Exiting...")
            exit(1)

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

    def save_metrics(self, split, metrics, combined=True):
        path = os.path.join(self.output_dir, f"{split}_results.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)
        if combined:
            path = os.path.join(self.output_dir, "all_results.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {}

            all_metrics.update(metrics)
            with open(path, "w") as f:
                json.dump(all_metrics, f, indent=4, sort_keys=True)

    def save_state(self):
        path = os.path.join(self.output_dir, "trainer_state.json")
        self.state.save_to_json(path)

    def reset(self):
        # reset model weights
        self.model.load_state_dict(torch.load(self.init_weight_path))
        # reset state and callbacks
        if self.early_stopping:
            self.es_callback.reset()
        self.state.reset()
        # reinit optimizer
        self.create_optimizer()

    def _prepare_inputs(self, inputs):
        inputs = {k: v.to(device=self.device) for k, v in inputs.items()}
        inputs = {k: v.float() for k, v in inputs.items()}
        return inputs

    def _get_train_sampler(self, sample_weights=None, num_samples=None):
        if num_samples is None:
            num_samples = len(self.train_dataset)
        assert not (sample_weights is not None and self.deterministic_mode), "can't use sample weights in deterministic mode"
        if sample_weights is not None:
            return WeightedRandomSampler(sample_weights, num_samples)
        if self.deterministic_mode:
            assert num_samples == len(self.train_dataset), "must use all samples when using sequential sampler"
            return SequentialSampler(self.train_dataset)
        if num_samples != len(self.train_dataset):
            return RandomSampler(self.train_dataset, replacement=True, num_samples=num_samples)
        return RandomSampler(self.train_dataset)

    def _maybe_log_save_evaluate(self, loss, epoch):
        stop_training = False
        if self.state.global_step % self.logging_steps == 0:
            self.state.log_history.append({"loss": loss.item(),
                                           "epoch": round(epoch, 2),
                                           "step": self.state.global_step})
        if self.state.global_step % self.eval_steps == 0:
            metrics = self.evaluate()
            metrics["epoch"] = round(epoch, 2)
            metrics["step"] = self.state.global_step
            if self.verbose:
                print("Validation loss: {}".format(metrics["eval_loss"]))
            self.state.log_history.append(metrics)
            if self.early_stopping:
                stop_training = self.es_callback.check_metric_value(metrics, self.state.best_metric,
                                                                    self.greater_is_better, self.metric_to_check)
                if stop_training:
                    print("Stopping training early at step {}".format(self.state.global_step))
            updated = self.state.maybe_update_best_metric(metrics, self.greater_is_better, self.metric_to_check)
            if self.load_best_model_at_end and updated:
                if self.verbose:
                    print("Saving new best model")
                # save model because it's a new best
                self.save_model()
        return stop_training
