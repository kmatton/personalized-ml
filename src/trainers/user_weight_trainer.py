"""
Trainer for jointly training user embedding model and prediction models, where distances in user embedding space
are used as weights for selecting training data to use for each person.
"""
import json
import math
import time
import os

import higher
import numpy as np
import torch
from IPython import embed
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers.optimization import AdamW, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import speed_metrics, TrainOutput

from common.utils import set_seed
from dataset.dataset_utils import filter_helper_single_val
from trainers.trainer_callbacks import EarlyStopping
from trainers.trainer_state import TrainerState


class UserWeightTrainer:
    # potential updates:
    # - penalty to encourage exploration
    # - computing difference in perf w.r.t. global pre-train + local fine-tune rather than just global pre-train
    # - would want to have these models pre-trained & saved then
    # - may want to train pred models to convergence
    # notes on pred models
    # 1 epoch, no early stopping, don't load best model at end
    # note on parameter settings --> may not want to use same settings as prediction model, but am starting by doing that
    def __init__(self, user_embed_model, train_users, train_user_input_embeds, pred_model_trainer, seed,
                 user_col, train_dataset, val_dataset, lr_scheduler_type, weight_decay, adam_beta1, adam_beta2,
                 adam_epsilon, learning_rate, warmup_steps, warmup_ratio, user_embed_batch_size, outer_steps,
                 inner_steps, eval_steps, user_batch_size, output_dir, early_stopping=False, es_patience=10,
                 es_threshold=0.0, load_best_model_at_end=False, deterministic_mode=False, num_workers=4,
                 pin_memory=True, max_grad_norm=1.0):
        # setup optimization
        self.lr_scheduler_type = lr_scheduler_type
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
        self.max_grad_norm = max_grad_norm
        self.eval_steps = eval_steps

        # data
        self.deterministic_mode = deterministic_mode
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # metrics and progress tracking
        # fixing these for now, but may want to change later
        self.greater_is_better = False
        self.metric_to_check = "eval_loss"
        self.state = TrainerState()
        self.output_dir = output_dir

        # Seed must be set before instantiating the model when using model
        self.seed = seed
        set_seed(self.seed)

        # setup device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        self.pred_model_trainer = pred_model_trainer

        # user embedding model args
        self.user_embed_model = user_embed_model
        self.user_embed_batch_size = user_embed_batch_size
        self.outer_steps = outer_steps
        self.train_user_input_embeds = train_user_input_embeds
        self.user_col = user_col
        train_users_per_sample = self.train_dataset[self.user_col]
        self.train_users = train_users
        self.train_user_ids = np.arange(len(self.train_users))
        self.user_to_id = {user: i for i, user in enumerate(self.train_users)}
        # user ids to train dataset for faster sample selection
        self.train_user_ids_per_sample = [self.user_to_id[user] for user in train_users_per_sample]
        self.train_dataset = self.train_dataset.add_column(name="user_id", column=self.train_user_ids_per_sample)
        self.user_train_counts = torch.tensor([train_users_per_sample.count(user) for user in self.train_users])
        self.user_train_counts = self.user_train_counts.to(self.device)

        self.user_embed_model = nn.DataParallel(self.user_embed_model)
        self.user_weight_matrix = None

        # get validation data for each user
        self.val_data_per_user = [filter_helper_single_val(self.val_dataset, self.user_col, user)
                                  for user in self.train_users]
        # initialize validation losses for each user
        self.init_val_losses = torch.zeros(len(self.train_users))
        for idx, user_val_data in enumerate(self.val_data_per_user):
            val_metrics = self.pred_model_trainer.evaluate(user_val_data)
            self.init_val_losses[idx] = val_metrics["eval_loss"]
        self.init_val_losses = self.init_val_losses.to(self.device)

        # place models on device
        self.user_embed_model = self.user_embed_model.to(self.device)

        # pred model args
        self.user_batch_size = user_batch_size
        self.inner_steps = inner_steps

        # operators and constants
        self.cos_sim = nn.CosineSimilarity()
        self.pi = torch.tensor(math.pi)

        # modes
        self.mode = "weight_samples"  # other modes are: select_samples and select_users (weight_samples)
        self.normalize_weights = False  # if true, weigh data to ensure uniform distr. over users when selecting samples
        self.use_ma_baseline = True
        self.training_strategy = "meta_learning"  # other option is RL

        # initialize moving average loss differences for each user
        if self.use_ma_baseline:
            self.ma_losses = torch.zeros(len(self.train_users))
            self.ma_losses = self.ma_losses.to(self.device)

        # other params
        # epsilon for taking log prob
        self.epsilon = 1e-8
        torch.autograd.set_detect_anomaly(True)
        self.sim_metric = "gibbs"  # other options are cosine, angular

        # set to debug mode or not
        self.debug = True

        # check validity of args
        assert not (self.load_best_model_at_end and self.eval_steps is None), "need eval steps if loading best model"

    def train(self):
        self.create_optimizer_and_scheduler(num_training_steps=self.outer_steps)
        print("***** Running Outer Training Loop *****")
        print(f"  Num people = {len(self.train_users)}")
        print(f"  Max Num Outer Steps = {self.outer_steps}")
        start_time = time.time()
        self.user_embed_model.train()
        self.user_embed_model.zero_grad()
        step_embed_losses = np.empty((self.outer_steps, len(self.train_users)))
        step_val_losses = np.empty((self.outer_steps, len(self.train_users)))
        step_val_loss_diffs = np.empty((self.outer_steps, len(self.train_users)))
        prev_W = None
        for step in range(self.outer_steps):
            init_time = time.time()
            # (1) get user weight matrix
            W = self.get_user_weight_matrix()
            if prev_W is not None:
                W_diff_mean = torch.mean(torch.abs(prev_W - W))
                print("Mean difference with last W {}".format(W_diff_mean))
            prev_W = W
            if self.debug:
                print("Starting Outer Step {}".format(step))
                print("Weight Matrix")
                print(W)
            # (3) sample batch of users to train prediction models for
            train_batch = np.random.choice(self.train_user_ids, self.user_batch_size, replace=False)
            if self.debug:
                print("Selected users are {}".format(train_batch))
            # (4) train pred model and get validation loss for each user
            losses = torch.zeros(len(train_batch))
            for idx, pred_user in enumerate(train_batch):
                if self.debug:
                    print("Starting model fine-tuning for user {}".format(pred_user))
                val_metrics, selection, selection_probs = self.train_and_evaluate_single_user(pred_user, W, step, phase="train")
                if self.debug:
                    print("Validation performance for user {}".format(pred_user))
                    print("loss", val_metrics["eval_loss"])
                    if "eval_accuracy" in val_metrics:
                        print("accuracy", val_metrics["eval_accuracy"])
                val_loss = val_metrics["eval_loss"]
                step_val_losses[step][pred_user] = val_loss
                # (5) get loss contribution w.r.t. embedding model
                if self.training_strategy == "RL":
                    user_embed_loss, loss_diff = self.get_user_embed_loss(val_loss, pred_user, selection, selection_probs, step + 1)
                    step_val_loss_diffs[step][pred_user] = loss_diff
                    step_embed_losses[step][pred_user] = user_embed_loss.detach().item()
                    losses[idx] = user_embed_loss
                else:
                    losses[idx] = val_loss
            # (6) update user embedding model with validation loss
            # take mean loss across all people
            loss = losses.mean()
            if self.training_strategy == "RL":
                print("Mean validation loss at step {} is {}".format(step, np.mean(step_val_losses[step])))
                print("Mean validation loss diff at step {} is {}".format(step, np.mean(step_val_loss_diffs[step])))
            print("Mean user embed loss at step {} is {}".format(step, loss.item()))
            if self.training_strategy == "RL":
                loss.backward()
                loss = loss.detach()
            self.state.total_train_loss += loss.item()
            nn.utils.clip_grad_norm_(self.user_embed_model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.user_embed_model.zero_grad()
            self.state.global_step += 1
            stop_training = self._maybe_save_evaluate(loss.item())
            if stop_training:
                break
            print("Finished inner loop step {} in time {}".format(step, time.time() - init_time))
        print("Finished embedding model training in {} steps.".format(self.state.global_step))
        metrics = speed_metrics("train", start_time,
                                num_samples=self.outer_steps * self.user_batch_size,
                                num_steps=self.outer_steps)
        self.state.log_history.append({**metrics, **{"step": self.state.global_step}})
        np.save(os.path.join(self.output_dir, "step_val_losses.npy"), step_val_losses)
        np.save(os.path.join(self.output_dir, "step_embed_losses.npy"), step_embed_losses)
        if self.load_best_model_at_end:
            print("loading best user embedding model")
            best_model_path = os.path.join(self.output_dir, "pytorch_model.bin")
            state_dict = torch.load(best_model_path, map_location="cpu")
            self.user_embed_model.load_state_dict(state_dict, strict=True)
        train_loss = self.state.total_train_loss / self.state.global_step
        return TrainOutput(self.state.global_step, train_loss, metrics)

    def get_user_weight_matrix(self, is_train=True):
        # (1) get user embeddings from user embedding model
        user_embeds = self.get_user_embeddings(self.train_user_input_embeds, is_train=is_train)
        # (2) get pairwise dists from user embeds
        num_users = len(user_embeds)
        lhs = user_embeds.repeat_interleave(num_users, dim=0)
        rhs = user_embeds.repeat(num_users, 1)
        W = self.cos_sim(lhs, rhs).reshape(num_users, num_users)
        if self.sim_metric == "gibbs": # convert to distance and take e^-x
            W = 1 - W
            W = torch.exp(-W)
        elif self.sim_metric == "angular":
            # convert to [0, 1] scale by computing angular similarity
            W = torch.acos(W)
            W = W / self.pi
            W = 1 - W
        return W

    def evaluate(self, eval_dataset=None, metric_key_prefix="eval", step=None):
        """
        Evaluate trained user embedding model for all users
        """
        if step is None:
            step = "final_eval"
        data_per_user = self.val_data_per_user
        eval_users = self.train_users
        eval_user_ids = self.train_user_ids
        if eval_dataset is not None:
            # get data per user --> can only evaluate on users seen during training
            eval_users = list(set(eval_dataset[self.user_col]))
            eval_user_ids = [self.user_to_id[user] for user in eval_users]
            data_per_user = [filter_helper_single_val(eval_dataset, self.user_col, user) for user in eval_users]
        print(f"***** Running Evaluation Across All Users *****")
        print(f"  Num users = {len(eval_users)}")
        start_time = time.time()
        # get user weight matrix
        val_losses = np.zeros(len(eval_users))
        idx = 0
        user_weight_matrix = self.get_user_weight_matrix(is_train=False)
        if self.debug:
            print("Weight matrix (in eval)")
            print(user_weight_matrix)
        metrics = {}
        for user, user_id, user_data in zip(eval_users, eval_user_ids, data_per_user):
            val_metrics, _, _ = self.train_and_evaluate_single_user(user_id, user_weight_matrix, step, phase="eval")
            metrics[str(user)] = val_metrics
            val_loss = val_metrics["{}_loss".format(metric_key_prefix)]
            val_losses[idx] = val_loss
            idx += 1
        metrics["{}_losses".format(metric_key_prefix)] = val_losses.tolist()
        metrics["users"] = eval_users
        metrics["{}_loss".format(metric_key_prefix)] = np.mean(val_losses)
        metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=len(eval_users),
                num_steps=1
            )
        )
        if self.debug:
            print("Mean {} loss across all users: {}".format(metric_key_prefix,
                                                             metrics["{}_loss".format(metric_key_prefix)]))
        return metrics

    def train_and_evaluate_single_user(self, user, user_weight_matrix, step, phase="train"):
        """
        Train model using the weights given by the user_weight_vector and evaluate on user's data.
        """
        self.pred_model_trainer.reset()
        # reset train output dir
        self.pred_model_trainer.output_dir = os.path.join(self.output_dir, phase, str(step), str(self.train_users[user]))
        # reset validation data
        user_val_data = self.val_data_per_user[user]
        self.pred_model_trainer.val_dataset = user_val_data
        # get sample weights and train model
        user_weights = user_weight_matrix[user]
        selection = None
        selection_probs = None
        if self.mode == "select_users":
            # sample users from user weights
            selected_users = torch.bernoulli(user_weights)
            if self.debug:
                print("selected users are: {}".format(selected_users))
            # get sample weights based on selected users
            sample_weights = self.user_weights_to_sample_weights(selected_users)
            # train model
            train_output = self.pred_model_trainer.train(sample_select_weights=sample_weights,
                                                         num_steps_per_epoch=self.inner_steps,
                                                         save_selected_samples=False)
            val_metrics = self.pred_model_trainer.evaluate(eval_dataset=user_val_data, print_loss=True)
            selection = selected_users
            selection_probs = user_weights
        elif self.mode == "select_samples":
            # get sample weights from user weights
            sample_weights = self.user_weights_to_sample_weights(user_weight_matrix[user])
            # train model
            train_output, selection = self.pred_model_trainer.train(sample_select_weights=sample_weights,
                                                                    num_steps_per_epoch=self.inner_steps,
                                                                    save_selected_samples=True)
            val_metrics = self.pred_model_trainer.evaluate(eval_dataset=user_val_data, print_loss=True)
            selection_probs = sample_weights
        else:
            assert self.mode == "weight_samples", print("invalid mode {}".format(self.mode))
            # get sample weights from user weights
            sample_weights = self.user_weights_to_sample_weights(user_weight_matrix[user])
            # switch model and optimizer to those returned by higher context manager
            with higher.innerloop_ctx(self.pred_model_trainer.model, self.pred_model_trainer.optimizer) as (fmodel, diffopt):
                val_loss = self.pred_model_trainer.meta_train(fmodel, diffopt, sample_loss_weights=sample_weights,
                                                              num_steps_per_epoch=self.inner_steps)
                # get validation perf
                val_metrics = {"eval_loss": val_loss}
        return val_metrics, selection, selection_probs

    def get_user_embed_loss(self, val_loss, pred_user, selection, selection_probs, step):
        # take difference between loss and global model perf
        curr_loss_minus_global = val_loss - self.init_val_losses[pred_user]
        loss_diff = curr_loss_minus_global
        if self.debug:
            print("loss diff with global model loss: {}".format(loss_diff))
        if self.use_ma_baseline:
            # take difference with M.A. baseline
            loss_diff = curr_loss_minus_global - self.ma_losses[pred_user]
            if self.debug:
                print("loss diff with ma baseline {}".format(loss_diff))
        # compute log prob of selection
        if self.mode == "select_users":
            # get log prob of user selection
            log_prob_select = torch.sum(selection * torch.log(selection_probs + self.epsilon)) + \
                              torch.sum((1 - selection) * torch.log(1 - selection_probs + self.epsilon))
            if self.debug:
                print("log prob of selection vector {}".format(log_prob_select))
        else:  # mode is select samples
            # get counts of the number of times each sample was selected
            sample_select_counts = torch.bincount(selection, minlength=len(selection_probs))
            # get log prob of sample selection
            log_prob_select = torch.sum(sample_select_counts * torch.log(selection_probs))
        # get final loss term for user embed model
        user_embed_loss = loss_diff * log_prob_select
        if self.debug:
            print("overall user embedding loss {}".format(user_embed_loss))
        if self.use_ma_baseline:
            # update moving average perf
            self.ma_losses[pred_user] = ((step - 1) / step) * self.ma_losses[pred_user] + \
                                        (1 / step) * curr_loss_minus_global
        return user_embed_loss, loss_diff

    def user_weights_to_sample_weights(self, user_weights):
        if self.normalize_weights:
            user_weights = user_weights / self.user_train_counts
        sample_weights = user_weights[self.train_user_ids_per_sample]
        return sample_weights

    def get_user_embeddings(self, input_user_embeds, is_train=True):
        user_embed_dataloader = self.get_user_embed_dataloader(input_user_embeds)
        num_batches = len(user_embed_dataloader)
        if is_train:
            self.user_embed_model.train()
        else:
            self.user_embed_model.eval()
        all_embeds = torch.zeros(len(self.train_users), self.user_embed_model.module.n_outputs).to(device=self.device)
        for i, batch in enumerate(user_embed_dataloader):
            start = i * self.user_embed_batch_size
            end = start + self.user_embed_batch_size
            if i == num_batches - 1:
                end = len(self.train_users)
            batch = batch.to(device=self.device).float()
            embeds = self.user_embed_model(batch)
            all_embeds[start:end] = embeds
        return all_embeds

    def get_user_embed_dataloader(self, input_user_embeds):
        user_embed_sampler = SequentialSampler(input_user_embeds)
        return DataLoader(
            input_user_embeds,
            batch_size=self.user_embed_batch_size,
            sampler=user_embed_sampler,
            drop_last=False,
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
        decay_parameters = get_parameter_names(self.user_embed_model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.user_embed_model.named_parameters() if n in decay_parameters],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.user_embed_model.named_parameters() if n not in decay_parameters],
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

    def save_model(self, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")
        state_dict = self.user_embed_model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    def _maybe_save_evaluate(self, loss):
        stop_training = False
        self.state.log_history.append({"loss": loss,
                                       "step": self.state.global_step})
        if self.state.global_step % self.eval_steps == 0:
            metrics = self.evaluate(step=self.state.global_step)
            metrics["step"] = self.state.global_step
            self.state.log_history.append(metrics)
            # save history so far
            self.save_state()
            if self.early_stopping:
                stop_training = self.es_callback.check_metric_value(metrics, self.state.best_metric,
                                                                    self.greater_is_better, self.metric_to_check)
                if stop_training:
                    print("Stopping training early at step {}".format(self.state.global_step))
            updated = self.state.maybe_update_best_metric(metrics, self.greater_is_better, self.metric_to_check)
            if self.load_best_model_at_end and updated:
                print("Saving new best user embedding model")
                # save model because it's a new best
                self.save_model()
        return stop_training

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
