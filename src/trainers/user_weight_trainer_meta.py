"""
Cleaner User Weight Trainer for Meta Learning Approach
"""
import inspect
import os
import math
import time
import json

from IPython import embed
import higher
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import EvalPrediction
from transformers.trainer_pt_utils import nested_concat, nested_detach, nested_numpify, nested_truncate, \
    get_parameter_names
from transformers.trainer_utils import denumpify_detensorize, speed_metrics, TrainOutput

from common.utils import set_seed
from dataset.dataset_utils import filter_helper_single_val
from trainers.trainer_callbacks import EarlyStopping
from trainers.trainer_state import TrainerState

torch.autograd.set_detect_anomaly(True)


class UserWeightTrainerMeta:
    def __init__(self, users, user_col_name, seed, output_dir, prediction_model, train_dataset, val_dataset,
                 train_collator, val_collator, pred_loss_fn, compute_metrics, pred_lr, meta_lr, user_batch_size,
                 n_outer_it):
        self.users = users
        self.num_users = len(users)
        self.user_col_name = user_col_name
        self.seed = seed
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.prediction_model = prediction_model
        self.es_callback_meta = EarlyStopping(3, 0.0)
        set_seed(self.seed)
        self.loss_fn = pred_loss_fn

        # setup params
        self.meta_lr = meta_lr
        self.pred_lr = pred_lr
        self.n_outer_it = n_outer_it
        self.n_train_epochs = 20
        self.verbose = False
        self.save_pred_model_results = False
        self.user_batch_size = user_batch_size
        self.pred_batch_size = 32
        self.eval_epochs_pred = 1  ## TODO: increase this!!
        self.early_stopping = False
        self.epsilon = 1e-5

        # setup data
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_collator = train_collator
        self.val_collator = val_collator
        train_users_per_sample = self.train_dataset[self.user_col_name]
        self.user_ids = np.arange(self.num_users)
        self.user_to_id = {user: i for i, user in enumerate(self.users)}
        self.train_user_ids_per_sample = [self.user_to_id[user] for user in train_users_per_sample]
        self.train_dataset = self.train_dataset.add_column(name="user_id", column=self.train_user_ids_per_sample)
        self.train_user_ids = list(set(self.train_dataset["user_id"]))
        self.train_dataloader = self.get_train_dataloader()
        # get validation data for each user
        self.val_data_per_user = [filter_helper_single_val(self.val_dataset, self.user_col_name, user)
                                  for user in self.users]
        self.val_dl_per_user = [self.get_eval_dataloader(val_data, self.val_collator) for val_data in self.val_data_per_user]

        # setup device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        # setup state and metric tracking
        self.state = TrainerState()
        self.compute_metrics = compute_metrics

        # setup pred model
        model_signature = inspect.signature(self.prediction_model.forward)
        self.model_args = list(model_signature.parameters.keys())
        self.prediction_model = self.prediction_model.to(self.device)

        # instead of weight matrix, init user embeddings
        # start with all ones for now, but may want different initialization later
        self.user_embed_dim = 5  # make this a an argument to init later!
        self.user_embeds = torch.ones((self.num_users, self.user_embed_dim), device=self.device)
        self.user_embeds = self.user_embeds.requires_grad_()

        # # init weight matrix --> was uniform distribution over [0, 1) (torch.rand), now trying all 1s
        # self.W = torch.ones((self.num_users, self.num_users), device=self.device)
        # self.W = self.W.requires_grad_()
        self.W = None
        self.W_prime = None

    def get_user_weight_matrix(self, user_embeds):
        """
        Use user embeddings to get
        """
        W = torch.matmul(user_embeds, user_embeds.T)
        return W

    def train(self):
        start_time = time.time()
        print("*** Running Outer Training Loop ***")
        # meta_opt = optim.Adam([self.W], self.meta_lr)
        meta_opt = optim.Adam([self.user_embeds], self.meta_lr)
        mean_losses = np.empty(self.n_outer_it)
        train_users = np.zeros((self.n_outer_it, self.user_batch_size))  # keep track of user trained on in inner loop
        user_losses = np.empty((self.n_outer_it, self.user_batch_size))
        W_matrices = np.empty((self.n_outer_it, self.user_batch_size, self.num_users, self.num_users))
        self.W_grads = np.empty((self.n_outer_it, self.user_batch_size, self.num_users, self.num_users))
        user_embed_grads = np.empty((self.n_outer_it, self.user_batch_size, self.num_users, self.user_embed_dim))
        step_user_embeds = np.empty((self.n_outer_it, self.user_batch_size, self.num_users, self.user_embed_dim))
        prev_W = None
        prev_embeds = None
        for step in range(self.n_outer_it):
            init_time = time.time()
            if prev_embeds is not None:
                embeds_diff_mean = torch.mean(torch.abs(prev_embeds - self.user_embeds))
                if self.verbose:
                    print("Mean difference with last embeds {}".format(embeds_diff_mean))
                if embeds_diff_mean <= self.epsilon:
                    print("User embeds converged, stopping training")
                    break
            if prev_W is not None:
                W_diff_mean = torch.mean(torch.abs(prev_W - self.W))
                if self.verbose:
                    print("Mean difference with last W {}".format(W_diff_mean))
                if W_diff_mean <= self.epsilon:
                    print("Weight matrix has converged, stopping training")
                    break
            prev_embeds = self.user_embeds.clone()
            if self.W is not None:
                prev_W = self.W.clone()
            # sample batch of users to train prediction models for
            train_batch = np.random.choice(self.train_user_ids, self.user_batch_size, replace=False)
            if self.verbose:
                print("Selected users are {}".format(train_batch))
            # train prediction model for each user
            for idx, pred_user in enumerate(train_batch):
                meta_opt.zero_grad()
                step_user_embeds[step, idx] = self.user_embeds.cpu().detach().numpy()
                train_users[step, idx] = pred_user
                if self.verbose:
                    print("Starting model fine-tuning for user {}".format(pred_user))
                    print("user embeds at outer step {}: {}".format(step, self.user_embeds))
                self.W = self.get_user_weight_matrix(self.user_embeds)
                self.W.retain_grad()
                W_matrices[step, idx] = self.W.cpu().detach().numpy()
                # normalize to be non-negative and so that max weight is 1
                self.W_prime = nn.functional.relu(self.W, inplace=False)
                self.W_prime /= torch.max(self.W_prime, dim=1).values[:, None]
                if self.verbose:
                    print("W at step {} = {}".format(step, self.W))
                # get weights for this user
                user_weights = self.W_prime[pred_user]
                sample_weights = user_weights[self.train_user_ids_per_sample]
                loss = self.pred_model_train(pred_user, sample_weights, step)
                user_losses[step, idx] = loss
                # report gradients
                if self.verbose:
                    print("User embeds gradient at step {} for user {}: {}".format(step, pred_user, self.user_embeds.grad))
                user_embed_grads[step, idx] = self.user_embeds.grad.cpu().detach().numpy()
                if self.verbose:
                    print("Weight Matrix gradient at step {} for user {}: {}".format(step, pred_user, self.W.grad.cpu().detach().numpy()))
                self.W_grads[step, idx] = self.W.grad.cpu().detach().numpy()
                meta_opt.step()  # switched to doing updates every time we see new user
            mean_losses[step] = np.mean(user_losses[step])
            if self.verbose:
                print("Mean loss at step {}: {}".format(step, np.mean(user_losses[step])))
            stop_training = self._maybe_save_evaluate(mean_losses[step])
            if stop_training:
                break
            self.state.global_step += 1
            if self.verbose:
                print("Finished outer loop step {} in time {}".format(step, time.time() - init_time))
        if self.verbose:
            print("Finished meta training in {} steps".format(self.state.global_step))
        metrics = speed_metrics("train", start_time,
                                num_samples=self.state.global_step * self.user_batch_size,
                                num_steps=self.state.global_step)
        self.state.log_history.append({**metrics, **{"step": self.state.global_step}})
        np.save(os.path.join(self.output_dir, "step_mean_losses.npy"), mean_losses)
        np.save(os.path.join(self.output_dir, "step_user_losses.npy"), user_losses)
        np.save(os.path.join(self.output_dir, "final_user_weight_matrix.npy"), self.W.cpu().detach().numpy())
        np.save(os.path.join(self.output_dir, "user_weight_matrices.npy"), W_matrices)
        np.save(os.path.join(self.output_dir, "weight_gradients.npy"), self.W_grads)
        np.save(os.path.join(self.output_dir, "user_embeds_gradients.npy"), user_embed_grads)
        np.save(os.path.join(self.output_dir, "user_embeds.npy"), step_user_embeds)
        np.save(os.path.join(self.output_dir, "train_users.npy"), train_users)
        return TrainOutput(self.state.global_step, mean_losses[self.state.global_step - 1], metrics)

    def save_model(self):
        pass

    def evaluate(self, eval_dataset=None, metric_key_prefix="eval"):
        """
        Evaluate trained user weight matrix
        """
        data_per_user = self.val_data_per_user
        eval_users = self.users
        eval_user_ids = self.train_user_ids
        if eval_dataset is not None:
            # get data per user --> can only evaluate on users seen during training
            eval_users = list(set(eval_dataset[self.user_col_name]))
            eval_user_ids = [self.user_to_id[user] for user in eval_users]
            data_per_user = [filter_helper_single_val(eval_dataset, self.user_col_name, user) for user in eval_users]
        if self.verbose:
            print(f"***** Running Evaluation Across All Users *****")
            print(f"  Num users = {len(eval_users)}")
        start_time = time.time()
        # get user weight matrix
        val_losses = np.zeros(len(eval_users))
        idx = 0
        if self.verbose:
            print("Weight matrix (in eval)")
            print(self.W)
        metrics = {}
        for user, user_id, user_data in zip(eval_users, eval_user_ids, data_per_user):
            user_weights = self.W_prime[user_id]
            sample_weights = user_weights[self.train_user_ids_per_sample]
            loss = self.pred_model_train(user_id, sample_weights, outer_step="eval")
            val_dl = self.get_eval_dataloader(user_data, self.val_collator)
            val_metrics = self.pred_model_evaluate(val_dl, metric_key_prefix=metric_key_prefix)
            val_metrics["final_epoch_train_loss"] = loss
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
        if self.verbose:
            print("Mean {} loss across all users: {}".format(metric_key_prefix,
                                                             metrics["{}_loss".format(metric_key_prefix)]))
        return metrics

    def pred_model_train(self, pred_user, sample_weights, outer_step):
        self.prediction_model.init_weights()
        self.prediction_model.zero_grad()
        pred_opt = optim.SGD(self.prediction_model.parameters(), self.pred_lr)
        self.prediction_model.train()
        es_callback_pred = EarlyStopping(3, 0.0)
        trainer_state_pred = TrainerState()
        with higher.innerloop_ctx(self.prediction_model, pred_opt) as (fmodel, diffopt):
            for epoch in range(self.n_train_epochs):
                epoch_loss = 0
                for step, inputs in enumerate(self.train_dataloader):
                    fmodel.train()
                    model_inputs = {k: v for k, v in inputs.items() if k in self.model_args and k != "labels"}
                    model_inputs = self._prepare_inputs(model_inputs)
                    labels = inputs.pop("labels")
                    labels = labels.to(device=self.device)
                    outputs = fmodel(**model_inputs)
                    logits = outputs['logits']
                    index = inputs["sample_index"]
                    weights = sample_weights[index]
                    weights = weights[:, None]
                    loss = self.loss_fn(logits=logits, labels=labels, weights=weights)
                    loss /= self.pred_batch_size
                    diffopt.step(loss)
                    trainer_state_pred.global_step += 1
                    trainer_state_pred.log_history.append({"loss": loss.item(), "epoch": round(epoch, 2),
                                                           "step": trainer_state_pred.global_step})
                    epoch_loss += loss.item()
                epoch_loss /= len(self.train_dataloader)
                if epoch % self.eval_epochs_pred == 0:
                    self.prediction_model.load_state_dict(fmodel.state_dict())
                    metrics = self.pred_model_evaluate(self.val_dl_per_user[pred_user])
                    metrics["epoch"] = round(epoch, 2)
                    metrics["step"] = trainer_state_pred.global_step
                    metrics["mean_train_loss_per_step"] = epoch_loss
                    if self.verbose:
                        print("Eval loss for pred model at inner epoch {}: {}".format(epoch, metrics["eval_loss"]))
                    trainer_state_pred.log_history.append(metrics)
                    # check for early stopping
                    stop_training = es_callback_pred.check_metric_value(metrics,
                                                                        trainer_state_pred.best_metric,
                                                                        False, "eval_loss")
                    if stop_training:
                        if self.verbose:
                            print("Stopping training model for user {} early at {} epochs".format(pred_user, epoch))
                        break
                    trainer_state_pred.maybe_update_best_metric(metrics, False, "eval_loss")
            if self.verbose:
                print("Completed pred model training in {} epochs.".format(epoch))
            # save weights to prediction model
            self.prediction_model.load_state_dict(fmodel.state_dict())
            if self.save_pred_model_results:
                # save trainer state
                save_dir = os.path.join(self.output_dir, "pred_model_results")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, "user_{}_step_{}_trainer_state.json".format(pred_user, outer_step))
                trainer_state_pred.save_to_json(save_path)
            if str(outer_step) == "eval":  # don't need to do meta learning validation loss backprop step
                return epoch_loss
            if self.verbose:
                print("Evaluating Final Trained Pred Model for User {} at outer step {}".format(pred_user, outer_step))
            eval_dataloader = self.val_dl_per_user[pred_user]
            losses = torch.zeros(len(self.val_data_per_user[pred_user]))
            fmodel.train()
            for step, inputs in enumerate(eval_dataloader):
                model_inputs = {k: v for k, v in inputs.items() if k in self.model_args and k != "labels"}
                model_inputs = self._prepare_inputs(model_inputs)
                labels = inputs.pop("labels")
                labels = labels.to(device=self.device)
                outputs = fmodel(**model_inputs)
                logits = outputs['logits']
                loss = self.loss_fn(logits=logits, labels=labels)
                step_losses = loss.repeat(len(inputs["embeddings"]))
                start_idx = step * self.pred_batch_size
                end_idx = start_idx + len(inputs["embeddings"])
                losses[start_idx:end_idx] = step_losses
            loss = losses.mean()
            if self.verbose:
                print("Final Eval Loss for User {} at Outer step {}: {}".format(pred_user, outer_step, loss))
            loss.backward(retain_graph=True)
            return loss.detach().item()

    def pred_model_evaluate(self, eval_dataloader, metric_key_prefix="eval"):
        self.prediction_model.eval()
        start_time = time.time()
        # initialize containers on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None

        for step, inputs in enumerate(eval_dataloader):
            loss, logits, labels = self.prediction_step(inputs)
            losses = loss.repeat(self.pred_batch_size)
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

        num_samples = len(eval_dataloader.dataset)
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
                    num_steps=math.ceil(num_samples / self.pred_batch_size)
                )
            )
            return metrics

    def prediction_step(self, inputs):
        model_inputs = {k: v for k, v in inputs.items() if k in self.model_args and k != "labels"}
        model_inputs = self._prepare_inputs(model_inputs)
        labels = inputs.pop("labels")
        labels = labels.to(device=self.device)
        with torch.no_grad():
            outputs = self.prediction_model(**model_inputs)
            logits = outputs['logits']
            loss = self.loss_fn(logits=logits, labels=labels)
            loss = loss.mean().detach()
        if self.compute_metrics is None:  # won't need to compute metrics so just return loss
            return loss, None, None
        logits = nested_detach(logits)
        return loss, logits, labels

    def get_train_dataloader(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.pred_batch_size,
            sampler=train_sampler,
            collate_fn=self.train_collator,
            drop_last=False,
            num_workers=4,
            pin_memory=True
        )
        return train_dataloader

    def get_eval_dataloader(self, eval_dataset, eval_collator):
        sampler = SequentialSampler(eval_dataset)
        return DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.pred_batch_size,
            collate_fn=eval_collator,
            drop_last=False,
            num_workers=4,
            pin_memory=True
        )

    def _prepare_inputs(self, inputs):
        inputs = {k: v.to(device=self.device) for k, v in inputs.items()}
        inputs = {k: v.float() for k, v in inputs.items()}
        return inputs

    def _maybe_save_evaluate(self, eval_loss):
        metrics = {"eval_loss": eval_loss, "step": self.state.global_step}
        self.state.log_history.append(metrics)
        stop_training = False
        if self.early_stopping:
            stop_training = self.es_callback_meta.check_metric_value(metrics, self.state.best_metric,
                                                                     False, "eval_loss")
        if self.verbose:
            print("Current Eval Loss: {} Compared to Best so Far {}".format(metrics["eval_loss"], self.state.best_metric))
        if stop_training:
            print("Stopping meta training early at step {}".format(self.state.global_step))
        self.state.maybe_update_best_metric(metrics, False, "eval_loss")
        return stop_training

    def save_state(self):
        path = os.path.join(self.output_dir, "trainer_state.json")
        self.state.save_to_json(path)

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
