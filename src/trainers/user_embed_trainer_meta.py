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


class UserEmbedTrainerMeta:
    def __init__(self, users, user_col_name, seed, output_dir, prediction_model, user_embed_model, init_user_embeds,
                 train_dataset, val_dataset, train_collator, val_collator, pred_loss_fn, compute_metrics, pred_lr, meta_lr,
                 user_batch_size, n_outer_it):
        self.users = users
        self.num_users = len(users)
        self.user_col_name = user_col_name
        self.seed = seed
        self.output_dir = output_dir
        self.prediction_model = prediction_model
        self.user_embed_model = user_embed_model
        self.es_callback_meta = EarlyStopping(3, 0.0)
        set_seed(self.seed)
        self.loss_fn = pred_loss_fn

        # setup data
        self.init_user_embeds = init_user_embeds
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
        # get validation data for each user
        self.val_data_per_user = [filter_helper_single_val(self.val_dataset, self.user_col_name, user)
                                  for user in self.users]

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
        # setup embed model
        self.user_embed_model = self.user_embed_model.to(self.device)

        # setup functions used
        self.cos_sim = nn.CosineSimilarity()

        # setup params
        self.meta_lr = meta_lr
        self.pred_lr = pred_lr
        self.n_outer_it = n_outer_it
        self.n_train_epochs = 20
        self.verbose = True
        self.user_batch_size = user_batch_size  # number of users to select for each round of inner loop training
        self.pred_batch_size = 32
        self.user_embed_batch_size = 32  # number of users to pass as inputs to user embed model
        self.eval_epochs_pred = 1
        self.early_stopping = False
        self.epsilon = 1e-5

        # init weight matrix
        self.W = None

    def train(self):
        start_time = time.time()
        meta_opt = optim.Adam(self.user_embed_model.parameters(), self.meta_lr)
        print("*** Running Outer Training Loop ***")
        mean_losses = np.empty(self.n_outer_it)
        user_losses = np.empty((self.n_outer_it, self.num_users))
        W_matrices = np.empty((self.n_outer_it, self.num_users, self.num_users))
        step_user_embeds = np.empty((self.n_outer_it, self.num_users, self.user_embed_model.n_outputs))
        prev_W = None
        self.user_embed_model.train()
        self.user_embed_model.zero_grad()
        for step in range(self.n_outer_it):
            init_time = time.time()
            meta_opt.zero_grad()
            user_embeds = self.get_user_embeddings(is_train=True)
            embed()
            step_user_embeds[step] = user_embeds.cpu().detach().numpy()
            if self.verbose:
                print("user embeds at step {}: {}".format(step, user_embeds))
            self.W = self.get_user_weight_matrix(user_embeds)
            embed()
            W_matrices[step] = self.W.cpu().detach().numpy()
            if self.verbose:
                print("W at step {} = {}".format(step, self.W))
            if prev_W is not None:
                W_diff_mean = torch.mean(torch.abs(prev_W - self.W))
                print("Mean difference with last W {}".format(W_diff_mean))
                if W_diff_mean <= self.epsilon:
                    print("Weight matrix has converged, stopping training")
                    break
            prev_W = self.W.clone()
            # sample batch of users to train prediction models for
            train_batch = np.random.choice(self.train_user_ids, self.user_batch_size, replace=False)
            if self.verbose:
                print("Selected users are {}".format(train_batch))
            # train prediction model for each user
            for idx, pred_user in enumerate(train_batch):
                if self.verbose:
                    print("Starting model fine-tuning for user {}".format(pred_user))
                # get weights for this user
                user_weights = self.W[pred_user]
                sample_weights = user_weights[self.train_user_ids_per_sample]
                loss = self.pred_model_train(pred_user, sample_weights, step)
                user_losses[step, pred_user] = loss
            mean_losses[step] = np.mean(user_losses[step])
            print("Mean loss at step {}: {}".format(step, np.mean(user_losses[step])))
            stop_training = self._maybe_save_evaluate(mean_losses[step])
            if stop_training:
                break
            embed()
            meta_opt.step()
            embed()
            self.state.global_step += 1
            print("Finished outer loop step {} in time {}".format(step, time.time() - init_time))
        print("Finished meta training in {} steps".format(self.state.global_step))
        metrics = speed_metrics("train", start_time,
                                num_samples=self.state.global_step * self.user_batch_size,
                                num_steps=self.state.global_step)
        self.state.log_history.append({**metrics, **{"step": self.state.global_step}})
        np.save(os.path.join(self.output_dir, "step_mean_losses.npy"), mean_losses)
        np.save(os.path.join(self.output_dir, "step_user_losses.npy"), user_losses)
        np.save(os.path.join(self.output_dir, "final_user_weight_matrix.npy"), self.W.cpu().detach().numpy())
        np.save(os.path.join(self.output_dir, "user_weight_matrices.npy"), W_matrices)
        np.save(os.path.join(self.output_dir, "user_embeds.npy"), step_user_embeds)
        return TrainOutput(self.state.global_step, mean_losses[self. state.global_step - 1], metrics)

    # def get_user_weight_matrix(self, user_embeds):
    #     num_users = len(user_embeds)
    #     lhs = user_embeds.repeat_interleave(num_users, dim=0)
    #     rhs = user_embeds.repeat(num_users, 1)
    #     W = self.cos_sim(lhs, rhs).reshape(num_users, num_users)
    #     W = 1 - W
    #     # NOTE: one potential problem is that the min value of cos_sim is -1 so the max value of cos_dist is 2
    #     # so the min value of e^-cos_dist ~= 0.2 (not zero!)
    #     W = torch.exp(-W)
    #     return W

    def get_user_weight_matrix(self, user_embeds):
        """
        Alternative version where we just use cosine similarity as is --> don't enforce that weights are in [0,1] range
        """
        num_users = len(user_embeds)
        lhs = user_embeds.repeat_interleave(num_users, dim=0)
        rhs = user_embeds.repeat(num_users, 1)
        W = self.cos_sim(lhs, rhs).reshape(num_users, num_users)
        embed()
        return W

    def get_user_embeddings(self, is_train=True):
        user_embed_dataloader = self.get_user_embed_dataloader()
        num_batches = len(user_embed_dataloader)
        if is_train:
            self.user_embed_model.train()
        else:
            self.user_embed_model.eval()
        all_embeds = torch.zeros(self.num_users, self.user_embed_model.n_outputs).to(device=self.device)
        for i, batch in enumerate(user_embed_dataloader):
            start = i * self.user_embed_batch_size
            end = start + self.user_embed_batch_size
            if i == num_batches - 1:
                end = self.num_users
            batch = batch.to(device=self.device).float()
            embeds = self.user_embed_model(batch)
            all_embeds[start:end] = embeds
        embed()
        return all_embeds

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
            user_weights = self.W[user_id]
            sample_weights = user_weights[self.train_user_ids_per_sample]
            loss = self.pred_model_train(user_id, sample_weights, outer_step="eval")
            val_metrics = self.pred_model_evaluate(user_data, metric_key_prefix=metric_key_prefix)
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
        train_dataloader = self.get_train_dataloader()
        with higher.innerloop_ctx(self.prediction_model, pred_opt) as (fmodel, diffopt):
            for epoch in range(self.n_train_epochs):
                epoch_loss = 0
                for step, inputs in enumerate(train_dataloader):
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
                    embed()
                    diffopt.step(loss)
                    embed()
                    trainer_state_pred.global_step += 1
                    trainer_state_pred.log_history.append({"loss": loss.item(), "epoch": round(epoch, 2),
                                                           "step": trainer_state_pred.global_step})
                    epoch_loss += loss.item()
                epoch_loss /= len(train_dataloader)
                if epoch % self.eval_epochs_pred == 0:
                    print("copying params from fmodel to prediction model")
                    self.prediction_model.load_state_dict(fmodel.state_dict())
                    metrics = self.pred_model_evaluate(self.val_data_per_user[pred_user])
                    metrics["epoch"] = round(epoch, 2)
                    metrics["step"] = trainer_state_pred.global_step
                    metrics["mean_train_loss_per_step"] = epoch_loss
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
            print("Completed pred model training in {} epochs.".format(epoch))
            embed()
            # save weights to prediction model
            self.prediction_model.load_state_dict(fmodel.state_dict())
            # save trainer state
            save_dir = os.path.join(self.output_dir, "pred_model_results")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, "user_{}_step_{}_trainer_state.json".format(pred_user, outer_step))
            trainer_state_pred.save_to_json(save_path)
            if str(outer_step) == "eval":  # don't need to do meta learning validation loss backprop step
                return epoch_loss
            print("Evaluating Final Trained Pred Model for User {} at outer step {}".format(pred_user, outer_step))
            eval_dataloader = self.get_eval_dataloader(self.val_data_per_user[pred_user], self.val_collator)
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
            embed()
            print("Final Eval Loss for User {} at Outer step {}: {}".format(pred_user, outer_step, loss))
            loss.backward(retain_graph=True)
            embed()
            return loss.detach().item()

    def pred_model_evaluate(self, pred_dataset, metric_key_prefix="eval"):
        eval_dataloader = self.get_eval_dataloader(pred_dataset, self.val_collator)
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

        embed()
        num_samples = len(pred_dataset)
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

    def get_user_embed_dataloader(self):
        user_embed_sampler = SequentialSampler(self.init_user_embeds)
        return DataLoader(
            self.init_user_embeds,
            batch_size=self.user_embed_batch_size,
            sampler=user_embed_sampler,
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
        # save history so far
        self.save_state()
        stop_training = False
        if self.early_stopping:
            stop_training = self.es_callback_meta.check_metric_value(metrics, self.state.best_metric,
                                                                     False, "eval_loss")
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
