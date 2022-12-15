"""
Script for running text classification experiment.
"""

import logging
import os
import time
from copy import deepcopy

from IPython import embed
import numpy as np
import pandas as pd
import transformers

# select GPU before importing anything else
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
assert torch.cuda.is_available(), "selected gpus, but cuda not available"

from common.factories import get_dataset, get_model, get_trainer, get_embed_model, get_cluster_model
from dataset.dataset_utils import filter_helper_single_val
from runners.runner_utils import parse_args, loss_to_perplexity, save_args
from sample_weights.fixed_weights import get_fixed_sample_weights

logger = logging.getLogger(__name__)

deterministic = False
if deterministic:
    print("WARNING: executing in deterministic mode")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
else:
    print("not using deterministic mode")


class ExpRunner:
    def __init__(self, exp_args, model_args, data_args, training_args):
        # general experiment setup
        self.exp_args = exp_args
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.dataset = None
        self.model = None
        self.train_data_collator = None
        self.trainer = None
        # setup logger
        self.setup_logger()
        # if deterministic mode, handle this
        if self.exp_args.deterministic_mode:
            print("running in deterministic mode")
            torch.use_deterministic_algorithms(True)

    def load_and_prepare_data(self):
        # load data
        self.dataset = get_dataset(self.data_args)
        # embed data
        if self.exp_args.embed_data:
            embed_model = get_embed_model(self.exp_args)
            self.dataset.embed_data(embed_model, self.exp_args.embed_path)
            if self.exp_args.save_data_with_embeds_dir is not None:
                print("saving dataset with embeddings")
                self.dataset.save_data(os.path.join(self.data_args.data_dir, self.exp_args.save_data_with_embeds_dir))
        # cluster data
        if self.exp_args.cluster_data:
            cluster_model = get_cluster_model(self.exp_args)
            self.dataset.cluster_data(cluster_model, self.exp_args.cluster_path, self.exp_args.cluster_train_data)
        # add user embeds
        if self.data_args.add_user_embeds:
            embed_method = self.data_args.user_embed_method
            print("adding user embeddings with method {}".format(embed_method))
            self.dataset.add_user_embeddings(method=embed_method)
        self.dataset.report_metrics(self.data_args.report_columns_discrete, self.data_args.report_columns_cont)

    def filter_data(self):
        # filter based on other criteria (apply this after personalization so restricting # of samples happens 2nd)
        self.dataset.filter_data(self.data_args.max_train_samples, self.data_args.select_random_train,
                                 self.data_args.filter_train_by, self.data_args.train_keep_vals,
                                 self.data_args.max_val_samples, self.data_args.select_random_val,
                                 self.data_args.filter_val_by, self.data_args.val_keep_vals,
                                 self.data_args.max_test_samples, self.data_args.select_random_test,
                                 self.data_args.filter_test_by, self.data_args.test_keep_vals)
        print("dataset after filtering")
        self.dataset.report_metrics(self.data_args.report_columns_discrete, self.data_args.report_columns_cont)

    def setup_exp(self):
        sample_weights = None
        # add sample fixed sample weights if using fixed sample weight scheme
        if self.exp_args.sample_weight_scheme is not None \
                and self.exp_args.sample_weight_scheme.split('_')[0] == "fixed":
            sample_weights = get_fixed_sample_weights(self.dataset.train_data,
                                                      self.exp_args.sample_weight_scheme,
                                                      self.exp_args.personalize_col,
                                                      self.exp_args.personalize_target,
                                                      self.exp_args.sample_weight_vals)
        self.model = get_model(self.model_args, self.dataset)
        self.trainer = get_trainer(self.exp_args, self.training_args, self.model_args,
                                   self.model, self.dataset, sample_weights)

    def setup_logger(self):
        # Setup logging
        if self.exp_args.should_log:
            transformers.utils.logging.set_verbosity_info()
            transformers.utils.logging.enable_default_handler()
            transformers.utils.logging.enable_explicit_format()

    def train(self):
        init_time = time.time()
        train_result = self.trainer.train()
        train_time = time.time() - init_time
        self.trainer.save_model()
        metrics = train_result.metrics
        max_train_samples = (
            self.data_args.max_train_samples if self.data_args.max_train_samples is not None
            else len(self.dataset.train_data)
        )
        metrics["train_samples"] = min(max_train_samples, len(self.dataset.train_data))
        # note: this is the mean training loss across all epochs - not the loss of the model after training is complete.
        metrics["training_loss"] = train_result.training_loss
        metrics["train_time"] = train_time
        # this saves to files train_results.json and all_results.json in output dir
        self.trainer.save_metrics("train", metrics)
        # this saves file trainer_state.json
        self.trainer.save_state()

    def validate(self):
        init_time = time.time()
        metrics = self.trainer.evaluate()
        val_time = time.time() - init_time
        max_val_samples = (
            self.data_args.max_val_samples if self.data_args.max_val_samples is not None
            else len(self.dataset.val_data)
        )
        metrics["val_samples"] = min(max_val_samples, len(self.dataset.val_data))
        metrics["val_time"] = val_time
        self.trainer.save_metrics("val", metrics)

    def test(self):
        init_time = time.time()
        if self.exp_args.generate_predictions:
            outputs = self.trainer.predict(test_dataset=self.dataset.test_data)
            metrics = outputs.metrics
            # save preds
            np.save(os.path.join(self.training_args.output_dir, "predictions.npy"), outputs.predictions)
        else:
            metrics = self.trainer.evaluate(eval_dataset=self.dataset.test_data, metric_key_prefix="test")
        test_time = time.time() - init_time
        max_test_samples = (
            self.data_args.max_test_samples if self.data_args.max_test_samples is not None
            else len(self.dataset.test_data)
        )
        metrics["test_samples"] = min(max_test_samples, len(self.dataset.test_data))
        metrics["test_time"] = test_time
        self.trainer.save_metrics("test", metrics)

    def get_per_sample_losses(self):
        init_time = time.time()
        data_loader = self.trainer.get_eval_dataloader(self.dataset.test_data)
        sample_losses = self.trainer.evaluation_loop_sample_losses(data_loader)
        test_time = time.time() - init_time
        print("finished computing sample metrics in time: {}".format(test_time))
        df = pd.DataFrame(data={'id': self.dataset.test_data['id'], 'loss': sample_losses})
        df.to_csv(os.path.join(self.training_args.output_dir, "sample_losses.csv"))

    def eval_by_group(self, group_name):
        metrics = dict()
        feat_vals = set(self.dataset.test_data[group_name])
        if self.exp_args.group_vals is not None:
            with open(self.exp_args.group_vals, 'r') as f:
                feat_vals = f.read().splitlines()
        losses = []
        for val in feat_vals:
            feat_dataset = filter_helper_single_val(self.dataset.test_data, filter_by=group_name, keep_val=val)
            feat_metrics = self.trainer.evaluate(eval_dataset=feat_dataset, metric_key_prefix=val)
            feat_metrics["{}_samples".format(val)] = len(feat_dataset)
            losses.append(feat_metrics["{}_loss".format(val)])
            metrics.update(feat_metrics)
        # get average values
        metrics["mean_loss"] = np.mean(losses)
        metrics["std_loss"] = np.std(losses)
        self.trainer.save_metrics("{}_test".format(group_name), metrics)

    def run_single_seed(self):
        if self.exp_args.sample_weight_scheme is not None:  # sweep over different sample weight parameters
            self._loop_over_sample_weight_params()
        else:
            self.setup_exp()
            self.run_single_exp()

    def run_single_exp(self):
        if self.training_args.do_train:
            print("Starting model training.")
            self.train()
        if self.training_args.do_eval:
            print("Starting model evaluation.")
            self.validate()
        # may need to switch data collator for testing if using HFTrainer
        if self.data_args.fix_test_labels and self.exp_args.trainer == "HFTrainer":
            self.trainer.data_collator = self.dataset.get_data_collator(is_train=False)
        if self.training_args.do_predict:
            print("Starting model testing (general).")
            self.test()
        if self.exp_args.eval_by_group is not None:
            print("Analyzing test performance by {}.".format(self.exp_args.eval_by_group))
            self.eval_by_group(self.exp_args.eval_by_group)

    def run(self):
        """
        Generic training set-up
        """
        self.load_and_prepare_data()
        self.filter_data()
        self._loop_through_seeds()

    def run_personalized(self, select_file=None, select_n_vals=None, save_selected=None, skip_completed=True):
        """
        Train and eval personalized/person-specific models.
        :param select_file: if not None, select people listed in this file only
        :param select_n_vals: how many different people to select. If none, will loop through all.
        :param save_selected: if provided, save list of selected people to file
        :param skip_completed: if True, will skip people that we have already trained personalized models for
        """
        self.load_and_prepare_data()
        full_dataset = deepcopy(self.dataset)
        people = set(self.dataset.test_data[self.exp_args.personalize_col])
        if select_file is not None:
            with open(select_file, 'r') as f:
                people = f.read().splitlines()
        if select_n_vals is None:
            select_people = people
        else:
            select_people = np.random.choice(list(people), select_n_vals, replace=False)
        if save_selected is not None:
            with open(save_selected, 'w+') as f:
                for person in select_people:
                    f.write("{}\n".format(person))
        base_output_dir = self.training_args.output_dir

        print("Starting personalized model experiment")
        print("Personalize column is {}".format(self.exp_args.personalize_col))
        for idx, person in enumerate(select_people):
            if skip_completed:
                if os.path.exists(os.path.join(base_output_dir, str(person))):
                    print("skipping {}={} (already completed)".format(self.exp_args.personalize_col, person))
                    continue
            # filter data to only be from this person
            print("selecting examples from {}".format(person))
            start_time = time.time()
            # filter train data only if personalize strategy == train_local
            if self.exp_args.personalize_strategy == "train_local":
                self.dataset.train_data = filter_helper_single_val(self.dataset.train_data, self.exp_args.personalize_col, person)
            self.dataset.val_data = filter_helper_single_val(self.dataset.val_data, self.exp_args.personalize_col, person)
            self.dataset.test_data = filter_helper_single_val(self.dataset.test_data, self.exp_args.personalize_col, person)
            print('filtered examples in time {}'.format(time.time() - start_time))
            print("Train count {} Val Count {} Test count {}".format(len(self.dataset.train_data),
                                                                     len(self.dataset.val_data),
                                                                     len(self.dataset.test_data)))
            # apply other filtering steps
            self.filter_data()
            # adjust output dir
            self.training_args.output_dir = os.path.join(base_output_dir, str(person))
            # adjust personalization target
            self.exp_args.personalize_target = person
            # run experiments
            start_time = time.time()
            print("Starting experiment for {}={} ({}/{})".format(self.exp_args.personalize_col, person, idx + 1, len(select_people)))
            self._loop_through_seeds()
            print("Finished experiment in time {}\n".format(time.time() - start_time))
            # reset dataset to full dataset
            self.dataset = deepcopy(full_dataset)

    def run_pretrained_models(self, levels):
        """
        Train (optionally) and evaluate pre-trained models located in the directory at model_name_or_path model_arg.
        Recursively look through this directory for models to evaluate.
        :param levels: list of the subdirectory levels to loop through when looking for models.
        """
        self.load_and_prepare_data()
        self.filter_data()
        # loop through models
        base_model_dir = self.model_args.model_name_or_path
        base_output_dir = self.training_args.output_dir
        self._loop_through_pretrained_models(base_model_dir, base_output_dir, levels)

    def _loop_through_pretrained_models(self, base_model_dir, base_output_dir, levels):
        _sub_model_dirs = os.listdir(base_model_dir)
        sub_model_dirs = [os.path.join(base_model_dir, sub_dir) for sub_dir in _sub_model_dirs
                          if os.path.isdir(os.path.join(base_model_dir, sub_dir))]
        # make corresponding output dirs
        sub_output_dirs = [os.path.join(base_output_dir, sub_dir) for sub_dir in _sub_model_dirs
                           if os.path.isdir(os.path.join(base_model_dir, sub_dir))]
        for sub_model_dir, sub_output_dir in zip(sub_model_dirs, sub_output_dirs):
            if len(levels) == 1:  # if last level find model and evaluate experiment
                self.model_args.model_name_or_path = os.path.join(sub_model_dir, "pytorch_model.bin")
                if self.model_args.config_name_or_path is not None:
                    self.model_args.config_name_or_path = os.path.join(sub_model_dir, "config.json")
                # update output dir
                self.training_args.output_dir = sub_output_dir
                # run exp with this model
                print("using model at path {}".format(self.model_args.model_name_or_path))
                print("saving results to {}".format(self.training_args.output_dir))
                self._loop_through_seeds()
            else:
                self._loop_through_pretrained_models(sub_model_dir, sub_output_dir, levels[1:])

    def _loop_through_seeds(self):
        # run for each seed
        seed = self.exp_args.start_seed
        base_output_dir = self.training_args.output_dir
        for i in range(self.exp_args.num_runs):
            print("Starting experiment run with seed {}".format(seed))
            self.training_args.seed = seed
            self.training_args.output_dir = os.path.join(base_output_dir, str(seed))
            self.run_single_seed()
            seed += 1

    def _loop_over_sample_weight_params(self):
        base_output_dir = self.training_args.output_dir
        for alpha in self.exp_args.sample_weight_alphas:
            print("Running with sample weight alpha={}".format(alpha))
            self.exp_args.sample_weight_vals = [alpha]
            # edit output dir to reflect chosen alpha
            self.training_args.output_dir = os.path.join(base_output_dir, str(alpha))
            self.setup_exp()
            self.run_single_exp()


def main():
    mr_args, exp_args, model_args, data_args, training_args = parse_args()
    save_args(training_args.output_dir)
    exp_runner = ExpRunner(exp_args, model_args, data_args, training_args)
    if exp_args.exp_type == "generic":
        exp_runner.run()
    elif exp_args.exp_type == "personalized":
        exp_runner.run_personalized(exp_args.person_select_file, exp_args.select_n_persons, exp_args.save_selected,
                                    exp_args.skip_completed_people)
    elif exp_args.exp_type == "pretrained_models":
        exp_runner.run_pretrained_models(exp_args.model_levels)
    else:
        print("ERROR: unrecognized experiment type {}".format(exp_args.exp_type))
        print("Exiting...")
        exit(1)


if __name__ == "__main__":
    main()
