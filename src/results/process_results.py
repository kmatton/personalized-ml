"""
Files for aggregating and analyzing experiment results.
"""
import json
import os

import numpy as np
import pandas as pd

from common.utils import get_folders_from_path


class ResultProcessor:
    def __init__(self, base_result_dir, levels, verbose=False):
        self.base_result_dir = base_result_dir
        self.levels = levels
        self.verbose = verbose
        self.train_hist_df = None
        self.results_df = None
        self.preds_df = None
        self.group_results_df = None
        self._load_results(self.base_result_dir, self.levels)

    def get_group_results(self, group_name):
        # if results are named by their sub-population value (e.g., username) make the group another column so that
        # results per group can be processed more easily
        # read names of group results from file
        result_dirs = self._get_full_paths(self.base_result_dir, self.levels, [])
        column_names = set()
        for result_dir in result_dirs:
            results_path = os.path.join(result_dir, "{}_test_results.json".format(group_name))
            assert os.path.exists(results_path), "did not find group result file at path {}".format(results_path)
            print("looking for group result column names from file at: {}".format(results_path))
            results_dict = json.load(open(results_path))
            del results_dict["mean_loss"]
            del results_dict["std_loss"]
            column_names.update(results_dict.keys())
        new_data = {}
        num_runs = len(self.results_df)
        for col in self.results_df.columns:
            if col in column_names:
                group_val = col.split("_")[0]
                new_col = "_".join(col.split("_")[1:])
                if group_val not in new_data:
                    new_data[group_val] = {}
                for i in range(num_runs):
                    if i not in new_data[group_val]:
                        new_data[group_val][i] = {}
                    new_data[group_val][i][new_col] = self.results_df[col].values[i]
        # consolidate dict
        new_result_list = []
        for group_val, run_dict in new_data.items():
            for run_num, result_dict in run_dict.items():
                _dict = {group_name: group_val, "seed": run_num}
                for result_name, result_val in result_dict.items():
                    _dict[result_name] = result_val
                new_result_list.append(_dict)
        new_results_df = pd.DataFrame(new_result_list)
        self.group_results_df = new_results_df
        return self.group_results_df

    def _add_result(self, trainer_state_dict, results_dict, preds_arr, result_dir, result_num):
        result_path_parts = get_folders_from_path(result_dir)
        # get parts that correspond to level values
        level_values = result_path_parts[-len(self.levels):]
        _train_hist_df = None
        if trainer_state_dict is not None:
            _train_hist_df = _get_train_and_val_history(trainer_state_dict)
        _preds_df = None
        if preds_arr is not None:  # convert to df
            cols = ["pred"]
            if len(preds_arr.shape) > 1 and preds_arr.shape[1] > 1:
                cols = ["pred_cls_{}".format(i + 1) for i in range(preds_arr.shape[1])]
            _preds_df = pd.DataFrame(preds_arr, index=np.arange(len(preds_arr)), columns=cols)
            # add column with index of sample
            _preds_df["sample_idx"] = _preds_df.index
        for (level_val, level) in zip(level_values, self.levels):
            # add level values to results
            if _train_hist_df is not None:
                _train_hist_df[level] = level_val
            if results_dict is not None:
                results_dict[level] = level_val
            if _preds_df is not None:
                _preds_df[level] = level_val
        # transform results_dict to df
        _results_df = None
        if results_dict is not None:
            _results_df = pd.DataFrame(results_dict, index=[result_num])
        # add to aggregate result dfs
        if self.train_hist_df is None:
            self.train_hist_df = _train_hist_df
        else:
            self.train_hist_df = pd.concat([self.train_hist_df, _train_hist_df])
        if self.results_df is None:
            self.results_df = _results_df
        else:
            self.results_df = pd.concat([self.results_df, _results_df])
        if self.preds_df is None:
            self.preds_df = _preds_df
        else:
            self.preds_df = pd.concat([self.preds_df, _preds_df])

    def _load_results(self, base_dir, levels):
        curr_level = levels[0]
        sub_dirs = os.listdir(base_dir)
        sub_dirs = [os.path.join(base_dir, sub_dir) for sub_dir in sub_dirs]
        sub_dirs = [sub_dir for sub_dir in sub_dirs if os.path.isdir(sub_dir)]
        if self.verbose:
            print("Base dir {}: Found results for {} {}s".format(base_dir, len(sub_dirs), curr_level))
        result_num = 0
        for sub_dir in sub_dirs:
            if len(levels) == 1:  # if at last level collect results
                trainer_state_dict, results_dict, preds_arr = _read_result_dir(sub_dir)
                self._add_result(trainer_state_dict, results_dict, preds_arr, sub_dir, result_num)
                result_num += 1
            else:
                self._load_results(sub_dir, levels[1:])

    def _get_full_paths(self, base_dir, levels, paths):
        """
        Search through for bottom level directory so that there are some example files. (e.g., user_test_results.json)
        """
        sub_dirs = os.listdir(base_dir)
        sub_dirs = [os.path.join(base_dir, sub_dir) for sub_dir in sub_dirs]
        sub_dirs = [sub_dir for sub_dir in sub_dirs if os.path.isdir(sub_dir)]
        for sub_dir in sub_dirs:
            if len(levels) == 1:  # if at last level, add full path
                paths.append(sub_dir)
            else:
                paths = self._get_full_paths(sub_dir, levels[1:], paths)
        return paths


### Helper Functions ###

def _read_result_dir(result_dir):
    trainer_state_dict = None
    trainer_state_path = os.path.join(result_dir, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        trainer_state_dict = json.load(open(trainer_state_path))
    results_dict = None
    results_path = os.path.join(result_dir, "all_results.json")
    if os.path.exists(results_path):
        results_dict = json.load(open(results_path))
    preds_arr = None
    preds_path = os.path.join(result_dir, "predictions.npy")
    if os.path.exists(preds_path):
        preds_arr = np.load(preds_path, allow_pickle=True)
    return trainer_state_dict, results_dict, preds_arr


def _get_train_and_val_history(trainer_state):
    train_steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []
    for item in trainer_state['log_history']:
        if 'loss' in item.keys():
            train_steps.append(item['step'])
            train_loss.append(item['loss'])
        elif 'eval_loss' in item.keys():
            eval_steps.append(item['step'])
            eval_loss.append(item['eval_loss'])
    losses = train_loss + eval_loss
    steps_all = train_steps + eval_steps
    data = ["train"] * len(train_loss) + ["val"] * len(eval_loss)
    df = pd.DataFrame({"step": steps_all, "data": data, "loss": losses})
    return df
