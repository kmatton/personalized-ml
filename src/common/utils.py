"""
General helper functions
"""
import os
import random

import torch
import numpy as np


def get_folders_from_path(path):
    """
    :param path: directory path
    :return: list of each component of path
    """
    folders = []
    while 1:
        path, folder = os.path.split(path)
        if folder != "":
            folders.append(folder)
        elif path != "":
            folders.append(path)
            break
    folders.reverse()
    return folders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
