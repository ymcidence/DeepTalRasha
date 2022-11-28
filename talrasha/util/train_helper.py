import os
import tensorflow as tf
from time import gmtime, strftime


def make_training_folder(root_path, l1_name, l2_name=''):
    """

    Args:
        root_path:
        l1_name: Level-1 experiment name, usually the task name
        l2_name: Level-2 sub-experiment name, usually a run with a specific configuration.
            Hence, different runs/configurations can be compared under the same l1 folder.

    Returns:
        a summary_path for tensorboard logging and a save_path for model saving
    """

    time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())

    result_path = os.path.join(root_path, 'result', l1_name)
    save_path = os.path.join(result_path, 'model', l2_name + '_' + time_string)
    summary_path = os.path.join(result_path, 'log', l2_name + '_' + time_string)

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return summary_path, save_path
