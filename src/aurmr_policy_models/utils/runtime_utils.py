import importlib
import random
import numpy as np
import torch
import time

# # initialize a given class (given by string) with the given args and kwargs
# def initialize_class(class_name, class_args):
#     """
#     Initialize a class with the given name using the provided arguments.
    
#     Args:
#         class_name (str): Name of the class to initialize.
#         class_args: Keyword arguments to pass to the class constructor
#     """

#     # Import the class from the models directory
#     module_name, class_name = class_name.rsplit('.', 1)
#     module = importlib.import_module(module_name)
#     class_ = getattr(module, class_name)
#     return class_(**class_args)

def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class Timer:
    """
    Simple timer from https://github.com/jannerm/diffuser/blob/main/diffuser/utils/timer.py
    """
    def __init__(self):
        self._start = time.time()

    def __call__(self, reset=True):
        now = time.time()
        diff = now - self._start
        if reset:
            self._start = now
        return diff