import sys
import copy
from functools import wraps
from collections import namedtuple

import hydra
from omegaconf import OmegaConf


experiment_parts = ('agent', 'env', 'model', 'trainer', 'train_data', 'val_data')
Experiment = namedtuple('Experiment', experiment_parts, defaults=(None,) * len(experiment_parts))


OmegaConf.register_new_resolver("eval", eval, replace=True)
# def load_config(func):
#     @wraps(func)
#     @hydra.main(version_base=None, config_path='../../../conf', config_name="config")
#     def wrapper(cfg):
#         print(OmegaConf.to_yaml(cfg))
#         return func(cfg)
#     return wrapper

def load_config():
    overrides = sys.argv[1:]  # Skip the script name in sys.argv[0]
    hydra.initialize(version_base=None, config_path='../../../conf')
    cfg = hydra.compose(config_name="config", overrides=overrides)
    print(OmegaConf.to_yaml(cfg))
    return cfg

def load_config_from_file(filename):
    return OmegaConf.load(filename)

def save_config(cfg, filename):
    OmegaConf.save(config=cfg, f=filename)

def instantiate(config, *args, **kwargs):
    obj = hydra.utils.instantiate(config, *args, **kwargs)
    if hasattr(obj, '__configure__'):
        obj.__configure__(config)
    return obj


def intitialize_experiment_from_config(cfg):
    exp = Experiment()
    if cfg.env:
        exp.env = hydra.utils.instantiate(cfg.env)
    
    if cfg.agent:
        exp.agent = hydra.utils.instantiate(cfg.agent, exp.env)

def initialize_agent_from_config(cfg, env):
    # agent_args = OmegaConf.to_container(cfg.agent, resolve=True)
    # del agent_args['agent_class']
    # agent_args['env'] = env
    # agent = initialize_class(cfg.agent.agent_class, agent_args)
    return hydra.utils.instantiate(cfg.agent, env)

def initialize_env_from_config(cfg):
    return hydra.utils.instantiate(cfg.env)
    # env_args = OmegaConf.to_container(cfg.env, resolve=True)
    # del env_args['env_class']
    # env = initialize_class(cfg.env.env_class, env_args)
    # return env

# def load_yaml_file(filename):
#     """Load YAML data from a file."""
#     with open(filename, 'r') as file:
#         return yaml.safe_load(file)

# def recursive_merge(d1, d2):
#     """Recursively merge two dictionaries, with values from the second dict overriding the first."""
#     for key in d2:
#         if key in d1 and isinstance(d1[key], dict) and isinstance(d2[key], dict):
#             recursive_merge(d1[key], d2[key])
#         else:
#             d1[key] = d2[key]

# def function_with_kwargs(**kwargs):
#     """A function that accepts any number of keyword arguments."""
#     for key, value in kwargs.items():
#         print(f"{key}: {value}")

# def get_config(overrides_file):
#     # Path to the defaults file relative to this script
#     script_dir = os.path.dirname(__file__)
#     defaults_file = os.path.join(script_dir, '../../../configs/defaults.yaml')
    
#     defaults = load_yaml_file(defaults_file)
#     overrides = load_yaml_file(overrides_file)
#     recursive_merge(defaults, overrides)
    
#     # by default the experiment name is the name of the config file
#     if 'experiment_name' not in defaults:
#         defaults['experiment_name'] = os.path.splitext(os.path.basename(overrides_file))[0].replace('.yaml', '')

#     return defaults