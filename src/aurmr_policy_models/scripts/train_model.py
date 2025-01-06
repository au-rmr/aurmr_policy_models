
import os

from aurmr_policy_models.utils.config_utils import load_config, instantiate, save_config
from aurmr_policy_models.utils.runtime_utils import set_random_seeds
# @hydra.main(version_base=None, config_path='../../../conf', config_name="config")
# @load_config
def main():
    cfg = load_config()
    set_random_seeds(cfg.seed)

    trainer = instantiate(cfg.trainer)
    trainer.train()

    save_config(cfg, os.path.join(cfg.trainer.output_dir, "config.yaml"))

if __name__ == "__main__":
    main()

# import argparse
# from ppo_trainer import PPOTrainer
# from your_model import YourModel  # Placeholder for your model class
# import torch

# def main(cfg):
#     pass

# def train_with_trainer(trainer_class, dataset_path, epochs, batch_size):
#     model = YourModel()  # Instantiate your model
    

#     trainer = trainer_class(model, optimizer, dataset_path, batch_size, epochs)
#     trainer.train()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Train a model using a specified trainer.")
#     parser.add_argument('--trainer', type=str, default='PPOTrainer', help="Class name of the trainer to use.")
#     parser.add_argument('--dataset_path', type=str, default='data.hdf5', help="Path to the dataset.")
#     parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
#     parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
#     parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate.")

#     args = parser.parse_args()

#     # Map trainer name to trainer class (this could be dynamically loaded if preferred)
#     trainers = {
#         'PPOTrainer': PPOTrainer
#     }

#     trainer_class = trainers.get(args.trainer)
#     if trainer_class is None:
#         raise ValueError(f"Unknown trainer class: {args.trainer}")

#     train_with_trainer(trainer_class, args.dataset_path, args.epochs, args.batch_size, args.learning_rate)
