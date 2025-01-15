# AURMR Policy Models

Tools and framework for training continuous visiomotor policies as part of Amazon-UW Robotic Manipulation Research (AURMR).

Currently supported models include Diffusion Policy.

Currently supported training algorithms include BC, DPPO, RLPD, and I-DQL.

## Installation

1. Clone the repository
```
git clone git@github.com:au-rmr/aurmr_policy_models.git
cd aurmr_policy_models
```

2. Create environment and install package
```
conda create -n apm python=3.8 -y
conda activate apm
pip install -e .
```

3. Configure data root path
```
export AURMR_POLICY_MODELS_DATA_ROOT=/data/aurmr_policy_models/
```

## Basic Usage Guide

All experiments have reproducable configurations under `conf/experiments`.

### Collect data from expert planner

```
python -m aurmr_policy_models.scripts.collect_agent_data \
    experiment=point_mass_expert_agent \
    collection.num_episodes=5000 \
    collection.collection_name="point_mass_expert_5000"
```

### Pre-train with collected dataset

```
python -m aurmr_policy_models.scripts.train_model \
    experiment=point_mass_diffusion_pretrain \
    train_dataset.file_paths='["/data/aurmr_policy_models/collections/point_mass_expert_5000.hdf5"]' \
    trainer.output_dir="/data/aurmr_policy_models/training_runs/point_mass_iter0_expert5k/"
```

### Evaluate pre-trained model
```
python -m aurmr_policy_models.scripts.evaluate_agent \
    experiment=point_mass_diffusion \
    model.network_path="/data/aurmr_policy_models/training_runs/point_mass_iter0_expert5k/final_model.pt" \
    env.render=True
```

### Fine-tune with PPO

```
python -m aurmr_policy_models.scripts.train_model \
    experiment=point_mass_diffusion_ppo \
    model.network_path="/data/aurmr_policy_models/training_runs/point_mass_iter0_expert10k/final_model.pt"
```