import hydra

from aurmr_policy_models.utils.config_utils import load_config, initialize_env_from_config

# @hydra.main(version_base=None, config_path='../../../conf', config_name="config")
# @load_config
def main():
    cfg = load_config()
    env = initialize_env_from_config(cfg)

    print(env)

if __name__ == "__main__":
    main()