import argparse
import gym

from aurmr_policy_models.utils.config_utils import load_config, instantiate

def main():
    cfg = load_config()
    env = instantiate(cfg.env)
    agent = instantiate(cfg.agent, env)
    run_evaluation(agent, env, cfg.eval.num_episodes, cfg.eval.max_steps)
    env.close()

def run_evaluation(agent, env, num_episodes, max_steps):
    """
    Run the agent against the environment for a given number of episodes and max steps per episode.
    """
    total_rewards = []
    for episode in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            if env.render_enabled:
                env.render()
            action = agent.select_action(observation)[0]
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    average_reward = sum(total_rewards) / num_episodes
    print(f"Average Reward: {average_reward}")

if __name__ == '__main__':
    main()

