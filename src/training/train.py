import argparse

import yaml

from src.envs.torcs_env import TorcsEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    env = TorcsEnv(vision=config.get("vision", True))

    algo = config.get("algorithm", "PPO")
    if algo == "PPO":
        from src.agents.ppo_agent import PPOAgent

        agent = PPOAgent(env, **config.get("agent_kwargs", {}))
    elif algo == "SAC":
        from src.agents.sac_agent import SACAgent

        agent = SACAgent(env, **config.get("agent_kwargs", {}))
    else:
        raise ValueError("Algoritmo desconhecido")

    agent.train(total_timesteps=config.get("total_timesteps", 1_000_000))


if __name__ == "__main__":
    main()
