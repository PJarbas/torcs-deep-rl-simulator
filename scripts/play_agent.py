import argparse
from envs.torcs_env import TorcsEnv
from agents.ppo_agent import PPOAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the trained model")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--render", type=bool, default=True, help="Whether to render the environment")
    args = parser.parse_args()

    # Create environment
    env = TorcsEnv(vision=True, rendering=args.render)
    
    # Create and load agent
    agent = PPOAgent(env)
    agent.load(args.model)
    
    try:
        for episode in range(args.episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            step = 0
            
            print(f"\nEpisódio {episode + 1}/{args.episodes}")
            
            while not done:
                action = agent.act(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                step += 1
                
                if step % 10 == 0:
                    print(f"Step: {step}, Reward: {total_reward:.2f}", end="\r")
            
            print(f"\nEpisódio {episode + 1} finalizado - Total Steps: {step}, Total Reward: {total_reward:.2f}")
    
    finally:
        env.close()

if __name__ == "__main__":
    main()
