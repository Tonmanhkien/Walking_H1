import sys
import os
import argparse

# --- Boilerplate to find modules ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# -----------------------------------

from isaaclab.app import AppLauncher

# Launch App
parser = argparse.ArgumentParser(description="Train PPO agent for H1 robot.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports must happen AFTER app launch
import torch 
import gymnasium as gym
from ppo import PPOAgent
from env.h1_env import H1RoughEnvCfg

class IsaacWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_obs = env.observation_space['policy'].shape[1]
        self.num_actions = env.action_space.shape[1]

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs['policy']
    
    def step(self, action):
        obs, r , terminated, truncated, info = self.env.step(action)
        return obs['policy'], r, terminated, truncated, info
    
def main():
    # --- CONFIGURATIONS ---
    num_envs = 4096
    num_steps = 24
    total_timesteps = 50000000
    device = 'cuda:0'
    
    # --- CHECKPOINT SETTINGS ---
    # The specific path you asked for
    ckpt_dir = "/home/kien/Walking_H1/ckpts/unitree_h1"
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[INFO] Checkpoints will be saved to: {ckpt_dir}")

    env_cfg = H1RoughEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = device

    env = gym.make("Isaac-Velocity-Rough-H1-v0", cfg=env_cfg)
    env = IsaacWrapper(env)

    agent = PPOAgent(
        num_obs=env.num_obs,
        num_actions=env.num_actions,
        batch_size=num_envs*num_steps,
        device=device
    )

    # --- BUFFERS (CPU) ---
    obs_buf = torch.zeros((num_steps, num_envs, env.num_obs))
    actions_buf = torch.zeros((num_steps, num_envs, env.num_actions)) 
    logprobs_buf = torch.zeros((num_steps, num_envs))
    rewards_buf = torch.zeros((num_steps, num_envs))
    dones_buf = torch.zeros((num_steps, num_envs))
    values_buf = torch.zeros((num_steps, num_envs))

    print("Start training!")

    next_obs = env.reset() 
    next_done = torch.zeros(num_envs, device=device) 
    num_updates = total_timesteps // (num_envs * num_steps)

    for update in range(1, num_updates + 1):
        
        # A. ROLLOUT PHASE (Collection)
        for step in range(num_steps):
            obs_buf[step] = next_obs.cpu()
            dones_buf[step] = next_done.cpu()

            with torch.no_grad():
                action, logprob, _, value = agent.get_action(next_obs)
                values_buf[step] = value.flatten().cpu()
                actions_buf[step] = action.cpu()
                logprobs_buf[step] = logprob.cpu()

            next_obs, reward, next_done, _, _ = env.step(action)
            rewards_buf[step] = reward.view(-1).cpu()
    
        # Pack memory
        memory = {
            'obs': obs_buf,
            'actions': actions_buf,
            'logprobs': logprobs_buf,
            'rewards': rewards_buf,
            'dones': dones_buf,
            'values': values_buf,
            'next_obs': next_obs,   
            'next_done': next_done  
        }
        
        # Update 
        metrics = agent.update(memory)

        # Logging
        if update % 5 == 0:
            print(f"Iter {update}/{num_updates} | Reward: {rewards_buf.mean().item():.3f} | Loss: {metrics['loss']:.3f}")

        # --- SAVE CHECKPOINT LOGIC ---
        # Save every 50 updates (approx every few minutes)
        if update % 50 == 0:
            save_path = os.path.join(ckpt_dir, f"model_{update}.pt")
            torch.save(agent.state_dict(), save_path)
            print(f"Saved checkpoint: {save_path}")

    # Final Save
    final_path = os.path.join(ckpt_dir, "model_final.pt")
    torch.save(agent.state_dict(), final_path)
    print(f"Training Finished! Model saved to {final_path}")
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()