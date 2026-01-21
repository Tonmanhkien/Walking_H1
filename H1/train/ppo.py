import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class PPOAgent(nn.Module):
    def __init__(self,
                 num_obs,
                 num_actions,
                 lr=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_coef=0.2,
                 ent_coef=0.01,
                 vf_coef=0.5,
                 max_grad_norm=1.0,
                 batch_size=2048,
                 mini_batch_size=1024,
                 num_epochs=10,
                 device='cuda:0'):
        
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.num_epochs = num_epochs

        # Switched to ELU activation (Better for locomotion than ReLU)
        self.actor = nn.Sequential(
            nn.Linear(num_obs, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, num_actions),
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(num_obs, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        ).to(self.device)

        self.log_std = nn.Parameter(torch.zeros(num_actions)).to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def get_action(self, x, action = None):
        action_mean = self.actor(x)
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(x)

        return action, log_prob, entropy, value.squeeze(-1)
    
    def get_value(self, x):
        return self.critic(x).squeeze(-1)
    
    def update(self, memory):
        # 1. Calculate GAE on CPU
        with torch.no_grad():
            next_value = self.get_value(memory['next_obs']).cpu() 
            
            advantages = torch.zeros_like(memory['rewards'])
            last_gae_lambda = 0
            num_steps = memory['rewards'].shape[0]

            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_non_terminal = 1.0 - memory['next_done'].cpu().float()
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - memory['dones'][t+1].float()
                    next_values = memory['values'][t+1]

                delta = memory['rewards'][t] + self.gamma * next_values * next_non_terminal - memory['values'][t]
                advantages[t] = last_gae_lambda = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda

            returns = advantages + memory['values']

        # Flatten buffers
        b_obs = memory['obs'].reshape((-1, memory['obs'].shape[2]))
        b_logprobs = memory['logprobs'].reshape(-1)
        b_actions = memory['actions'].reshape((-1, memory['actions'].shape[2]))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = memory['values'].reshape(-1)

        b_inds = torch.arange(self.batch_size)
        mean_v_loss = 0

        for epoch in range(self.num_epochs):
            b_inds = b_inds[torch.randperm(self.batch_size)]

            for start in range(0, self.batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_inds = b_inds[start:end]

                # Move mini-batch to GPU
                mb_obs = b_obs[mb_inds].to(self.device)
                mb_actions = b_actions[mb_inds].to(self.device)
                mb_logprobs = b_logprobs[mb_inds].to(self.device)
                mb_advantages = b_advantages[mb_inds].to(self.device)
                mb_returns = b_returns[mb_inds].to(self.device)

                _, newlogprob, entropy, newvalue = self.get_action(mb_obs, mb_actions)
                
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                # Normalize Advantages
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss (Clipped)
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = b_values[mb_inds].to(self.device) + torch.clamp(
                    newvalue - b_values[mb_inds].to(self.device),
                    -self.clip_coef,
                    self.clip_coef,
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                
                # --- CRITICAL FIX: GRADIENT CLIPPING ---
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                # ---------------------------------------
                
                self.optimizer.step()
                mean_v_loss += v_loss.item()

        return {'loss': mean_v_loss / self.num_epochs}
    