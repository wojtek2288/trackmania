import torch
import torch.nn.functional as F
from networks import QNetCNN
from actor import CustomActorModule
from tmrl.training import TrainingAgent
from tmrl.custom.utils.nn import copy_shared, no_grad
from copy import deepcopy
from torch.optim import AdamW

class CustomTrainer(TrainingAgent):
    def __init__(self, observation_space, action_space, device):
        super().__init__(observation_space, action_space, device)
        self._init_hyperparameters()
        self._init_networks(device, observation_space, action_space)
        self._init_optimizers()

    def _init_hyperparameters(self):
        self.gamma = 0.99
        self.polyak = 0.995
        self.alpha = 0.01
        self.lr = 0.00004
        self.alpha_tensor = torch.tensor(float(self.alpha))

    def _init_networks(self, device, observation_space, action_space):
        self.actor = CustomActorModule(observation_space, action_space).to(device)
        self.q1 = QNetCNN().to(device)
        self.q2 = QNetCNN().to(device)
        self.q1_targ = no_grad(deepcopy(self.q1))
        self.q2_targ = no_grad(deepcopy(self.q2))
        self.alpha_tensor = self.alpha_tensor.to(device)

    def _init_optimizers(self):
        self.actor_optim = AdamW(self.actor.parameters(), self.lr)
        self.q_optim = AdamW(list(self.q1.parameters()) + list(self.q2.parameters()), self.lr)

    def train(self, batch):
        obs, actions, rewards, next_obs, dones, truncated = batch
        y = self.compute_targets(next_obs, rewards, dones)
        q_loss = self.update_q_functions(obs, actions, y)
        actor_loss = self.update_actor(obs)
        self.update_q_target(self.q1, self.q1_targ)
        self.update_q_target(self.q2, self.q2_targ)
        return {
            'actor_loss': actor_loss.detach().item(),
            'q_loss': q_loss.detach().item(),
            'reward_sum': torch.sum(rewards).detach().item()
        }

    def compute_targets(self, next_obs, rewards, dones):
        with torch.no_grad():
            next_action, next_log_probability = self.actor(next_obs)
            next_q1 = self.q1_targ(next_obs, next_action)
            next_q2 = self.q2_targ(next_obs, next_action)
            return rewards + self.gamma * (1 - dones) * (torch.min(next_q1, next_q2) - self.alpha_tensor * next_log_probability)

    def update_q_functions(self, obs, actions, y):
        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)

        q1_loss = F.mse_loss(q1, y)
        q2_loss = F.mse_loss(q2, y)
        q_loss = q1_loss + q2_loss

        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()
        return q_loss
    
    def toggle_q_grad(self, requires_grad):
        for q1_p in self.q1.parameters():
            q1_p.requires_grad = requires_grad
        for q2_p in self.q2.parameters():
            q2_p.requires_grad = requires_grad

    def update_actor(self, obs):
        action, log_probability = self.actor(obs)

        self.toggle_q_grad(False)
        q1 = self.q1(obs, action)
        q2 = self.q2(obs, action)

        actor_loss = (self.alpha_tensor * log_probability - torch.min(q1, q2)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.toggle_q_grad(True)
        return actor_loss

    def update_q_target(self, q, q_targ):
        with torch.no_grad():
            q_targ_iter = iter(q_targ.parameters())
            for q_p in q.parameters():
                q_targ_p = next(q_targ_iter)
                q_targ_p.data.mul_(self.polyak).add_(q_p.data * (1 - self.polyak))

    def get_actor(self):
        return no_grad(copy_shared(self.actor))
