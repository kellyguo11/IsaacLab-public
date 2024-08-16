from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.actor_critic import get_activation

class CustomCNN(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([16, 110, 110]),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([32, 54, 54]),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([64, 26, 26]),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([128, 12, 12]),
            nn.AvgPool2d(12)
        )

        self.linear = nn.Sequential(
            nn.Linear(128, 256), 
            nn.Linear(256, 512), 
        )

    def forward(self, x):
        cnn_x = self.cnn(x.permute(0, 3, 1, 2))
        out = self.linear(cnn_x.view(-1, 128))
        return out


class ActorSH(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = dict()
        self.mlp = nn.Sequential()

    def forward(self, x):
        outputs = []
        for name, head in self.heads.items():
            head_out = head(x[name])
            outputs.append(head_out)
        head_outputs = torch.cat(outputs, dim=-1)
        return self.mlp(head_outputs)

class CriticSH(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = dict()
        self.mlp = nn.Sequential()

    def forward(self, x):
        outputs = []
        for name, head in self.heads.items():
            head_out = head(x[name])
            outputs.append(head_out)
        head_outputs = torch.cat(outputs, dim=-1)
        return self.mlp(head_outputs)


class ActorCriticSH(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        nn.Module.__init__(self)
        activation = get_activation(activation)
        self.activation = activation

        self.actor = ActorSH()
        self.critic = CriticSH()

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor.mlp = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic.mlp = nn.Sequential(*critic_layers)

        # print(f"Actor model: {self.actor}")
        # print(f"Critic model: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def add_actor_head(self, name, in_dim, cnn):
        if cnn:
            self.actor.heads[name] = CustomCNN(in_dim)
        else:
            self.actor.heads[name] = nn.Sequential(nn.Linear(in_dim, 512), self.activation)

    def add_critic_head(self, name, in_dim, cnn):
        if cnn:
            self.critic.heads[name] = CustomCNN(in_dim)
        else:
            self.critic.heads[name] = nn.Sequential(nn.Linear(in_dim, 512), self.activation)
