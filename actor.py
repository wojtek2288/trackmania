import json
import torch
import torch.nn as nn
from networks import CNN
from torch.distributions.normal import Normal
from tmrl.actor import TorchActorModule
from encoders import JSONDecoder, JSONEncoder

class CustomActorModule(TorchActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        self.log_std_min = -20
        self.log_std_max = 2
        self.action_limit = action_space.high[0]
        self.cnn_network = CNN()
        self.action_mean_layer = nn.Linear(512, action_space.shape[0])
        self.action_log_std_layer = nn.Linear(512, action_space.shape[0])

    def load(self, path, device):
        self.device = device
        with open(path, 'r', encoding="utf8") as json_file:
            state_dict = json.load(json_file, cls=JSONDecoder)
        self.load_state_dict(state_dict)
        self.to_device(device)
        return self

    def save(self, path):
        model_data = self.state_dict()
        with open(path, 'w', encoding="utf8") as outfile:
            json.dump(model_data, outfile, cls=JSONEncoder)

    def forward(self, observation, test=False):
        cnn_output = self.cnn_network(observation)
        mean_action = self.action_mean_layer(cnn_output)
        log_std_action = self.action_log_std_layer(cnn_output)

        std_action = torch.exp(torch.clamp(log_std_action, self.log_std_min, self.log_std_max))
        action_distribution = Normal(mean_action, std_action)

        action = mean_action if test else action_distribution.rsample()

        tanh_action = torch.tanh(action)
        tanh_correction = torch.sum(torch.log(1 - tanh_action ** 2 + 1e-6), axis=1)
        log_probability = action_distribution.log_prob(action).sum(axis=-1) - tanh_correction

        scaled_action = self.action_limit * tanh_action.squeeze()

        return scaled_action, log_probability

    def act(self, observation, test=False):
        with torch.no_grad():
            action, _ = self.forward(observation, test)
            return action.cpu().numpy()
