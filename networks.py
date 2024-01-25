import torch
import torch.nn.functional as F
import torch.nn as nn
import tmrl.config.config_constants as cfg

class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.layer1 = nn.Conv2d(cfg.IMG_HIST_LEN, 32, kernel_size=8, stride=4)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.output_features = self.get_output_features()

    def get_output_features(self):
        x = torch.randn(1, cfg.IMG_HIST_LEN, cfg.IMG_HEIGHT, cfg.IMG_WIDTH)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))

        return x.view(x.size(0), -1).size(1)

    def create_multi_layer_perceptron(self, sizes):
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        layers.append(nn.Identity())

        return nn.Sequential(*layers)

class QNetCNN(BaseCNN):
    def __init__(self):
        super(QNetCNN, self).__init__()
        self.perceptron = self.create_multi_layer_perceptron([self.output_features + 12] + [512, 512, 1])
    
    def forward(self, obs, action):
        speed, gear, rpm, images, action1, action2 = obs

        obs = F.relu(self.layer1(images))
        obs = F.relu(self.layer2(obs))
        obs = F.relu(self.layer3(obs))
        obs = obs.view(obs.size(0), -1)
        obs = torch.cat((speed, gear, rpm, obs, action1, action2, action), -1)
        q = self.perceptron(obs)

        return torch.squeeze(q, -1)

class CNN(BaseCNN):
    def __init__(self):
        super(CNN, self).__init__()
        self.perceptron = self.create_multi_layer_perceptron([self.output_features + 9] + [512, 512])

    def forward(self, obs):
        speed, gear, rpm, images, action1, action2 = obs
        obs = F.relu(self.layer1(images))
        obs = F.relu(self.layer2(obs))
        obs = F.relu(self.layer3(obs))
        obs = obs.view(obs.size(0), -1)
        obs = torch.cat((speed, gear, rpm, obs, action1, action2), -1)

        return self.perceptron(obs)
