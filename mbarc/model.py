from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from mbarc.utils import standardize_frame


class RewardHead(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.reward_features = nn.Linear(size, 128)
        self.reward_classifier = nn.Linear(128, 2)

    def forward(self, x):
        x = self.reward_features(x)
        x = self.reward_classifier(x)
        return x


class ModifiedRewardHead(nn.Module):

    def __init__(self, input_shape, c1, c2, hidden_size=64, dropout=0.1):
        super().__init__()
        self.dropout = dropout

        self.embed1 = nn.Conv2d(c1, hidden_size, kernel_size=1)
        self.embed2 = nn.Conv2d(c2, hidden_size // 4, kernel_size=1)

        self.downscale1 = nn.Conv2d(hidden_size // 4, hidden_size // 2, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.InstanceNorm2d(hidden_size // 2)
        self.downscale2 = nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(hidden_size)

        units = hidden_size * 2 * ceil(input_shape[0] / 4) * ceil(input_shape[1] / 4)
        self.reward_features = nn.Linear(units, 128)
        self.reward_classifier = nn.Linear(128, 2)

    def forward(self, x_mid, x_fin):
        x1 = self.embed1(x_mid)
        x2 = self.embed2(x_fin)

        x2 = F.dropout(x2, self.dropout)
        x2 = self.downscale1(x2)
        x2 = F.relu(x2)
        x2 = self.norm1(x2)

        x2 = F.dropout(x2, self.dropout)
        x2 = self.downscale2(x2)
        x2 = F.relu(x2)
        x2 = self.norm2(x2)

        x = torch.cat((x1, x2), dim=1)
        x = torch.flatten(x, start_dim=1)

        x = self.reward_features(x)
        x = self.reward_classifier(x)

        return x


class WorldModel(nn.Module):

    def __init__(
            self,
            channels,
            n_action,
            hidden_size=32,
            dropout=0.1,
            activation=F.relu,
            modified_model=False,
            env_shape=None
    ):
        super().__init__()

        if modified_model:
            assert env_shape is not None

        self.dropout = dropout
        self.activation = activation
        self.modified_model = modified_model

        self.input_embedding = nn.Conv2d(channels, hidden_size, kernel_size=1)

        self.downscale1 = nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.InstanceNorm2d(hidden_size * 2, affine=True)
        self.downscale2 = nn.Conv2d(hidden_size * 2, hidden_size * 4, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(hidden_size * 4, affine=True)

        self.ai11 = nn.Linear(n_action, hidden_size * 4)
        self.ai12 = nn.Linear(n_action, hidden_size * 4)

        self.middle1 = nn.Conv2d(hidden_size * 4, hidden_size * 4, kernel_size=3, padding=1)
        self.middle2 = nn.Conv2d(hidden_size * 4, hidden_size * 4, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm2d(hidden_size * 4, affine=True)

        self.ai21 = nn.Linear(n_action, hidden_size * 4)
        self.ai22 = nn.Linear(n_action, hidden_size * 4)
        self.upscale1 = nn.ConvTranspose2d(
            hidden_size * 4,
            hidden_size * 2,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.norm4 = nn.InstanceNorm2d(hidden_size * 2, affine=True)
        self.ai31 = nn.Linear(n_action, hidden_size * 2)
        self.ai32 = nn.Linear(n_action, hidden_size * 2)
        self.upscale2 = nn.ConvTranspose2d(
            hidden_size * 2,
            hidden_size,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.norm5 = nn.InstanceNorm2d(hidden_size, affine=True)

        self.output_embedding = nn.Conv2d(hidden_size, channels, kernel_size=1)

        if modified_model:
            self.reward_head = ModifiedRewardHead(env_shape, hidden_size * 4, hidden_size)
        else:
            self.reward_head = RewardHead(hidden_size * 5)

    def forward(self, x, action):
        x = torch.stack([standardize_frame(frame) for frame in x])
        x = self.input_embedding(x)

        res = []
        res.append(x)
        x = F.dropout(x, self.dropout)
        x = self.downscale1(x)
        x = self.activation(x)
        x = self.norm1(x)
        res.append(x)
        x = F.dropout(x, self.dropout)
        x = self.downscale2(x)
        x = self.activation(x)
        x = self.norm2(x)

        mask = self.ai11(action)
        mask = mask.view((-1, x.shape[1], 1, 1))
        x = x * torch.sigmoid(mask)
        mask = self.ai12(action)
        mask = mask.view((-1, x.shape[1], 1, 1))
        x = x + mask

        if self.modified_model:
            x_mid = x
        else:
            x_mid = torch.mean(x, dim=(2, 3))

        x = F.dropout(x, self.dropout)
        x = self.middle1(x)
        x = self.activation(x)
        y = F.dropout(x)
        y = self.middle2(y)
        y = self.activation(y)
        x = self.norm3(x + y)

        x = F.dropout(x, self.dropout)
        mask = self.ai21(action)
        mask = mask.view((-1, x.shape[1], 1, 1))
        x = x * torch.sigmoid(mask)
        mask = self.ai22(action)
        mask = mask.view((-1, x.shape[1], 1, 1))
        x = x + mask
        x = self.upscale1(x)
        x = self.activation(x)
        x = x + res[1]
        x = self.norm4(x)

        x = F.dropout(x, self.dropout)
        mask = self.ai31(action)
        mask = mask.view((-1, x.shape[1], 1, 1))
        x = x * torch.sigmoid(mask)
        mask = self.ai32(action)
        mask = mask.view((-1, x.shape[1], 1, 1))
        x = x + mask
        x = self.upscale2(x)
        x = self.activation(x)
        x = x + res[0]
        x = self.norm5(x)

        if self.modified_model:
            x_fin = x
            reward_pred = self.reward_head(x_mid, x_fin)
        else:
            x_fin = torch.mean(x, dim=(2, 3))
            reward_pred = self.reward_head(torch.cat((x_mid, x_fin), dim=1))

        x = self.output_embedding(x)

        return x, reward_pred
