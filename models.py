import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):

    def __init__(self, num_channels, board_height, board_width, num_actions):
        super(ActorCritic, self).__init__()

        # Shared convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Flattened size after conv layers
        self.flatten_size = board_height * board_width * 128

        # Actor head
        self.actor_fc1 = nn.Linear(self.flatten_size, 256)
        self.actor_out = nn.Linear(
            256, num_actions)  # Output logits for each column

        # Critic head
        self.critic_fc1 = nn.Linear(self.flatten_size, 256)
        self.critic_out = nn.Linear(256, 1)  # Output scalar value

    def forward(self, x):
        # x shape: (batch_size, 3, board_height, board_width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten

        # Actor
        actor_hidden = F.relu(self.actor_fc1(x))
        policy_logits = self.actor_out(actor_hidden)

        # Critic
        critic_hidden = F.relu(self.critic_fc1(x))
        value = self.critic_out(critic_hidden)

        return policy_logits, value
