from torch.distributions import Categorical
from models import ActorCritic
import torch
import numpy as np
import random
import os


class PpoAgent():

    def __init__(self,
                 state_dims,
                 num_actions,
                 n_epochs=8,
                 memory_size=64,
                 batch_size=8,
                 policy_clip=0.2,
                 gamma=0.99,
                 gae_lambda=0.95,
                 alpha=0.000005,
                 view_training_process=False,
                 load_from_path=None,
                 save_to_path="model.pth"):
        super(PpoAgent, self).__init__()
        self.train = True
        self.action_probabilties = np.empty(memory_size)
        self.values = np.empty(memory_size)
        self.actions_taken = np.empty(memory_size)
        self.rewards = np.empty(memory_size)
        self.dones = np.empty(memory_size)
        self.network = ActorCritic(state_dims[0], state_dims[1], state_dims[2],
                                   state_dims[2])
        self.states = np.empty((memory_size, ) + state_dims)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.total_reward = 0
        self.view_training_process = view_training_process
        self.iteration = 0
        self.save_to_path = save_to_path
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # configure optimizeres for both heads and shared backbone
        actor_params = list(self.network.actor_fc1.parameters()) + list(
            self.network.actor_out.parameters())
        critic_params = list(self.network.critic_fc1.parameters()) + list(
            self.network.critic_out.parameters())
        shared_params = list(self.network.conv1.parameters()) + list(
            self.network.conv2.parameters()) + list(
                self.network.conv3.parameters())
        actor_optimizer = torch.optim.Adam(actor_params + shared_params,
                                           lr=3e-4)
        critic_optimizer = torch.optim.Adam(critic_params + shared_params,
                                            lr=1e-3)

        self.actor_optimizer = torch.optim.Adam(actor_params, lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=1e-3)

        if load_from_path:
            self.network.load_state_dict(torch.load(load_from_path))

    def choose_action(self, board):
        state = torch.tensor(np.expand_dims(board.board,
                                            axis=0)).float().to(self.device)
        self.states[self.iteration] = state

        # the policy output, aka a probability distribution
        logits, value = self.network(state)
        valid_moves = board.get_valid_moves()
        valid_moves_mask = torch.zeros(state.shape[-1])
        valid_moves_mask[valid_moves] = 1
        valid_moves_mask = 1 - valid_moves_mask
        logits = logits - (valid_moves_mask * 100000000)
        pi = Categorical(torch.nn.functional.softmax(logits, dim=-1))

        # the sampled action
        a = pi.sample().int()

        if self.view_training_process:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(board)
            print(pi.probs)
            print(valid_moves)
            input()

        return a, pi, value

    def act(self, board):
        # Check whether memory is full and perform update if so
        if self.train and self.iteration == self.memory_size:
            print(f"---------- episode ---------- reward: {self.total_reward}")
            self.learn()
            self.iteration = 0
            self.total_reward = 0
            torch.save(self.network.state_dict(), self.save_to_path)

        a, pi, value = self.choose_action(board)

        # a = torch.tensor([random.Random().randint(0, 11)]).cuda() # FOR DEBUGGING
        self.actions_taken[self.iteration] = a

        # the state value approximation, i.e. the Q-value approximation.
        self.values[self.iteration] = value

        # the probability of the sampled action
        self.action_probabilties[self.iteration] = pi.log_prob(a)

        # perform the action
        board.make_move(a)

        self.dones[self.iteration] = board.game_won() or board.game_tied()
        reward = board.get_reward()
        if reward == 1:
            self.rewards[
                self.iteration -
                1] = -1  # when opponent wins, the previous move was bad
        self.rewards[self.iteration] = reward
        self.total_reward += reward
        self.iteration += 1

    # Implementation based on
    # https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/ppo_torch.py
    def learn(self):
        for _ in range(self.n_epochs):

            # Calculate advantages with GAE
            A = np.zeros(len(self.rewards), dtype=np.float32)
            for t in range(len(self.rewards) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(self.rewards) - 1):
                    a_t += discount * (self.rewards[k] + self.gamma * self.values[k + 1] * \
                                       (1 - int(self.dones[k])) - self.values[k])
                    discount *= self.gamma * self.gae_lambda
                A[t] = a_t
            A = torch.tensor(A).to(self.device)

            values = torch.tensor(self.values).to(self.device)

            for batch in self.create_random_batches():
                states = torch.tensor(self.states[batch],
                                      dtype=torch.float).to(self.device)
                pi_old = torch.tensor(self.action_probabilties[batch]).to(
                    self.device)
                actions = torch.tensor(self.actions_taken[batch]).to(
                    self.device)

                logits, value = self.network(states)
                dist = Categorical(torch.nn.functional.softmax(logits, dim=-1))
                critic_value = torch.squeeze(value)

                # Compare old and new probs and use advantage to calculate actor loss
                pi_new = dist.log_prob(actions)
                prob_ratio = pi_new.exp() / pi_old.exp()
                weighted_probs = A[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(
                    prob_ratio, 1 - self.policy_clip,
                    1 + self.policy_clip) * A[batch]
                actor_loss = -torch.min(weighted_probs,
                                        weighted_clipped_probs).mean()

                # Calculate the return of each state and use MSELoss to update the critic
                returns = A[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

    def create_random_batches(self):
        indices = np.arange(self.memory_size, dtype=np.int64)
        random.shuffle(indices)
        groups = [
            indices[i:i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        return groups
