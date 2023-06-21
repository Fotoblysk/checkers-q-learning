import torch
import torch.nn as nn
import csv
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import signal
from checkers.game import Game

from torch import optim

from src.board_processing import get_learnable_board, get_readable_board


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add(self, state, action, mask, done, reward, next_state, masks_next):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, mask, done, reward, next_state, masks_next)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class DQN(nn.Module):
    def __init__(self, input_size=None, output_size=None, hidden_layer_size=None):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, output_size)
        self.settings = {
            'input_size': input_size,
            'output_size': output_size,
            'hidden_layer_size': hidden_layer_size
        }

    def forward(self, x, mask):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = x * mask + mask * 0.000000000000000000001  # element-wise multiplication with the mask, avoid when all outputs are 0
        return x

    def save(self, path):
        settings = self.settings.copy()
        settings['model_path'] = path + '.model'

        # save model settings
        json_object = json.dumps(settings, indent=4)
        with open(path + ".json", "w") as jsonfile:
            jsonfile.write(json_object)

        # save model
        torch.save(self.state_dict(), settings['model_path'])
    @staticmethod
    def load(path):
        import os
        print(os.getcwd())
        with open(path, 'r') as file:
            settings = json.load(file)
            loaded_model = DQN(
                settings['input_size'],
                settings['output_size'],
                settings['hidden_layer_size'],
            )
            loaded_model.load_state_dict(torch.load(settings['model_path']))
            loaded_model.eval()
            return loaded_model

    def select_action(self, game, epsilon=0.):
        possible_actions = game.get_possible_moves()
        player = game.whose_turn()
        mask = calc_mask(possible_actions)

        if np.random.rand() < epsilon:
            # select a random action
            action = possible_actions[np.random.choice(len(possible_actions), 1)[0]]

        else:
            # select the action with the highest Q-value
            q_values = self(
                torch.from_numpy(np.array(get_learnable_board(get_readable_board(game.board), player))).float(), mask)
            q_values.sort()
            start = q_values.argmax().item() // 32 + 1
            end = q_values.argmax().item() % 32 + 1
            action = [start, end]
        return action


def train(network, optimizer, replay_memory, batch_size, discount_factor):
    # sample a random batch from replay memory
    # p1
    batch = replay_memory.sample(batch_size)
    states = torch.from_numpy(np.vstack([b[0] for b in batch])).float()
    actions = torch.from_numpy(np.vstack([b[1] for b in batch])).long()
    masks = torch.from_numpy(np.vstack([b[2] for b in batch])).float()

    # possibly in p2
    dones = torch.from_numpy(np.vstack([b[3] for b in batch]).astype(np.uint8)).float()
    rewards = torch.from_numpy(np.vstack([b[4] for b in batch])).float()
    next_states = torch.from_numpy(np.vstack([b[5] for b in batch])).float()
    masks_next = torch.from_numpy(np.vstack([b[6] for b in batch])).float()

    # iterate over the possible moves and mark the corresponding position in the mask as 1.0
    q_values = network(states, masks).gather(1, actions)

    # Get max Q values for (s',a') from target model
    Qsa_prime_target_values = network(next_states, masks_next).detach()
    Qsa_prime_targets = Qsa_prime_target_values.max(1)[0].unsqueeze(1)

    # Compute Q targets for current states
    target = rewards + (discount_factor * Qsa_prime_targets * (1 - dones))

    loss = nn.MSELoss()(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def select_action(network, state, epsilon=0):
    possible_actions = state.get_possible_moves()
    player = state.whose_turn()
    mask = calc_mask(possible_actions)

    if np.random.rand() < epsilon:
        # select a random action
        action = possible_actions[np.random.choice(len(possible_actions), 1)[0]]

    else:
        # select the action with the highest Q-value
        q_values = network(
            torch.from_numpy(np.array(get_learnable_board(get_readable_board(state.board), player))).float(), mask)
        q_values.sort()
        start = q_values.argmax().item() // 32 + 1
        end = q_values.argmax().item() % 32 + 1
        action = [start, end]
    return action


def get_rl_data_from_state(game, player):
    if game.get_winner() == player:
        reward = 1
    elif game.get_winner() is None:
        reward = 0
    else:
        reward = -1
    done = game.is_over()
    return game, reward, done


def calc_mask(possible_actions):
    mask = torch.zeros(1, 32 * 32)
    # iterate over the possible moves and mark the corresponding position in the mask as 1.0
    for start, end in possible_actions:
        index = (start - 1) * 32 + (end - 1)
        mask[0, index] = 1.0
    return mask


running = True


def signal_handler(sig, frame):
    global running
    running = False


def run_learning(model_name, hidden_size, num_games=1000000, pop_size=10, epsilon=0.005, mem_size=10000,
                 discount_factor=0.995, batch_size=64 * 2):
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    global running
    losses = []
    game_lens = []
    rewards = []
    single_player_strikes = []

    replay_memory = ReplayMemory(mem_size)
    network = DQN(32, 32 * 32, hidden_layer_size=hidden_size)
    network_pop = [DQN(32, 32 * 32, hidden_layer_size=hidden_size) for _ in range(pop_size)]
    optimizer_pop = [optim.Adam(network_pop[i].parameters()) for i in range(pop_size)]

    replay_memory_p1 = replay_memory
    replay_memory_p2 = replay_memory

    signal.signal(signal.SIGINT, signal_handler)
    for i in range(num_games):
        if not running:
            break

        if i % 100 == 0:
            print(f"Game:{i} mem_size:{len(replay_memory.memory)}")
        game = Game()
        next_step_learn = False

        last_state = None
        last_actions = None
        last_player = None
        last_mask = None

        game_len = 0
        single_player_strike = 0

        opt_index = random.randint(0, 9)
        opt_index_2 = random.randint(0, 9)

        network_p1 = network_pop[opt_index]
        network_p2 = network_pop[opt_index_2]

        optimizer_p1 = optimizer_pop[opt_index]
        optimizer_p2 = optimizer_pop[opt_index_2]

        while not game.is_over():
            player = game.whose_turn()
            if player == 1:
                replay_memory_c, network_c = replay_memory_p1, network_p1
                replay_memory_o = replay_memory_p2
            else:
                replay_memory_c, network_c = replay_memory_p2, network_p2
                replay_memory_o = replay_memory_p1

            game_len += 1
            state_board_redable = get_readable_board(game.board)

            possible_actions = game.get_possible_moves()
            mask = calc_mask(possible_actions)

            action = network_c.select_action(game, epsilon=epsilon)
            encoded_action = (action[0] - 1) * 32 + (action[1] - 1)

            state_board_player_adjusted = get_learnable_board(state_board_redable, player)
            state_board_encoded = torch.from_numpy(np.array(state_board_player_adjusted)).float()

            # make move
            game.move(action)
            next_state, reward, done = get_rl_data_from_state(game, player)
            next_state_redable = get_readable_board(game.board)

            next_state_board_player_adjusted = get_learnable_board(next_state_redable, player)
            next_state_board_encoded = torch.from_numpy(np.array(next_state_board_player_adjusted)).float()

            if next_step_learn and (player != next_state.whose_turn() or done):
                next_state_board_player_adjusted_other_player = get_learnable_board(next_state_redable, last_player)
                next_state_board_encoded_other_player = torch.from_numpy(
                    np.array(next_state_board_player_adjusted_other_player)).float()

                reward_for_other = -reward if player != last_player else reward

                replay_memory_o.add(last_state, last_actions, last_mask, done, reward_for_other,
                                    next_state_board_encoded_other_player,
                                    mask)  # TODO make sure state is ok for the player
                # just in case
                last_state = None
                last_actions = None
                last_player = None
                last_mask = None

            # add expirences when new state is available
            # if it's player turn again
            if player == next_state.whose_turn() or done:
                single_player_strike += 1
                possible_actions = next_state.get_possible_moves()
                mask_next_state = calc_mask(possible_actions)
                replay_memory_c.add(state_board_encoded, encoded_action, mask, done, reward, next_state_board_encoded,
                                    mask_next_state)

            # if player changed
            else:
                next_step_learn = True
                last_state = torch.from_numpy(
                    np.array(get_learnable_board(get_readable_board(game.board), player))).float()
                last_actions = encoded_action
                last_player = player
                last_mask = mask

            game = next_state

            if done and len(replay_memory_c.memory) > batch_size:
                loss = train(network_p1, optimizer_p1, replay_memory_p1, batch_size,
                             discount_factor)  # no for 2nd player

                train(network_c, optimizer_p2, replay_memory_p2, batch_size, discount_factor)

                losses.append(loss)
                single_player_strikes.append(single_player_strike)
                game_lens.append(game_len)
                rewards.append(reward if player == 1 else -reward)

    running = True
    signal.signal(signal.SIGINT, original_sigint_handler)

    network.save(f"models/{model_name}")

    loses_list = [loss.item() for loss in losses]
    # plt.figure()
    # plt.title(f"Loss {model_name}")
    # plt.plot(loses_list)
    # plt.figure()
    # plt.title(f"Combos {model_name}")
    # plt.plot(single_player_strikes)
    # plt.figure()
    # plt.title(f"Len {model_name}")
    # plt.plot(game_lens)
    # plt.figure()
    # plt.title(f"Reward {model_name}")
    # plt.plot(rewards)

    with open(f'{model_name}.csv', 'w', newline='') as result:
        writer = csv.writer(result, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows([
            loses_list,
            single_player_strikes,
            game_lens,
            rewards])

# To load model use:
# DQN.load(model_file_name)
