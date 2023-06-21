import math

# Inicjalizacja graczy
import numpy as np
from matplotlib import pyplot as plt

from src.board_processing import print_board

# players = ['Algorytm 1', 'Algorytm 2', 'Algorytm 3', 'Algorytm 4']
# elo_ratings = {player: 1000 for player in players}
#
#
## Funkcja obliczająca oczekiwaną wartość wyniku na podstawie różnicy rankingów
ELO_DECAY = 32


def expected_score(player_rating, opponent_rating):
    return 1 / (1 + 10 ** ((opponent_rating - player_rating) / 400))


#
#
## Przeprowadzenie turnieju
# num_rounds = 5
#
# for _ in range(num_rounds):
#    for i in range(len(players)):
#        for j in range(i + 1, len(players)):
#            player_a = players[i]
#            player_b = players[j]
#
#            # Symulacja wyniku meczu - tutaj można dodać własną logikę
#            # Załóżmy, że player_a wygrał
#            result = 1
#
#            # Obliczenie oczekiwanych wyników
#            expected_a = expected_score(elo_ratings[player_a], elo_ratings[player_b])
#            expected_b = expected_score(elo_ratings[player_b], elo_ratings[player_a])
#
#            # Obliczenie zmiany rankingów na podstawie wyniku meczu
#            k = 32  # Parametr K kontrolujący szybkość zmiany rankingów
#            change_a = k * (result - expected_a)
#            change_b = k * ((1 - result) - expected_b)
#
#            # Aktualizacja rankingów
#            elo_ratings[player_a] += change_a
#            elo_ratings[player_b] += change_b
#
## Wyświetlenie rankingów ELO po wszystkich rundach
# for player, rating in elo_ratings.items():
#    print(f'{player}: {math.ceil(rating)}')

from enum import Enum
from checkers.game import Game
import src.mcts
import time
import src.Qlearning_model as qm
import random

Result = Enum("Result", ["Win", "Loss", "Draw"])



class Player:
    def __init__(self, name, move_func, move_func_args=None):
        self.name = name
        self.move_func = move_func
        self.move_func_args = move_func_args

        self.games_total = 0
        self.games_won = 0
        self.games_lost = 0
        self.games_draw = 0
        self.elo = 1000
        self.detail_wins = {}
        self.detail_games = {}

    def get_move(self, game):
        if self.move_func_args is None:
            return self.move_func(game)
        else:
            return self.move_func(game, *self.move_func_args)

    def add_result(self, result, op_name, op_elo):
        if op_name not in self.detail_wins:
            self.detail_wins[op_name] = 0

        if op_name not in self.detail_games:
            self.detail_games[op_name] = 0

        self.games_total = self.games_total + 1
        self.detail_games[op_name] += 1
        expected_elo = expected_score(self.elo, op_elo)
        if result is Result.Win:
            self.elo += ELO_DECAY * (1 - expected_elo)
            self.games_won += 1
            self.detail_wins[op_name] += 1
        elif result is Result.Loss:
            self.elo += ELO_DECAY * (0 - expected_elo)
            self.games_lost += 1
        elif result is Result.Draw:
            self.elo += ELO_DECAY * (0.5 - expected_elo)
            self.games_draw += 1

    def clean_results(self):
        self.games_total = 0
        self.games_won = 0
        self.games_lost = 0
        self.games_draw = 0
        self.elo = 1000

    def win_ratio(self):
        return self.games_won / self.games_total

    def draw_ratio(self):
        return self.games_draw / self.games_total


def run_game(p1, p2, display=False):
    game = Game()
    while not game.is_over():
        if game.whose_turn() == 1:
            move = p1.get_move(game)
        else:
            move = p2.get_move(game)
        game.move(move)
        if display:
            print_board(game.board)

    p1_elo, p2_elo = p1.elo, p2.elo
    if game.get_winner() == 1:
        if display:
            print('Player %s wins' % p1.name)
        p1.add_result(Result.Win, op_name=p2.name, op_elo=p2_elo)
        p2.add_result(Result.Loss, op_name=p1.name, op_elo=p1_elo)
    elif game.get_winner() == 2:
        if display:
            print('Player %s wins' % p2.name)
        p1.add_result(Result.Loss, op_name=p2.name, op_elo=p2_elo)
        p2.add_result(Result.Win, op_name=p1.name, op_elo=p1_elo)
    else:
        if display:
            print('Draw')
        p1.add_result(Result.Draw, op_name=p2.name, op_elo=p2_elo)
        p2.add_result(Result.Draw, op_name=p1.name, op_elo=p1_elo)


def print_win_ratio_and_clean_results(players):
    for player in players:
        print(
            player.name.rjust(24) +
            ' | ' + str(round(player.win_ratio(), 5)).ljust(7) +
            ' | ' + str(round(player.draw_ratio(), 5)).ljust(7) +
            ' | ' + str(round(player.elo, 5))
        )
        # player.clean_results()

def win_ratio_plotter():
    win_ratio = np.zeros((len(players), len(players)))

    for i, player in enumerate(players):
        for j, opponent in enumerate(players):
            if player != opponent:
                win_ratio[i][j] = player.detail_wins[opponent.name] / player.detail_games[opponent.name]

    fig, ax = plt.subplots()
    ax.imshow(win_ratio, cmap='Blues')

    ax.set_xticks(np.arange(len(players)))
    ax.set_yticks(np.arange(len(players)))
    players_names = [i.name for i in players]
    ax.set_xticklabels(players_names)
    ax.set_yticklabels(players_names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(len(players)):
        for j in range(len(players)):
            ax.text(j, i, f'{win_ratio[i][j] * 100:.1f}%',
                           ha="center", va="center", color="w")

    ax.set_title("Win Ratio")
    fig.tight_layout()
    plt.savefig("plots/tournament_details.png")


def run_tournament(players, games_per_tour=10, display=False):
    for game in range(games_per_tour):
        for player_a in players:
            for player_b in players:
                if player_a == player_b:
                    break
                print(player_a.name + ' VS ' + player_b.name)
                run_game(player_a, player_b, display)
                src.mcts.clean_table()

    print('==================== Win ratios ====================')
    print((' Player name ').rjust(24) + (' | Wins').ljust(10) + (' | Draws').ljust(10) + ' | ELO'.ljust(10))
    for player in players:
        print('-'.rjust(24, '-') + '-|-'.ljust(10, '-') + '-|-'.ljust(10, '-') + '-|-'.ljust(10, '-'))
        print_win_ratio_and_clean_results([player])
    win_ratio_plotter()



q_model_small = qm.DQN.load('models/model_small_neurons.json')
q_model_mid = qm.DQN.load('models/model_mid_neurons.json')
q_model_big = qm.DQN.load('models/model_big_neurons.json')

players = [
    Player(
        name='MonteCarlo 1',
        move_func=src.mcts.monte_carlo_tree_search,
        move_func_args=(1,)
    ),
    Player(
        name='MonteCarlo 2',
        move_func=src.mcts.monte_carlo_tree_search,
        move_func_args=(2,)
    ),
    Player(
        name='MonteCarlo 4',
        move_func=src.mcts.monte_carlo_tree_search,
        move_func_args=(4,),
    ),
    Player(
        name='Qmodel small',
        move_func=q_model_small.select_action,
        move_func_args=None
    ),
    Player(
        name='Qmodel mid',
        move_func=q_model_mid.select_action,
        move_func_args=None
    ),
    Player(
        name='Qmodel big',
        move_func=q_model_big.select_action,
        move_func_args=None
    )
]


def main():
    run_tournament(players, games_per_tour=10, display=False)


if __name__ == "__main__":
    # warmup_engine()
    main()
