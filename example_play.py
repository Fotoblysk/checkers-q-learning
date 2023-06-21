# This is a sample Python script.
import time

from checkers.game import Game

from src.board_processing import print_board, get_random_move

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from src.mcts import warmup_engine, monte_carlo_tree_search, get_table


def main():
    game = Game()
    # populate array

    while not game.is_over():
        if game.whose_turn() == 1:
            move = monte_carlo_tree_search(game, 10)
        else:
            moves = game.get_possible_moves()
            move = moves[get_random_move(moves)]
        game.move(move)
        time.sleep(0.1)
        print_board(game.board)
    print(f"Winner is player {game.get_winner()},  Buffer size:{len(get_table())}")


if __name__ == '__main__':
    warmup_engine()
    while True:
        main()
