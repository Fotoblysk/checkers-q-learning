# This is a sample Python script.
import random
import time

from checkers.game import Game

from src.board_processing import print_board


def main():
    game = Game()
    while not game.is_over():
        print_board(game.board)
        moves = game.get_possible_moves()
        rand_move = random.randint(0, len(moves) - 1)
        game.move(moves[rand_move])
        time.sleep(0.2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
