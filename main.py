# This is a sample Python script.
import random
import time
from copy import deepcopy

from checkers.checkers.game import Game


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def print_board(board):
    board_array = []

    for piece in sorted(board.pieces, key=lambda piece: piece.position if piece.position else 0):
        if piece.position is not None:
            empty_positions = piece.position - 1 - len(board_array)
            for i in range(empty_positions):
                board_array.append(0)
            if piece.king:
                board_array.append(piece.player*10)
            else:
                board_array.append(piece.player)
    empty_positions = 32 - len(board_array)
    for i in range(empty_positions):
        board_array.append(0)

    board_array_split = [[] for _ in range(8)]
    for i, v in enumerate(board_array):
        board_array_split[i // 4].append(v)

    board_array_stringed = [[] for _ in range(8)]
    for i in range(len(board_array_split)):
        if i % 2 == 0:
            for j in board_array_split[i]:
                board_array_stringed[i].append(-1)
                board_array_stringed[i].append(j)
        else:
            for j in board_array_split[i]:
                board_array_stringed[i].append(j)
                board_array_stringed[i].append(-1)

    #for i in board_array_stringed:
    #   print(i)

    board_array_stringed_encoded = ''
    for i in board_array_stringed:
        for j in i:
            if j == 1:
                board_array_stringed_encoded += '♟ '
            elif j == 2:
                board_array_stringed_encoded += '♙ '
            elif j == 10:
                board_array_stringed_encoded += '♚ '
            elif j == 20:
                board_array_stringed_encoded += '♔ '
            else:
                board_array_stringed_encoded += '▭ '

        board_array_stringed_encoded += '\n'

    print(board_array_stringed_encoded)


#        print(piece.player)  # 1 or 2
#        print(piece.position)  # 1-32


def scan_pieces(game):
    for piece in game.board.pieces:
        print(piece.position)  # 1-32
        # print(piece.player)  # 1 or 2
        #        print(piece.other_player)  # 1 or 2
        #        print(piece.king)  # True or False
        #        print(piece.captured)  # True or False


#        print(piece.get_possible_capture_moves())  # [[int, int], [int, int], ...]
#        print(piece.get_possible_positional_moves())  # [[int, int], [int, int], ...]

def main():
    game = Game()
    while not game.is_over():
        print_board(game.board)
        moves = game.get_possible_moves()
        rand_move = random.randint(0,len(moves)-1)
        game.move(moves[rand_move])
        time.sleep(0.2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
