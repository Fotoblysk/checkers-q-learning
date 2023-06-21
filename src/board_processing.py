import random


def get_readable_board(board):
    board_array = [0] * 32
    for piece in sorted(board.pieces, key=lambda piece: piece.position if piece.position else 0):
        if piece.position is not None:
            if piece.king:
                board_array[piece.position - 1] = (piece.player * 10)
            else:
                board_array[piece.position - 1] = (piece.player)
    return board_array

def get_learnable_board(readable_board, current_player):
    learnable_board = readable_board.copy()
    player_opponent_d = {1: 2, 2: 1}
    current_player_opponent = player_opponent_d[current_player]
    for i in range(len(learnable_board)):
        if learnable_board[i] != 0:
            if learnable_board[i] == current_player:
                learnable_board[i] = 1
            elif learnable_board[i] == current_player_opponent:
                learnable_board[i] = -1
            elif learnable_board[i] == current_player * 10:
                learnable_board[i] = 10
            elif learnable_board[i] == current_player_opponent * 10:
                learnable_board[i] = -10
            else:
                raise Exception(
                    'Unknown value in readable_board[%d]: %d (only 0,1,2,10,20 acceptable)' % (
                        i, learnable_board[i])
                )
    return learnable_board

def get_hashable_state(game):  # probably we should also add some more data like possible moves is end ect.
    return (tuple(get_readable_board(game.board)), game.whose_turn())

def print_board(board):
    board_array = get_readable_board(board)

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

    # for i in board_array_stringed:
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

def get_random_move(moves):
    return random.randint(0, len(moves) - 1)

