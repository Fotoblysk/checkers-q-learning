# This is a sample Python script.
import math
import random
import time
import random
from copy import deepcopy

from checkers.game import Game


def get_readable_board(board):  # TODO fix mtcs and visu too
    board_array = [0] * 32
    for piece in sorted(board.pieces, key=lambda piece: piece.position if piece.position else 0):
        if piece.position is not None:
            if piece.king:
                board_array[piece.position - 1] = (piece.player * 10)
            else:
                board_array[piece.position - 1] = (piece.player)
    return board_array


def get_hashable_state(game):  # probably we should also add some more data like possible moves is end ect.
    return (tuple(get_readable_board(game.board)), game.whose_turn())


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
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

def get_random_move(moves):
    return random.randint(0, len(moves) - 1)


### MCTS inpl
table = dict()


def clean_table():
    global table
    table = dict()


# implement the playout function
def playout(state: Game):
    while not state.is_over():
        moves = state.get_possible_moves()
        if not moves:
            break
        move = random.choice(moves)
        state.move(move)


# implement the MCTS algorithm
def monte_carlo_tree_search(game: Game, iterations=100):
    root = Node(game)
    for _ in range(iterations):
        # select a node to expand
        node = root
        while not node.state.is_over() and node.children:
            node = node.select_child()

        # if the node represents a leaf, expand it
        if not node.children:
            node.expand()

        # check if the state of the node is in the dynamic programming table
        # else:
        # if the value is not in the table, simulate a playout and update the node's value and visit count
        playout_state = deepcopy(node.state)
        playout(playout_state)

        # propagate the results of the playout back up the tree
        while node is not None:
            node.update(playout_state)
            node = node.parent

    # select the best move based on the values of the root's children
    selected_move = root.children[0].move
    selected_score = root.children[0].value
    for child in root.children:
        if child.state.whose_turn() != root.state.whose_turn():
            if child.value < selected_score:  # todo whe should select child with worst score as child has opposite goal?
                selected_score = child.value
                selected_move = child.move
        else:
            if child.value > selected_score:  # todo whe should select child with worst score as child has opposite goal?
                selected_score = child.value
                selected_move = child.move

    return selected_move


# define a node in the MCTS tree
class Node:
    def __init__(self, state: Game, move=None, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0

    def expand(self):
        # create child nodes for all legal moves
        for move in self.state.get_possible_moves():
            child_state = deepcopy(self.state)
            child_state.move(move)
            child = Node(child_state, move=move, parent=self)
            self.children.append(child)

    def select_child(self):
        # select a child node using the UCB1 formula
        if self.state.whose_turn() == self.children[0].state.whose_turn():
            selected_score = self.children[0].value / max(self.children[0].visits, 1) + math.sqrt(
                2 * math.log(self.visits) / max(self.children[0].visits, 1))
        else:
            selected_score = -self.children[0].value / max(self.children[0].visits, 1) + math.sqrt(
                2 * math.log(self.visits) / max(self.children[0].visits, 1))

        selected_child = self.children[0]
        for child in self.children:
            if self.state.whose_turn() == child.state.whose_turn():
                score = child.value / max(child.visits, 1) + math.sqrt(2 * math.log(self.visits) / max(child.visits, 1))
            else:
                score = -child.value / max(child.visits, 1) + math.sqrt(
                    2 * math.log(self.visits) / max(child.visits, 1))
            if score > selected_score:
                selected_score = score
                selected_child = child
        return selected_child

    def update(self, playout_state):
        # update the node's value and visit count based on the results of the playout
        key = hash(get_hashable_state(self.state))  # todo make state hashable
        if key in table:
            # if the value is in the table, use it to update the node's value and visit count
            self.value = table[key][0]
            self.visits = table[key][1]
        self.visits += 1

        if playout_state.get_winner() == self.state.whose_turn():
            # print("win")
            self.value += 1
        elif playout_state.get_winner() is None:
            # print("draw")
            self.value += 0
        else:
            # print("lose") # todo can I use negative values
            self.value += -1
        key = hash(get_hashable_state(self.state))  # todo make state hashable
        table[key] = (self.value, self.visits)


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
    print(f"Winner is player {game.get_winner()},  Buffer size:{len(table)}")


def warmup_engine():
    monte_carlo_tree_search(Game(), 100)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    warmup_engine()
    while True:
        main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
