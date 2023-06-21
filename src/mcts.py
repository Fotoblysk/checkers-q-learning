# This is a sample Python script.
import math
import random
from copy import deepcopy

from checkers.game import Game

from src.board_processing import get_hashable_state, print_board, get_random_move

table = dict()
def get_table():
    return table
def clean_table():
    global table
    table = dict()


def playout(state: Game):
    while not state.is_over():
        moves = state.get_possible_moves()
        if not moves:
            break
        move = random.choice(moves)
        state.move(move)


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
        print_board(game.board)
    print(f"Winner is player {game.get_winner()},  Buffer size:{len(table)}")


def warmup_engine():
    monte_carlo_tree_search(Game(), 100)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    warmup_engine()
    while True:
        clean_table()
        main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
