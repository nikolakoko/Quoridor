from game_state import GameState
from time import time, sleep
from utils.stuff import WallDirection, Color
from algorithms.minimax import minimax
from algorithms.expectimax import expectimax
from algorithms.monte_carlo import SearchNode
import math


class Game:
    def __init__(self):

        self.player_simulation_algorithms = ["minimax", "minimax"]
        self.game_state = GameState()
        self.algorithms = ["minimax", "expectimax", "monte-carlo-tree-search"]
        self.execution_times = []
        self.alpa = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7,
                     "I": 8, "J": 9, "K": 10, "L": 11, "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16}
        self.initialize()

    def print_commands(self):
        print(
            "You can move your piece or place a wall by entering" + Color.CYAN + " xy " + Color.RESET + "where x is the row letter and y column letter")

    def initialize(self):
        Game.print_colored_output("WELCOME TO QUORIDOR!", Color.CYAN)
        print("\n")
        self.print_commands()
        print("{0:-<100}".format(""))

        a = input("\nDo you want to play against a computer?[Y/n]: ")
        if a == "Y" or a == "y":
            self.game_state.is_simulation = False

            print("Choose the second player algorithm: ")
            print("1. Minimax")
            print("2. Expectimax")
            print("3. Monte Carlo Tree Search")
            while True:
                x = input("Choose: ")
                if not x.isdigit() and x != "x" and x != "X":
                    Game.print_colored_output("Illegal input!", Color.RED)
                elif x == "x" or x == "X":
                    exit(0)
                else:
                    if 0 <= int(x) - 1 < len(self.algorithms):
                        self.player_simulation_algorithms[1] = self.algorithms[int(
                            x) - 1]
                        Game.print_colored_output("Chosen algorithm for player 2 is {0:30}".format(
                            self.player_simulation_algorithms[1].upper()), Color.CYAN)
                        break
                    else:
                        Game.print_colored_output("Illegal input!", Color.RED)
        else:
            self.game_state.is_simulation = True
            print("Choose the players algorithms [Player 1, Player 2]")
            print("1. Minimax")
            print("2. Expectimax")
            print("3. Monte Carlo Tree Search")
            while True:
                x = input("Choose: ")
                if not len(x.split(",")) == 2 and x != "x" and x != "X":
                    Game.print_colored_output("Illegal input!", Color.RED)
                elif x == "x" or x == "X":
                    exit(0)
                else:
                    one, two = x.split(",")
                    if 0 <= int(one) - 1 < len(self.algorithms) and 0 <= int(two) - 1 < len(self.algorithms):
                        self.player_simulation_algorithms[0] = self.algorithms[int(
                            one) - 1]
                        self.player_simulation_algorithms[1] = self.algorithms[int(
                            two) - 1]
                        Game.print_colored_output("Chosen algorithm for player 1 is {0:30}".format(
                            self.player_simulation_algorithms[0].upper()), Color.CYAN)
                        Game.print_colored_output("Chosen algorithm for player 2 is {0:30}".format(
                            self.player_simulation_algorithms[1].upper()), Color.CYAN)
                        break
                    else:
                        Game.print_colored_output("Illegal input!", Color.RED)

    def choose_action(self, d):
        if len(d.keys()) == 0:
            return None
        k = max(d)
        winner = d[k]
        action = winner[1]

        if len(action) == 2:
            self.game_state.move_piece(action)
        else:
            self.game_state.place_wall(action)
        return action

    def minimax_agent(self, player_one_minimax):
        d = {}
        for child in self.game_state.all_possible_moves(player_one_minimax):
            value = minimax(child[0], 3, -math.inf, math.inf, False, player_one_minimax)
            d[value] = child
        return self.choose_action(d)

    def expectimax_agent(self, player_one_maximizer):
        d = {}
        for child in self.game_state.all_possible_moves(player_one_maximizer):
            value = expectimax(child[0], 2, False, player_one_maximizer)
            d[value] = child
        return self.choose_action(d)

    def map_alpha(self, place):
        if len(place) != 2:
            return False
        x, y = place[0].upper(), place[1].upper()
        if x not in self.alpa.keys() or y not in self.alpa.keys():
            return False
        return self.alpa[x], self.alpa[y]

    def get_wall_direction(self, place):
        if place[0] % 2 == 0 and place[1] % 2 == 1:
            if place[0] < 15:
                return WallDirection.SOUTH
            else:
                return WallDirection.NORTH
        elif place[0] % 2 == 1 and place[1] % 2 == 0:
            if place[1] < 15:
                return WallDirection.EAST
            else:
                return WallDirection.WEST
        return None

    def player_one_user(self):
        while True:
            ipt = input("Pawn(1) or wall(2)?")
            if ipt != "1" or ipt != "2":
                while ipt != "1" and ipt != "2":
                    print("Must pick 1 or 2!")
                    ipt = input("Pawn(1) or wall(2)?")

            if ipt == "1":
                value = input("Enter move: ")
                available_moves = self.game_state.get_available_moves(False)
                move = self.map_alpha(value)
                if move not in available_moves:
                    Game.print_colored_output("Illegal move!", Color.RED)
                else:
                    self.game_state.move_piece(move)
                    break
            elif ipt == "2":
                value = input("Enter wall: ")
                move = self.map_alpha(value)
                direction = self.get_wall_direction(move)
                print(move, direction)
                is_placement_valid, coords = self.game_state.check_wall_placement(move,
                                                                                  direction)
                if not is_placement_valid:
                    Game.print_colored_output(
                        "Illegal wall placement!", Color.RED)
                else:
                    self.game_state.place_wall(coords)
                    break
            else:
                Game.print_colored_output("Illegal command!", Color.RED)

    def player_simulation(self, player_number):
        if player_number == 1:
            index = 0
            maximizer = True
        else:
            index = 1
            maximizer = False
        t1 = time()
        print("Player {0:1} is thinking...\n".format(player_number))
        action = (0, 0)
        if self.player_simulation_algorithms[index] == "minimax":
            action = self.minimax_agent(maximizer)
        elif self.player_simulation_algorithms[index] == "expectimax":
            action = self.expectimax_agent(maximizer)
        elif self.player_simulation_algorithms[index] == "monte-carlo-tree-search":
            start = SearchNode(state=self.game_state,
                               is_maximizing=maximizer)
            selected_node = start.best_action()
            action = selected_node.parent_action
            self.game_state.execute_action(action, False)

        if action is not None:
            if len(action) == 2:
                self.print_colored_output(
                    "Player {0:1} has moved his piece.".format(player_number), Color.CYAN)
            else:
                self.print_colored_output(
                    "Player {0:1} has placed a wall.".format(player_number), Color.CYAN)
            t2 = time()
            self.execution_times.append(t2 - t1)
            self.print_colored_output(
                "It took him " + str(round(t2 - t1, 2)) + " seconds.", Color.CYAN)
            # sleep(1.5)
            return True
        else:
            self.print_colored_output(
                "Player {0:1} has no moves left.".format(player_number), Color.CYAN)
            return False

    def check_end_state(self):
        if self.game_state.is_goal_state():
            winner = self.game_state.get_winner()
            if not self.game_state.is_simulation:
                if winner == "P1":
                    self.print_colored_output("You won!", Color.GREEN)
                else:
                    self.print_colored_output("You lost!", Color.RED)
            else:
                self.print_colored_output(
                    "The winner is " + winner + ".", Color.CYAN)
            return True
        else:
            return False

    def play(self):
        while True:
            print()
            self.game_state.print_game_stats()
            print("\n")
            self.game_state.print_board()
            print()

            if self.check_end_state():
                print("Execution average: ", sum(
                    self.execution_times) / len(self.execution_times))
                break

            if self.game_state.player_one:
                if not self.game_state.is_simulation:
                    self.player_one_user()
                else:
                    res = self.player_simulation(1)
                    sleep(1.5)
                    if not res:
                        break
            else:
                res = self.player_simulation(2)
                if not res:
                    break

            self.game_state.player_one = not self.game_state.player_one

    @staticmethod
    def print_colored_output(text, color):
        print(color + text + Color.RESET)


if __name__ == '__main__':
    g = Game()
    g.play()
