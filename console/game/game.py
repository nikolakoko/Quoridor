import math
import os
import time
from algorithms.expectimax import expectimax
from algorithms.minimax import minimax
from game_state import GameState


def clear_screen():
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For macOS and Linux
    else:
        _ = os.system('clear')


class Game:
    def __init__(self):
        self.game_state = GameState()
        self.number = {0: "A", 1: "b", 2: "C", 3: "d", 4: "E", 5: "f", 6: "G", 7: "h",
                       8: "I", 9: "j", 10: "K", 11: "l", 12: "M", 13: "n", 14: "O", 15: "p", 16: "Q"}
        self.alpa = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7,
                     "I": 8, "J": 9, "K": 10, "L": 11, "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16}

    def map_num(self, place):
        x, y = int(place[0]), int(place[1])
        if x not in self.number.keys() or y not in self.number:
            return False
        return (self.number[x], self.number[y])

    def map_alpha(self, place):
        x, y = place[0].upper(), place[1].upper()
        if x not in self.alpa.keys() or y not in self.alpa.keys():
            return False
        return (self.alpa[x], self.alpa[y])

    def random(self):
        from algorithms.random import random_move
        walls = []
        while not self.game_state.is_goal():
            self.game_state.player_stats()
            self.game_state.print_board()
            print(f'Player: {self.game_state.turn}')
            ipt, pawn, wall = random_move(
                self.game_state.possible_moves_pawn(), self.game_state.possible_moves_wall())
            if ipt == 1:
                print("Move: ")
                while True:
                    if not self.game_state.move_pawn(self.game_state.possible_moves_pawn()[pawn]):
                        print("Move: ")
                    else:
                        break
            elif ipt == 2:
                print("Wall: ")
                while True:
                    wall_ = self.game_state.possible_moves_wall()[wall]
                    if not self.game_state.place_wall(wall_):
                        print("Move: ")
                    else:
                        walls.append(wall_)
                        break
            # time.sleep(0.5)
            clear_screen()

        self.game_state.player_stats()
        self.game_state.print_board()
        print(f'{self.game_state.get_winner()} is the winner!')

    def player_v_player(self):
        stateGame = GameState()
        while not stateGame.is_goal():
            stateGame.player_stats()
            stateGame.print_board()
            print(f'Player: {stateGame.turn}')
            print(f'Possible moves: {stateGame.possible_moves_pawn()}')
            print(f'Possible walls: {stateGame.possible_moves_wall()}')
            ipt = input("Pawn(1) or wall(2)?")
            if ipt != "1" or ipt != "2":
                while ipt != "1" and ipt != "2":
                    print("Must pick 1 or 2!")
                    ipt = input("Pawn(1) or wall(2)?")
            if ipt == "1":
                move = input("Move: ")
                move = self.map_alpha(move)
                while type(move) != tuple or not stateGame.move_pawn(move):
                    move = input("Move: ")
                    move = self.map_alpha(move)
            elif ipt == "2":
                move = input("Wall: ")
                move = self.map_alpha(move)
                while type(move) != tuple or not stateGame.place_wall(move):
                    move = input("Wall: ")
                    move = self.map_alpha(move)
            clear_screen()

        stateGame.player_stats()
        stateGame.print_board()
        print(f'{stateGame.get_winner()} is the winner!')

    def other(self):
        # if alg == 1:

        # elif alg == 2:
        #     from algorithms.monte_carlo_tree_search import get_best_move
        # from algorithms.expectimax import get_best_move

        stateGame = GameState()
        ans = input("Do you want to play? (y/n)")
        while ans != "y" and ans != "n":
            ans = input("Do you want to play? (y/n)")
        if ans == "y":
            human_turn = True
            ans = input("Choose player: (1/2)")
            while ans != "1" and ans != "2":
                ans = input("Choose player: (1/2)")
            if ans == "2":
                human_turn = False
            while not stateGame.is_goal():
                if human_turn:
                    stateGame.player_stats()
                    stateGame.print_board()
                    ipt = input("Pawn(1) or wall(2)?")
                    if ipt != "1" or ipt != "2":
                        while ipt != "1" and ipt != "2":
                            print("Must pick 1 or 2!")
                            ipt = input("Pawn(1) or wall(2)?")
                    if ipt == "1":
                        print(f'Possible moves: {
                              stateGame.possible_moves_pawn()}')
                        move = input("Move: ")
                        while not stateGame.move_pawn(self.map_alpha(move)):
                            move = input("Move: ")
                    elif ipt == "2":
                        print(f'Possible walls: {
                              stateGame.possible_moves_wall()}')
                        move = input("Wall: ")
                        while not stateGame.place_wall(self.map_alpha(move)):
                            move = input("Wall: ")

                    human_turn = False
                    clear_screen()

                else:
                    stateGame.player_stats()
                    stateGame.print_board()
                    print("\nAI is thinking...")
                    move = self.minimax_agent(stateGame.turn)

                    if move[0] % 2 == 0:
                        print(f"AI moves pawn to: {move}")
                        stateGame.move_pawn(move)
                    else:
                        wall_coords = stateGame.get_wall_coords(move)
                        print(f"AI places wall at: {wall_coords}")
                        stateGame.place_wall(move)

                    human_turn = True
                    time.sleep(0.5)
                    clear_screen()

        else:
            while not stateGame.is_goal():
                stateGame.player_stats()
                stateGame.print_board()
                print(f'Player: {stateGame.turn}')
                # Get AI move
                move = minimax(stateGame, depth=2)

                if move[0] % 2 == 0:  # Pawn move
                    print(f"AI moves pawn to: {move}")
                    stateGame.move_pawn(move)
                else:  # Wall placement
                    print(f"AI places wall at: {move}")
                    stateGame.place_wall(move)

                time.sleep(0.5)  # To make the game visible
                clear_screen()

        stateGame.player_stats()
        stateGame.print_board()
        print(f'{stateGame.get_winner()} is the winner!')

    def choose_action(self, moves):
        if len(moves.keys()) == 0:
            return None
        k = max(moves)
        winner = k
        action = winner[0]

        return action

    def minimax_agent(self, turn):
        moves = {}
        for move in self.game_state.all_possible_moves(True):
            value = minimax(move[1], 3, -math.inf, math.inf, maximizing_player=False,
                            turn=turn)

            moves[move] = value
        return self.choose_action(moves)

    def expectimax_agent(self):
        moves = {}
        for move in self.game_state.all_possible_moves(True):
            value = expectimax(move[1], 2, False, maximizing_player=False)
            moves[value] = move
        return self.choose_action(moves)

    # TODO implement Monte Carlo Tree Search
    def monte_carlo_agent(self):
        pass


if __name__ == '__main__':
    game = Game()
    # game.random()
    # game.player_v_player()
    game.other()
