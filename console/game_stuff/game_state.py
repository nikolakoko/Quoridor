import numpy as np
from utils.stuff import Stuff
from utils.stuff import BoardPieceStat
from utils.stuff import Direction
from copy import copy
from searching import astar


class GameState:
    def __init__(self, initialization=True):
        self.rows = 17  # 9 rows for movement of pawn + 8 rows for wall placement
        self.cols = 17  # 9 columns for movement of pawn + 8 columns for wall placement
        self.player_one_walls = 10
        self.player_two_walls = 10
        self.turn = True

        if initialization:
            self.player_one_pos = (16, 8)
            self.player_two_pos = (0, 8)
            self.player_two_pos = (14, 8)
            self.board = np.zeros((289,), dtype=int)
            self.setup_board()

        self.moves = {
            'up': (2, 0),
            'down': (-2, 0),
            'left': (0, -2),
            'right': (0, 2),
        }

    def setup_board(self):

        # Initializing the pawn and wall cells

        for row in range(self.rows):
            for col in range(self.cols):
                if row % 2 == 0 and col % 2 == 0:
                    self.board[row * self.rows +
                               col] = BoardPieceStat.FREE_PAWN
                else:
                    self.board[row * self.rows +
                               col] = BoardPieceStat.FREE_WALL

        # Initializing starting positions for each player's pawn

        self.board[self.player_one_pos[0] * self.rows +
                   self.player_one_pos[1]] = BoardPieceStat.OCCUPIED_BY_PLAYER_ONE
        self.board[self.player_two_pos[0] * self.rows +
                   self.player_two_pos[1]] = BoardPieceStat.OCCUPIED_BY_PLAYER_TWO

    def test_state(self):
        state = copy(self)
        state.player_one_pos = copy(self.player_one_pos)
        state.player_two_pos = copy(self.player_two_pos)
        state.board = copy(self.board)
        return state

    def get_id(self, x, y):
        return x * self.cols + y

    def player_stats(self):
        print(f'Player 1 remaining walls: {
              self.player_one_walls}   {Stuff.PLAYER_ONE}')
        print(f'Player 2 remaining walls: {
              self.player_two_walls}   {Stuff.PLAYER_TWO}')

    def print_board(self):
        player_positions = "|"
        wall_positions = "\u2500" * 5 + "o"
        ascii_capital = 65
        ascii_lower = 98

        # Printing the letters for the rows

        print("  ", end="")

        for row in range(self.rows):
            if row % 2 == 0:
                print(f'  {chr(ascii_capital)}  ', end="")
                ascii_capital = ascii_capital + 2
            else:
                print(Stuff.WALL_COLOR + chr(ascii_lower) +
                      Stuff.COLOR_RESET, end="")
                ascii_lower = ascii_lower + 2

        ascii_capital = 65
        ascii_lower = 98

        # Printing the letters for the columns and the grid\

        for col in range(self.cols):
            if col % 2 == 0:
                print(f'\n{chr(ascii_capital)} ', end="")
                ascii_capital = ascii_capital + 2
                for place in range(self.cols):
                    id = self.get_id(col, place)
                    # if self.board[id] == BoardPieceStat.FREE_PAWN:
                    #     print(f'{col, place}', end="")
                    if self.board[id] == BoardPieceStat.FREE_PAWN:
                        print(f'     ', end="")
                    elif self.board[id] == BoardPieceStat.FREE_WALL:
                        print(f'{player_positions}', end="")
                    elif self.board[id] == BoardPieceStat.OCCUPIED_BY_PLAYER_ONE:
                        print(f'  {Stuff.PLAYER_ONE} ', end="")
                    elif self.board[id] == BoardPieceStat.OCCUPIED_BY_PLAYER_TWO:
                        print(f'  {Stuff.PLAYER_TWO} ', end="")
                    else:
                        print(Stuff.WALL_COLOR + player_positions +
                              Stuff.COLOR_RESET, end="")

            else:
                print("\n" + Stuff.WALL_COLOR + chr(ascii_lower) +
                      " " + Stuff.COLOR_RESET, end="")
                ascii_lower = ascii_lower + 2
                for place in range(self.cols):
                    id = self.get_id(col, place)
                    if place % 2 == 0 and place < 16:
                        if self.board[id] == BoardPieceStat.OCCUPIED_WALL:
                            print(Stuff.WALL_COLOR + wall_positions[:5] +
                                  Stuff.COLOR_RESET + wall_positions[5], end="")
                        else:
                            print(f'{wall_positions}', end="")
                    elif place == 16:
                        if self.board[id] == BoardPieceStat.OCCUPIED_WALL:
                            print(Stuff.WALL_COLOR + wall_positions[:5] +
                                  Stuff.COLOR_RESET, end="")
                        else:
                            print(f'{wall_positions[:5]}', end="")
                    else:
                        self.board[id] == None

        print()

    def is_place_free(self, place):
        x, y = place
        id = self.get_id(x, y)
        return self.board[id] == BoardPieceStat.FREE_PAWN or self.board[id] == BoardPieceStat.FREE_WALL

    # def determine_direction(self):
    #     if self.player_one_pos[0] - self.player_two_pos[0] == 2 and self.player_one_pos[1] == self.player_two_pos[1]:
    #         if self.turn:
    #             return Direction.NORTH
    #         return Direction.SOUTH
    #     elif self.player_one_pos[0] - self.player_two_pos[0] == -2 and self.player_one_pos[1] == self.player_two_pos[1]:
    #         if self.turn:
    #             return Direction.SOUTH
    #         return Direction.NORTH
    #     elif self.player_one_pos[0] == self.player_two_pos[0] and self.player_one_pos[1] - self.player_two_pos[1] == 2:
    #         if self.turn:
    #             return Direction.WEST
    #         return Direction.EAST
    #     elif self.player_one_pos[0] == self.player_two_pos[0] and self.player_one_pos[1] - self.player_two_pos[1] == -2:
    #         if self.turn:
    #             return Direction.EAST
    #         return Direction.WEST

    def is_diagonal_or_jump(self):
        for place in self.moves.values():
            opponent = (self.player_one_pos[0] + place[0],
                        self.player_one_pos[1] + place[1])
            if opponent == self.player_one_pos or opponent == self.player_two_pos:
                return True
        return False

    def is_goal(self):
        return self.player_one_pos[0] == 0 or self.player_two_pos[0 == 16]

    def move_pawn(self, move):
        x, y = move
        id = self.get_id(x, y)
        if self.turn:
            old_x, old_y = self.player_one_pos
            self.player_one_pos = (x, y)
            self.board[id] = BoardPieceStat.OCCUPIED_BY_PLAYER_ONE
        else:
            old_x, old_y = self.player_two_pos
            self.player_two_pos = (x, y)
            self.board[id] = BoardPieceStat.OCCUPIED_BY_PLAYER_TWO
        self.board[self.get_id(old_x, old_y)] = BoardPieceStat.FREE_PAWN

    def place_wall(self, place):

        # Placement of vertical walls:
        if self.check_valid_wall(place):
            if place[0] % 2 == 0:
                if place[0] < 15:
                    self.board[self.get_id(place[0], place[1])] = self.board[self.get_id(
                        place[0] + 2, place[1])] = BoardPieceStat.OCCUPIED_WALL
                else:
                    self.board[self.get_id(place[0], place[1])] = self.board[self.get_id(
                        place[0] - 2, place[1])] = BoardPieceStat.OCCUPIED_WALL

            # Placement of horizontal walls

            else:
                if place[1] < 15:
                    self.board[self.get_id(place[0], place[1])] = self.board[self.get_id(
                        place[0], place[1] + 2)] = BoardPieceStat.OCCUPIED_WALL
                else:
                    self.board[self.get_id(place[0], place[1])] = self.board[self.get_id(
                        place[0], place[1] - 2)] = BoardPieceStat.OCCUPIED_WALL

            if self.turn:
                self.player_one_walls -= 1
            else:
                self.player_two_walls -= 1
        else:
            return False

    def possible_moves_wall(self):
        pass

    def possible_moves_pawn(self):
        if self.turn:
            old_place = self.player_one_pos
        else:
            old_place = self.player_two_pos
        wall_places = {
            'up': (1, 0),
            'down': (-1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

        moves = list()
        for wall, pawn in zip(wall_places.values(), self.moves.values()):
            wall_place = (old_place[0] + wall[0], old_place[1] + wall[1])
            pawn_place = (old_place[0] + pawn[0], old_place[1] + pawn[1])
            if self.check_valid_pawn(pawn_place) and self.is_place_free(wall_place):
                moves.append(pawn_place)
        if self.is_diagonal_or_jump():
            opp = self.get_opp_location()
            for wall, pawn in zip(wall_places.values(), self.moves.values()):
                wall_place = (opp[0] + wall[0], opp[1] + wall[1])
                pawn_place = (opp[0] + pawn[0], opp[1] + pawn[1])
                if self.check_valid_pawn(pawn_place) and self.is_place_free(wall_place):
                    moves.append(pawn_place)
        return moves

    def get_opp_location(self):
        if self.turn:
            return self.player_two_pos
        return self.player_one_pos

    def check_valid_pawn(self, place):
        x, y = place[0], place[1]
        id = self.get_id(x, y)
        if x % 2 == 1 or y % 2 == 1 or x < 0 or x > 16 or y < 0 or y > 16 or self.board[id] != BoardPieceStat.FREE_PAWN:
            return False
        return True

    def check_valid_wall(self, place):
        if self.turn and self.player_one_walls == 0:
            return False
        elif not self.turn and self.player_two_walls == 0:
            return False
        if place[0] % 2 == 0 and place[1] % 2 == 1:
            if place[0] < 15:
                if self.board[self.get_id(place[0], place[1])] == self.board[self.get_id(place[0] + 2, place[1])] == BoardPieceStat.OCCUPIED_WALL:
                    return False
            else:
                if self.board[self.get_id(place[0], place[1])] == self.board[self.get_id(place[0] - 2, place[1])] == BoardPieceStat.OCCUPIED_WALL:
                    return False
        elif place[0] % 2 == 1 and place[1] % 2 == 0:
            if place[1] < 15:
                if self.board[self.get_id(place[0], place[1])] == self.board[self.get_id(place[0], place[1] + 2)] == BoardPieceStat.OCCUPIED_WALL:
                    return False
            else:
                if self.board[self.get_id(place[0], place[1])] == self.board[self.get_id(place[0], place[1] - 2)] == BoardPieceStat.OCCUPIED_WALL:
                    return False
        else:
            return False
        return True
        # game_state = self.test_state()
        # game_state.place_wall(place)
        # return astar.heuristic(game_state)


if __name__ == '__main__':
    stateGame = GameState()
    # stateGame.board[stateGame.get_id(14, 7)] = BoardPieceStat.OCCUPIED_WALL
    # stateGame.board[stateGame.get_id(14, 9)] = BoardPieceStat.OCCUPIED_WALL
    # stateGame.board[stateGame.get_id(15, 4)] = BoardPieceStat.OCCUPIED_WALL
    stateGame.place_wall((15, 2))
    stateGame.place_wall((10, 3))
    stateGame.player_stats()
    stateGame.print_board()
    print(f'Possible moves: {stateGame.possible_moves_pawn()}')
