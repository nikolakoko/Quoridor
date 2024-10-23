import numpy as np
from utils.stuff import Stuff
from utils.stuff import BoardPieceStat
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
            self.player_one_pos = np.array([16, 8])
            self.player_two_pos = np.array([0, 8])
            self.board = np.zeros((289,), dtype=int)
            self.setup_board()

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
                print(f'{chr(ascii_lower)}', end="")
                ascii_lower = ascii_lower + 2

        ascii_capital = 65
        ascii_lower = 98

        # Printing the letters for the columns and the grid

        # for i in range(len(self.board)):
        #     if self.board[i] == BoardPieceStat.OCCUPIED_BY_PLAYER_TWO:
        #         print(i)

        for col in range(self.cols):
            if col % 2 == 0:
                print(f'\n{chr(ascii_capital)} ', end="")
                ascii_capital = ascii_capital + 2
                for place in range(self.cols):
                    id = self.get_id(col, place)
                    if self.board[id] == BoardPieceStat.FREE_PAWN:
                        print(f'{col, place}', end="")
                    elif self.board[id] == BoardPieceStat.FREE_WALL:
                        print(f'{player_positions}', end="")
                    elif self.board[id] == BoardPieceStat.OCCUPIED_BY_PLAYER_ONE:
                        print(f'  {Stuff.PLAYER_ONE}  ', end="")
                    elif self.board[id] == BoardPieceStat.OCCUPIED_BY_PLAYER_TWO:
                        print(f'  {Stuff.PLAYER_TWO}  ', end="")
                    else:
                        print(Stuff.WALL_COLOR + player_positions +
                              Stuff.COLOR_RESET, end="")

            else:
                print(f'\n{chr(ascii_lower)} ', end="")
                ascii_lower = ascii_lower + 2

                for place in range(self.cols - 1):
                    id = self.get_id(col, place)
                    if self.board[id] == BoardPieceStat.FREE_PAWN:
                        print(f'{player_positions}', end="")
                for place in range(1, self.cols + 1):
                    if place % 2 == 1 and place < 17:
                        print(f'{wall_positions}', end="")
                    elif place == 17:
                        print(f'{wall_positions[:5]}', end="")

    def is_palce_free(self, x, y):
        id = self.get_id(x, y)
        return self.board[id] == BoardPieceStat.FREE_PAWN or self.board[id] == BoardPieceStat.FREE_WALL

    def is_diagonal(self, move):
        if self.turn:
            return abs(self.player_one_pos[0] - move[0]) == 2 and abs(self.player_one_pos[1] - move[1]) == 2
        return abs(self.player_two_pos[0] - move[0]) == 2 and abs(self.player_two_pos[1] - move[1]) == 2

    def is_jump(self, move):
        if self.turn:
            return abs(self.player_one_pos[0] - move[0]) == 4 and self.player_one_pos[1] == move[1] or abs(self.player_one_pos[1] - move[1]) == 4 and self.player_one_pos[0] == move[0]
        return abs(self.player_two_pos[0] - move[0]) == 4 and self.player_two_pos[1] == move[1] or abs(self.player_two_pos[1] - move[1]) == 4 and self.player_two_pos[0] == move[0]

    def is_goal(self):
        return self.player_one_pos[0] == 0 or self.player_two_pos[0 == 16]

    def move_pawn(self, move):
        x, y = move
        id = self.get_id(x, y)
        if self.turn:
            old_x, old_y = self.player_one_pos
            self.player_one_pos = np.array([x, y])
            self.board[id] = BoardPieceStat.OCCUPIED_BY_PLAYER_ONE
        else:
            old_x, old_y = self.player_two_pos
            self.player_two_pos = np.array([x, y])
            self.board[id] = BoardPieceStat.OCCUPIED_BY_PLAYER_TWO
        self.board[self.get_id(old_x, old_y)] = BoardPieceStat.FREE_PAWN

    def place_wall(self, place):

        # Placement of vertical walls:

        if place[0] % 2 == 0:
            if place[0] < 15:
                self.board[place[0] * self.cols + place[1]] = self.board[place[0] +
                                                                         2 * self.cols + place[1]] = BoardPieceStat.OCCUPIED_WALL
            else:
                self.board[place[0] * self.cols + place[1]] = self.board[place[0] -
                                                                         2 * self.cols + place[1]] = BoardPieceStat.OCCUPIED_WALL

        # Placement of horizontal walls

        else:
            if place[1] < 15:
                self.board[place[0] * self.cols + place[1]] = self.board[place[0]
                                                                         * self.cols + place[1] + 2] = BoardPieceStat.OCCUPIED_WALL
            else:
                self.board[place[0] * self.cols + place[1]] = self.board[place[0]
                                                                         * self.cols + place[1] - 2] = BoardPieceStat.OCCUPIED_WALL

        if self.turn:
            self.player_one_walls -= 1
        else:
            self.player_two_walls -= 1

    def check_valid_wall(self, place):
        if self.turn and self.player_one_walls == 0:
            return False
        elif not self.turn and self.player_two_walls == 0:
            return False
        if place[0] % 2 == 0 and place[1] % 2 == 1:
            if place[0] < 15:
                if self.board[place[0] * self.cols + place[1]] == self.board[place[0] + 2 * self.cols + place[1]] == BoardPieceStat.OCCUPIED_WALL:
                    return False
            else:
                if self.board[place[0] * self.cols + place[1]] == self.board[place[0] - 2 * self.cols + place[1]] == BoardPieceStat.OCCUPIED_WALL:
                    return False
        elif place[0] % 2 == 1 and place[1] % 2 == 0:
            if place[1] < 15:
                if self.board[place[0] * self.cols + place[1]] == self.board[place[0] * self.cols + place[1] + 2] == BoardPieceStat.OCCUPIED_WALL:
                    return False
            else:
                if self.board[place[0] * self.cols + place[1]] == self.board[place[0] * self.cols + place[1] - 2] == BoardPieceStat.OCCUPIED_WALL:
                    return False

        game_state = self.test_state()
        game_state.place_wall(place)
        return astar.heuristic(game_state)


if __name__ == '__main__':
    stateGame = GameState()
    stateGame.player_stats()
    stateGame.print_board()
