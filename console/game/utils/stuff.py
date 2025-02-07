class Stuff:
    PLAYER_ONE = "\u2B55"
    PLAYER_TWO = "\u2B50"
    WALL_COLOR = "\u001b[33m"
    COLOR_RESET = "\u001b[0m"

class BoardPieceStat:
    OCCUPIED_BY_PLAYER_1 = 1
    OCCUPIED_BY_PLAYER_2 = 2
    FREE_PLAYER = 3
    FREE_WALL = 4
    OCCUPIED_WALL = 5

class Color:
    BLACK = "\u001b[30m"
    RED = "\u001b[31m"
    GREEN = "\u001b[32m"
    YELLOW = "\u001b[33m"
    BLUE = "\u001b[34m"
    MAGENTA = "\u001b[35m"
    CYAN = "\u001b[36m"
    WHITE = "\u001b[37m"
    RESET = "\u001b[0m"
    BROWN = "\033[0;33m"
    BACKGROUND_YELLOW = "\u001b[43m"

class WallDirection:
    NORTH = 0,
    SOUTH = 1,
    EAST = 2,
    WEST = 3
