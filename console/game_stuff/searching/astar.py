from typing import List, Tuple, Set
from heapq import heappush, heappop
from utils.stuff import BoardPieceStat


class Node:
    def __init__(self, position, g_cost, h_cost, parent=None):
        self.position = position
        self.g_cost = g_cost  # Cost from start to current node
        self.h_cost = h_cost  # Estimated cost from current node to goal
        self.f_cost = g_cost + h_cost  # Total cost
        self.parent = parent

    def __lt__(self, other):
        return self.f_cost < other.f_cost


def manhattan_distance(pos, goal) -> int:
    return abs(pos[0] - goal)


def get_neighbors(state, position) -> List[Tuple[int, int]]:
    x, y = position
    possible_moves = [
        (x + 2, y),  # Down
        (x - 2, y),  # Up
        (x, y + 2),  # Right
        (x, y - 2),  # Left
    ]

    valid_moves = []
    wall_offsets = {
        (x + 2, y): (x + 1, y),  # Wall check for down move
        (x - 2, y): (x - 1, y),  # Wall check for up move
        (x, y + 2): (x, y + 1),  # Wall check for right move
        (x, y - 2): (x, y - 1),  # Wall check for left move
    }

    for move in possible_moves:
        # Check if move is within board boundaries
        if 0 <= move[0] <= 16 and 0 <= move[1] <= 16:
            # Check if there's no wall blocking the path
            wall_pos = wall_offsets[move]
            if state.board[state.get_id(wall_pos[0], wall_pos[1])] != BoardPieceStat.OCCUPIED_WALL:
                valid_moves.append(move)

    return valid_moves


def path_exists(state, start_pos, is_player_one):
    goal_row = 0 if is_player_one else 16
    visited = set()
    open_list = []

    # Create start node
    start_node = Node(
        position=start_pos,
        g_cost=0,
        h_cost=manhattan_distance(start_pos, goal_row),
        parent=None
    )
    heappush(open_list, start_node)

    while open_list:
        current = heappop(open_list)

        # If we've reached the goal row
        if current.position[0] == goal_row:
            return True, current.g_cost

        # Skip if we've already visited this position
        if current.position in visited:
            continue

        visited.add(current.position)

        # Check all neighboring positions
        for neighbor_pos in get_neighbors(state, current.position):
            if neighbor_pos not in visited:
                g_cost = current.g_cost + 1
                h_cost = manhattan_distance(neighbor_pos, goal_row)

                neighbor = Node(
                    position=neighbor_pos,
                    g_cost=g_cost,
                    h_cost=h_cost,
                    parent=current
                )
                heappush(open_list, neighbor)

    return False
