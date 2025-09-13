import copy
import math
import random
import sys
import threading
from dataclasses import dataclass
from enum import Enum
from heapq import heappush, heappop
from typing import List, Optional

import pygame

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 700
BOARD_SIZE = 9
CELL_SIZE = 60
BOARD_OFFSET_X = 100
BOARD_OFFSET_Y = 100
WALL_THICKNESS = 4

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (139, 69, 19)
LIGHT_BROWN = (205, 133, 63)
BLUE = (0, 100, 200)
RED = (200, 50, 50)
GREEN = (50, 200, 50)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (64, 64, 64)
HOVER_COLOR = (255, 255, 0, 100)
YELLOW = (255, 255, 0)
WALL_COLOR = YELLOW


class GameMode(Enum):
    MENU = 0
    PVP = 1
    PVA = 2


class AIAlgorithm(Enum):
    MINIMAX = 0
    EXPECTIMAX = 1
    MONTE_CARLO = 2


class WallOrientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class Player(Enum):
    ONE = 0
    TWO = 1


@dataclass
class Wall:
    row: int
    col: int
    orientation: WallOrientation


class GameState:
    def __init__(self):
        # Player positions (row, col)
        self.player_positions = {
            Player.ONE: [8, 4],  # Bottom middle
            Player.TWO: [0, 4]  # Top middle
        }

        # Wall counts
        self.walls_remaining = {
            Player.ONE: 10,
            Player.TWO: 10
        }

        # Placed walls
        self.walls = []

        # Current player
        self.current_player = Player.ONE

        # Game state
        self.game_over = False
        self.winner = None

    def copy(self):
        new_state = GameState()
        new_state.player_positions = copy.deepcopy(self.player_positions)
        new_state.walls_remaining = copy.deepcopy(self.walls_remaining)
        new_state.walls = copy.deepcopy(self.walls)
        new_state.current_player = self.current_player
        new_state.game_over = self.game_over
        new_state.winner = self.winner
        return new_state

    def is_valid_move(self, player: Player, new_row: int, new_col: int) -> bool:
        # Check bounds
        if not (0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE):
            return False

        current_pos = self.player_positions[player]

        # Check if it's a one-square move
        row_diff = abs(new_row - current_pos[0])
        col_diff = abs(new_col - current_pos[1])

        if not ((row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)):
            return False

        # Check if there's a wall blocking the move
        if self.is_wall_between(current_pos[0], current_pos[1], new_row, new_col):
            return False

        # Check if another player is in that position
        other_player = Player.TWO if player == Player.ONE else Player.ONE
        if self.player_positions[other_player] == [new_row, new_col]:
            return False

        return True

    def is_wall_between(self, row1: int, col1: int, row2: int, col2: int) -> bool:
        for wall in self.walls:
            if wall.orientation == WallOrientation.HORIZONTAL:
                # Horizontal wall blocks vertical movement
                if col1 == col2 and abs(row1 - row2) == 1:
                    wall_row = max(row1, row2)
                    if wall.row == wall_row and wall.col <= col1 <= wall.col + 1:
                        return True
            else:  # Vertical wall
                # Vertical wall blocks horizontal movement
                if row1 == row2 and abs(col1 - col2) == 1:
                    wall_col = max(col1, col2)
                    if wall.col == wall_col and wall.row <= row1 <= wall.row + 1:
                        return True
        return False

    def is_valid_wall_placement(self, wall: Wall) -> bool:
        # Check bounds
        if wall.orientation == WallOrientation.HORIZONTAL:
            if not (0 <= wall.row <= BOARD_SIZE and 0 <= wall.col <= BOARD_SIZE - 2):
                return False
        else:  # Vertical
            if not (0 <= wall.row <= BOARD_SIZE - 2 and 0 <= wall.col <= BOARD_SIZE):
                return False

        # Check for overlapping walls
        for existing_wall in self.walls:
            if self.walls_overlap(wall, existing_wall):
                return False

        # Check if wall would block all paths using pathfinding
        temp_walls = self.walls + [wall]
        if not self.path_exists_for_player(Player.ONE, temp_walls):
            return False
        if not self.path_exists_for_player(Player.TWO, temp_walls):
            return False

        return True

    def walls_overlap(self, wall1: Wall, wall2: Wall) -> bool:
        if wall1.orientation != wall2.orientation:
            return False

        if wall1.orientation == WallOrientation.HORIZONTAL:
            if wall1.row != wall2.row:
                return False
            return not (wall1.col + 1 < wall2.col or wall2.col + 1 < wall1.col)
        else:  # Vertical
            if wall1.col != wall2.col:
                return False
            return not (wall1.row + 1 < wall2.row or wall2.row + 1 < wall1.row)

    def path_exists_for_player(self, player: Player, walls_list: List[Wall]) -> bool:
        """Check if a player can reach their goal using A* pathfinding"""
        start_pos = self.player_positions[player]
        goal_row = 0 if player == Player.ONE else 8

        visited = set()
        open_list = []
        heappush(open_list, (0, start_pos[0], start_pos[1], 0))

        while open_list:
            _, row, col, cost = heappop(open_list)

            if (row, col) in visited:
                continue
            visited.add((row, col))

            # Check if reached goal
            if row == goal_row:
                return True

            # Try all four directions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc

                # Check bounds
                if not (0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE):
                    continue

                # Check if the wall blocks this move
                wall_blocked = False
                for wall in walls_list:
                    if wall.orientation == WallOrientation.HORIZONTAL:
                        # Horizontal wall blocks vertical movement
                        if dc == 0 and abs(dr) == 1:
                            wall_row = max(row, new_row)
                            if wall.row == wall_row and wall.col <= col <= wall.col + 1:
                                wall_blocked = True
                                break
                    else:  # Vertical wall
                        # Vertical wall blocks horizontal movement
                        if dr == 0 and abs(dc) == 1:
                            wall_col = max(col, new_col)
                            if wall.col == wall_col and wall.row <= row <= wall.row + 1:
                                wall_blocked = True
                                break

                if not wall_blocked and (new_row, new_col) not in visited:
                    heuristic = abs(new_row - goal_row)
                    heappush(open_list, (cost + 1 + heuristic, new_row, new_col, cost + 1))

        return False

    def get_shortest_path_length(self, player: Player) -> int:
        """Get the shortest path length to goal for heuristic evaluation"""
        start_pos = self.player_positions[player]
        goal_row = 0 if player == Player.ONE else 8

        visited = set()
        open_list = []
        heappush(open_list, (0, start_pos[0], start_pos[1], 0))

        while open_list:
            _, row, col, cost = heappop(open_list)

            if (row, col) in visited:
                continue
            visited.add((row, col))

            if row == goal_row:
                return cost

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc

                if not (0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE):
                    continue

                wall_blocked = False
                for wall in self.walls:
                    if wall.orientation == WallOrientation.HORIZONTAL:
                        if dc == 0 and abs(dr) == 1:
                            wall_row = max(row, new_row)
                            if wall.row == wall_row and wall.col <= col <= wall.col + 1:
                                wall_blocked = True
                                break
                    else:
                        if dr == 0 and abs(dc) == 1:
                            wall_col = max(col, new_col)
                            if wall.col == wall_col and wall.row <= row <= wall.row + 1:
                                wall_blocked = True
                                break

                if not wall_blocked and (new_row, new_col) not in visited:
                    heuristic = abs(new_row - goal_row)
                    heappush(open_list, (cost + 1 + heuristic, new_row, new_col, cost + 1))

        return 999  # No path found

    def get_all_possible_moves(self, player: Player) -> List['GameState']:
        """Get all possible game states after one move"""
        moves = []

        # Try all possible pawn moves
        current_pos = self.player_positions[player]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = current_pos[0] + dr, current_pos[1] + dc
            if self.is_valid_move(player, new_row, new_col):
                new_state = self.copy()
                new_state.make_move(player, new_row, new_col)
                moves.append(new_state)

        # Try all possible wall placements
        if self.walls_remaining[player] > 0:
            for row in range(BOARD_SIZE + 1):
                for col in range(BOARD_SIZE + 1):
                    for orientation in [WallOrientation.HORIZONTAL, WallOrientation.VERTICAL]:
                        wall = Wall(row, col, orientation)
                        if self.is_valid_wall_placement(wall):
                            new_state = self.copy()
                            if new_state.place_wall(player, wall):
                                moves.append(new_state)

        return moves

    def evaluate_position(self, maximizing_player: Player) -> float:
        """Heuristic evaluation function"""
        if self.game_over:
            if self.winner == maximizing_player:
                return 1000
            else:
                return -1000

        player1_distance = self.get_shortest_path_length(Player.ONE)
        player2_distance = self.get_shortest_path_length(Player.TWO)

        if maximizing_player == Player.ONE:
            # Player ONE wants to minimize their distance and maximize opponent's
            score = player2_distance - player1_distance
            score += (self.walls_remaining[Player.ONE] - self.walls_remaining[Player.TWO]) * 2
            return score
        else:
            # Player TWO wants to minimize their distance and maximize opponent's
            score = player1_distance - player2_distance
            score += (self.walls_remaining[Player.TWO] - self.walls_remaining[Player.ONE]) * 2
            return score

    def make_move(self, player: Player, new_row: int, new_col: int):
        if self.is_valid_move(player, new_row, new_col):
            self.player_positions[player] = [new_row, new_col]
            self.check_win_condition()
            return True
        return False

    def place_wall(self, player: Player, wall: Wall):
        if (self.walls_remaining[player] > 0 and
                self.is_valid_wall_placement(wall)):
            self.walls.append(wall)
            self.walls_remaining[player] -= 1
            return True
        return False

    def check_win_condition(self):
        # Player ONE wins by reaching row 0
        if self.player_positions[Player.ONE][0] == 0:
            self.game_over = True
            self.winner = Player.ONE

        # Player TWO wins by reaching row 8
        elif self.player_positions[Player.TWO][0] == 8:
            self.game_over = True
            self.winner = Player.TWO

    def switch_player(self):
        self.current_player = Player.TWO if self.current_player == Player.ONE else Player.ONE


class AIPlayer:
    def __init__(self, algorithm: AIAlgorithm):
        self.algorithm = algorithm

    def get_best_move(self, game_state: GameState) -> Optional[GameState]:
        """Get the best move according to the selected algorithm - optimized version"""
        if self.algorithm == AIAlgorithm.MINIMAX:
            return self.minimax_move(game_state)
        elif self.algorithm == AIAlgorithm.EXPECTIMAX:
            return self.expectimax_move(game_state)
        elif self.algorithm == AIAlgorithm.MONTE_CARLO:
            return self.monte_carlo_move(game_state)
        return None

    def minimax_move(self, game_state: GameState, depth=2) -> Optional[GameState]:
        """Optimized minimax with alpha-beta pruning"""
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        # Get all possible moves for current player (AI)
        possible_moves = game_state.get_all_possible_moves(game_state.current_player)

        # Prioritize moves: pawn moves first (usually faster), then walls
        possible_moves.sort(key=lambda move: self._move_priority(game_state, move))

        for move in possible_moves:
            # Evaluate this move
            value = self._minimax(move, depth - 1, alpha, beta, False, game_state.current_player)

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, value)
            if beta <= alpha:  # Alpha-beta pruning
                break

        return best_move

    def _minimax(self, state: GameState, depth: int, alpha: float, beta: float,
                 maximizing_player: bool, ai_player: Player) -> float:
        """Core minimax algorithm with alpha-beta pruning"""
        if depth == 0 or state.game_over:
            return self._evaluate_position(state, ai_player)

        current_player = state.current_player
        possible_moves = state.get_all_possible_moves(current_player)

        if not possible_moves:
            return self._evaluate_position(state, ai_player)

        if maximizing_player:
            max_eval = float('-inf')
            for move in possible_moves:
                move.switch_player()  # Switch turn for next evaluation
                eval_score = self._minimax(move, depth - 1, alpha, beta, False, ai_player)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in possible_moves:
                move.switch_player()  # Switch turn for next evaluation
                eval_score = self._minimax(move, depth - 1, alpha, beta, True, ai_player)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def expectimax_move(self, game_state: GameState, depth=2) -> Optional[GameState]:
        """Optimized expectimax algorithm"""
        best_move = None
        best_value = float('-inf')

        possible_moves = game_state.get_all_possible_moves(game_state.current_player)
        possible_moves.sort(key=lambda move: self._move_priority(game_state, move))

        for move in possible_moves:
            value = self._expectimax(move, depth - 1, False, game_state.current_player)

            if value > best_value:
                best_value = value
                best_move = move

        return best_move

    def _expectimax(self, state: GameState, depth: int, maximizing_player: bool, ai_player: Player) -> float:
        """Core expectimax algorithm"""
        if depth == 0 or state.game_over:
            return self._evaluate_position(state, ai_player)

        possible_moves = state.get_all_possible_moves(state.current_player)
        if not possible_moves:
            return self._evaluate_position(state, ai_player)

        if maximizing_player:
            # AI maximizes
            values = []
            for move in possible_moves:
                move.switch_player()
                values.append(self._expectimax(move, depth - 1, False, ai_player))
            return max(values)
        else:
            # Human player - expect average of all moves
            values = []
            for move in possible_moves:
                move.switch_player()
                values.append(self._expectimax(move, depth - 1, True, ai_player))
            return sum(values) / len(values) if values else 0

    def monte_carlo_move(self, game_state: GameState) -> Optional[GameState]:
        """Monte Carlo Tree Search - adapted from your implementation"""
        if game_state.game_over:
            return None

        root = MCTSNode(game_state, ai_player=game_state.current_player)

        # Determine number of simulations based on remaining walls
        if game_state.walls_remaining[game_state.current_player] > 0:
            simulations = 20  # More simulations when walls available
        else:
            simulations = 30  # More simulations in endgame

        # Run MCTS simulations
        for _ in range(simulations):
            # Selection and expansion
            node = root.select_and_expand()

            # Simulation (rollout)
            result = node.rollout()

            # Backpropagation
            node.backpropagate(result)

        # Return best move
        if root.children:
            best_child = root.best_child(c_param=0.0)  # Exploitation only for final choice
            return best_child.state

        return None

    def _move_priority(self, original_state: GameState, move_state: GameState) -> int:
        """Priority function for move ordering (lower = higher priority)"""
        # Prioritize pawn moves over wall placements for faster search
        original_pos = original_state.player_positions[original_state.current_player]
        new_pos = move_state.player_positions[original_state.current_player]

        if original_pos != new_pos:
            # This is a pawn move - higher priority
            return 0
        else:
            # This is a wall placement - lower priority
            return 1

    def _evaluate_position(self, state: GameState, ai_player: Player) -> float:
        """Evaluation function - will be replaced with your heuristic"""
        if state.game_over:
            if state.winner == ai_player:
                return 1000
            elif state.winner is not None:
                return -1000
            else:
                return 0

        # Get shortest path distances
        ai_distance = state.get_shortest_path_length(ai_player)
        opponent = Player.TWO if ai_player == Player.ONE else Player.ONE
        opponent_distance = state.get_shortest_path_length(opponent)

        # Basic evaluation: minimize AI distance, maximize opponent distance
        score = opponent_distance - ai_distance

        # Add wall advantage
        wall_advantage = (state.walls_remaining[ai_player] - state.walls_remaining[opponent]) * 2
        score += wall_advantage

        return score


class MCTSNode:
    """Monte Carlo Tree Search Node - adapted from your implementation"""

    def __init__(self, state: GameState, parent=None, ai_player=None):
        self.state = state
        self.parent = parent
        self.ai_player = ai_player if ai_player else state.current_player
        self.children = []
        self.visits = 0
        self.wins = 0
        self._untried_moves = None

    def untried_moves(self):
        """Get untried moves for this node"""
        if self._untried_moves is None:
            self._untried_moves = self.state.get_all_possible_moves(self.state.current_player)
        return self._untried_moves

    def is_fully_expanded(self):
        """Check if all possible moves have been tried"""
        return len(self.untried_moves()) == 0

    def is_terminal(self):
        """Check if this is a terminal node"""
        return self.state.game_over

    def select_and_expand(self):
        """Selection and expansion phase of MCTS"""
        node = self

        # Selection: traverse down the tree using UCB1
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()

        # Expansion: add a new child if possible
        if not node.is_terminal():
            untried = node.untried_moves()
            if untried:
                move_state = untried.pop()
                child = MCTSNode(move_state, parent=node, ai_player=self.ai_player)
                node.children.append(child)
                return child

        return node

    def fast_rollout(self):
        """Faster rollout with early termination and heuristic guidance"""
        current_state = self.state.copy()
        max_moves = 50  # Reduced max moves for faster rollouts
        moves = 0

        while not current_state.game_over and moves < max_moves:
            possible_moves = current_state.get_all_possible_moves(current_state.current_player)
            if not possible_moves:
                break

            # Slightly biased random selection - prefer moves toward goal
            if len(possible_moves) > 1:
                # Quick heuristic: prefer pawn moves toward goal
                pawn_moves = []
                wall_moves = []

                for move in possible_moves:
                    original_pos = current_state.player_positions[current_state.current_player]
                    new_pos = move.player_positions[current_state.current_player]

                    if original_pos != new_pos:
                        pawn_moves.append(move)
                    else:
                        wall_moves.append(move)

                # 70% chance to prefer pawn moves if available
                if pawn_moves and (not wall_moves or random.random() < 0.7):
                    move = random.choice(pawn_moves)
                else:
                    move = random.choice(possible_moves)
            else:
                move = possible_moves[0]

            current_state = move
            current_state.switch_player()
            moves += 1

        # Evaluation-based result if game didn't finish
        if current_state.game_over:
            if current_state.winner == self.ai_player:
                return 1.0
            elif current_state.winner is not None:
                return 0.0
            else:
                return 0.5
        else:
            # Use distance-based heuristic for incomplete games
            ai_dist = current_state.get_shortest_path_length(self.ai_player)
            opponent = Player.TWO if self.ai_player == Player.ONE else Player.ONE
            opp_dist = current_state.get_shortest_path_length(opponent)

            if ai_dist == 999:  # No path
                return 0.0
            elif opp_dist == 999:  # Opponent blocked
                return 1.0
            elif ai_dist < opp_dist:
                return 0.8
            elif opp_dist < ai_dist:
                return 0.2
            else:
                return 0.5

    def rollout(self):
        """Original rollout method - kept for compatibility"""
        return self.fast_rollout()

    def backpropagate(self, result):
        """Backpropagation phase"""
        node = self
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

    def best_child(self, c_param=1.4):
        """Select best child using UCB1 formula"""
        if not self.children:
            return None

        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                weight = float('inf')
            else:
                exploitation = child.wins / child.visits
                exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
                weight = exploitation + exploration
            choices_weights.append(weight)

        best_idx = choices_weights.index(max(choices_weights))
        return self.children[best_idx]


class QuoridorGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Quoridor")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # Game state
        self.game_mode = GameMode.MENU
        self.ai_algorithm = AIAlgorithm.MINIMAX
        self.game_state = None
        self.ai_player = None

        # UI state
        self.wall_orientation = WallOrientation.HORIZONTAL
        self.hover_wall = None

        # AI threading / state
        self.ai_thinking = False
        self.ai_lock = threading.Lock()

        # Menu buttons
        self.menu_buttons = []
        self.ai_buttons = []
        self.show_ai_selection = False

        # Back to the menu button
        self.back_button = pygame.Rect(WINDOW_WIDTH - 150, WINDOW_HEIGHT - 50, 120, 40)

        self.setup_menu_buttons()

    def setup_menu_buttons(self):
        # Main menu buttons
        self.menu_buttons = [
            {
                'text': 'Player vs Player',
                'rect': pygame.Rect(WINDOW_WIDTH // 2 - 150, 250, 300, 50),
                'action': 'pvp'
            },
            {
                'text': 'Player vs AI',
                'rect': pygame.Rect(WINDOW_WIDTH // 2 - 150, 320, 300, 50),
                'action': 'pva'
            }
        ]

        # AI selection buttons
        self.ai_buttons = [
            {
                'text': 'Minimax',
                'rect': pygame.Rect(WINDOW_WIDTH // 2 - 150, 300, 300, 40),
                'algorithm': AIAlgorithm.MINIMAX
            },
            {
                'text': 'Expectimax',
                'rect': pygame.Rect(WINDOW_WIDTH // 2 - 150, 360, 300, 40),
                'algorithm': AIAlgorithm.EXPECTIMAX
            },
            {
                'text': 'Monte Carlo',
                'rect': pygame.Rect(WINDOW_WIDTH // 2 - 150, 420, 300, 40),
                'algorithm': AIAlgorithm.MONTE_CARLO
            }
        ]

    def handle_menu_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_pos = pygame.mouse.get_pos()

                if self.show_ai_selection:
                    # Handle AI algorithm selection
                    for button in self.ai_buttons:
                        if button['rect'].collidepoint(mouse_pos):
                            self.ai_algorithm = button['algorithm']
                            self.ai_player = AIPlayer(self.ai_algorithm)
                            self.game_mode = GameMode.PVA
                            self.game_state = GameState()
                            self.show_ai_selection = False
                            break
                else:
                    # Handle main menu selection
                    for button in self.menu_buttons:
                        if button['rect'].collidepoint(mouse_pos):
                            if button['action'] == 'pvp':
                                self.game_mode = GameMode.PVP
                                self.game_state = GameState()
                            elif button['action'] == 'pva':
                                self.show_ai_selection = True
                            break

    def handle_game_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_pos = pygame.mouse.get_pos()

                # Check if the back button was clicked
                if self.back_button.collidepoint(mouse_pos):
                    self.game_mode = GameMode.MENU
                    self.show_ai_selection = False
                    return

        if self.game_state.game_over:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.game_state = GameState()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                self.game_mode = GameMode.MENU
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                # Toggle wall orientation
                self.wall_orientation = (WallOrientation.VERTICAL
                                         if self.wall_orientation == WallOrientation.HORIZONTAL
                                         else WallOrientation.HORIZONTAL)

            # WASD movement
            elif event.key in [pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d]:
                if self.game_mode == GameMode.PVP or self.game_state.current_player == Player.ONE:
                    self.handle_player_movement(event.key)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                # Don't handle wall placement if clicking on the back button
                mouse_pos = pygame.mouse.get_pos()
                if not self.back_button.collidepoint(mouse_pos):
                    self.handle_wall_placement(mouse_pos)

        elif event.type == pygame.MOUSEMOTION:
            self.update_wall_hover(pygame.mouse.get_pos())

    def handle_player_movement(self, key):
        # Ignore human movement while AI is thinking
        if self.game_mode == GameMode.PVA and self.ai_thinking:
            return

        current_pos = self.game_state.player_positions[self.game_state.current_player]
        new_row, new_col = current_pos[0], current_pos[1]

        if key == pygame.K_w:
            new_row -= 1
        elif key == pygame.K_s:
            new_row += 1
        elif key == pygame.K_a:
            new_col -= 1
        elif key == pygame.K_d:
            new_col += 1

        if self.game_state.make_move(self.game_state.current_player, new_row, new_col):
            if not self.game_state.game_over:
                self.game_state.switch_player()

                # If it's now AI's turn, make AI move (non-blocking)
                if (self.game_mode == GameMode.PVA and
                        self.game_state.current_player == Player.TWO):
                    self.make_ai_move()

    def handle_wall_placement(self, mouse_pos):
        # Ignore wall placement while AI is thinking
        if self.game_mode == GameMode.PVA and self.ai_thinking:
            return

        if self.game_state.walls_remaining[self.game_state.current_player] == 0:
            return

        wall = self.get_wall_from_mouse(mouse_pos)
        if wall and self.game_state.place_wall(self.game_state.current_player, wall):
            if not self.game_state.game_over:
                self.game_state.switch_player()

                # If it's now AI's turn, make AI move (non-blocking)
                if (self.game_mode == GameMode.PVA and
                        self.game_state.current_player == Player.TWO):
                    self.make_ai_move()

    def update_wall_hover(self, mouse_pos):
        if self.game_mode == GameMode.PVA and self.ai_thinking:
            self.hover_wall = None
            return
        self.hover_wall = self.get_wall_from_mouse(mouse_pos)

    def get_wall_from_mouse(self, mouse_pos) -> Optional[Wall]:
        x, y = mouse_pos

        # Convert screen coordinates to grid coordinates
        grid_x = (x - BOARD_OFFSET_X) / CELL_SIZE
        grid_y = (y - BOARD_OFFSET_Y) / CELL_SIZE

        if self.wall_orientation == WallOrientation.HORIZONTAL:
            # Horizontal walls are placed between rows
            row = int(grid_y + 0.5)
            col = int(grid_x)

            if 0 <= row <= BOARD_SIZE and 0 <= col <= BOARD_SIZE - 2:
                return Wall(row, col, WallOrientation.HORIZONTAL)
        else:  # Vertical
            # Vertical walls are placed between columns
            row = int(grid_y)
            col = int(grid_x + 0.5)

            if 0 <= row <= BOARD_SIZE - 2 and 0 <= col <= BOARD_SIZE:
                return Wall(row, col, WallOrientation.VERTICAL)

        return None

    def make_ai_move(self):
        # Start AI thinking in a background thread so the main loop can continue drawing
        if (self.ai_player and
                self.game_state.current_player == Player.TWO and
                not self.ai_thinking):
            threading.Thread(target=self._do_ai_move, daemon=True).start()

    def _do_ai_move(self):
        # Run AI search on a copy to avoid locking the main game state while thinking.
        self.ai_thinking = True
        try:
            state_copy = self.game_state.copy()
            best_move = self.ai_player.get_best_move(state_copy)
            if best_move:
                # Safely apply the AI's move back to the main game_state
                with self.ai_lock:
                    # Verify it's still AI's turn (defensive)
                    if self.game_state.current_player == Player.TWO and not self.game_state.game_over:
                        self.game_state = best_move
                        if not self.game_state.game_over:
                            self.game_state.switch_player()
        finally:
            self.ai_thinking = False

    def draw_menu(self):
        self.screen.fill(WHITE)

        title = self.font_large.render("QUORIDOR", True, BLACK)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 150))
        self.screen.blit(title, title_rect)

        if self.show_ai_selection:
            # AI Selection menu
            subtitle = self.font_medium.render("Select AI Algorithm:", True, BLACK)
            subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 220))
            self.screen.blit(subtitle, subtitle_rect)

            # Draw AI selection buttons
            for button in self.ai_buttons:
                # Button background
                color = LIGHT_GRAY if self.is_mouse_over_button(button) else GRAY
                pygame.draw.rect(self.screen, color, button['rect'])
                pygame.draw.rect(self.screen, BLACK, button['rect'], 2)

                # Button text
                text = self.font_medium.render(button['text'], True, BLACK)
                text_rect = text.get_rect(center=button['rect'].center)
                self.screen.blit(text, text_rect)
        else:
            # Main menu
            # Draw main menu buttons
            for button in self.menu_buttons:
                # Button background
                color = LIGHT_GRAY if self.is_mouse_over_button(button) else GRAY
                pygame.draw.rect(self.screen, color, button['rect'])
                pygame.draw.rect(self.screen, BLACK, button['rect'], 2)

                # Button text
                text = self.font_medium.render(button['text'], True, BLACK)
                text_rect = text.get_rect(center=button['rect'].center)
                self.screen.blit(text, text_rect)

    def is_mouse_over_button(self, button):
        mouse_pos = pygame.mouse.get_pos()
        return button['rect'].collidepoint(mouse_pos)

    def draw_game(self):
        self.screen.fill(WHITE)

        # Draw board
        self.draw_board()

        # Draw walls
        self.draw_walls()

        # Draw hover wall
        self.draw_hover_wall()

        # Draw players
        self.draw_players()

        # Draw UI
        self.draw_ui()

        # Draw back to the menu button
        self.draw_back_button()

        # Draw game over screen
        if self.game_state.game_over:
            self.draw_game_over()

    def draw_back_button(self):
        # Draw a yellow button with a black border
        mouse_pos = pygame.mouse.get_pos()
        button_color = LIGHT_GRAY if self.back_button.collidepoint(mouse_pos) else YELLOW

        pygame.draw.rect(self.screen, button_color, self.back_button)
        pygame.draw.rect(self.screen, BLACK, self.back_button, 2)

        # Button text
        text = self.font_small.render("Back to Menu", True, BLACK)
        text_rect = text.get_rect(center=self.back_button.center)
        self.screen.blit(text, text_rect)

    def draw_board(self):
        # Draw board background
        board_rect = pygame.Rect(BOARD_OFFSET_X - 10, BOARD_OFFSET_Y - 10,
                                 BOARD_SIZE * CELL_SIZE + 20, BOARD_SIZE * CELL_SIZE + 20)
        pygame.draw.rect(self.screen, DARK_GRAY, board_rect)

        # Draw cells
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                x = BOARD_OFFSET_X + col * CELL_SIZE
                y = BOARD_OFFSET_Y + row * CELL_SIZE

                cell_color = LIGHT_BROWN if (row + col) % 2 == 0 else BROWN
                pygame.draw.rect(self.screen, cell_color,
                                 (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(self.screen, BLACK,
                                 (x, y, CELL_SIZE, CELL_SIZE), 1)

    def draw_walls(self):
        for wall in self.game_state.walls:
            self.draw_wall(wall, WALL_COLOR)

    def draw_hover_wall(self):
        if (self.hover_wall and
                self.game_state.walls_remaining[self.game_state.current_player] > 0 and
                self.game_state.is_valid_wall_placement(self.hover_wall)):
            # Create a surface with alpha for transparency
            hover_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            self.draw_wall_on_surface(hover_surface, self.hover_wall, (255, 255, 0, 100))
            self.screen.blit(hover_surface, (0, 0))

    def draw_wall(self, wall: Wall, color):
        if wall.orientation == WallOrientation.HORIZONTAL:
            x = BOARD_OFFSET_X + wall.col * CELL_SIZE
            y = BOARD_OFFSET_Y + wall.row * CELL_SIZE - WALL_THICKNESS // 2
            width = CELL_SIZE * 2
            height = WALL_THICKNESS
        else:  # Vertical
            x = BOARD_OFFSET_X + wall.col * CELL_SIZE - WALL_THICKNESS // 2
            y = BOARD_OFFSET_Y + wall.row * CELL_SIZE
            width = WALL_THICKNESS
            height = CELL_SIZE * 2

        pygame.draw.rect(self.screen, color, (x, y, width, height))

    def draw_wall_on_surface(self, surface, wall: Wall, color):
        if wall.orientation == WallOrientation.HORIZONTAL:
            x = BOARD_OFFSET_X + wall.col * CELL_SIZE
            y = BOARD_OFFSET_Y + wall.row * CELL_SIZE - WALL_THICKNESS // 2
            width = CELL_SIZE * 2
            height = WALL_THICKNESS
        else:  # Vertical
            x = BOARD_OFFSET_X + wall.col * CELL_SIZE - WALL_THICKNESS // 2
            y = BOARD_OFFSET_Y + wall.row * CELL_SIZE
            width = WALL_THICKNESS
            height = CELL_SIZE * 2

        pygame.draw.rect(surface, color, (x, y, width, height))

    def draw_players(self):
        for player, pos in self.game_state.player_positions.items():
            x = BOARD_OFFSET_X + pos[1] * CELL_SIZE + CELL_SIZE // 2
            y = BOARD_OFFSET_Y + pos[0] * CELL_SIZE + CELL_SIZE // 2

            color = BLUE if player == Player.ONE else RED
            pygame.draw.circle(self.screen, color, (x, y), 20)
            pygame.draw.circle(self.screen, BLACK, (x, y), 20, 2)

    def draw_ui(self):
        # Current player indicator
        if self.game_mode == GameMode.PVP:
            player_text = f"Current Player: {'Player 1 (Blue)' if self.game_state.current_player == Player.ONE else 'Player 2 (Red)'}"
        else:  # PVA
            player_text = f"Current Player: {'Human (Blue)' if self.game_state.current_player == Player.ONE else 'AI (Red)'}"

        text = self.font_medium.render(player_text, True, BLACK)
        self.screen.blit(text, (BOARD_OFFSET_X + BOARD_SIZE * CELL_SIZE + 50, BOARD_OFFSET_Y))

        # AI Algorithm indicator
        if self.game_mode == GameMode.PVA:
            algorithm_names = {
                AIAlgorithm.MINIMAX: "Minimax",
                AIAlgorithm.EXPECTIMAX: "Expectimax",
                AIAlgorithm.MONTE_CARLO: "Monte Carlo"
            }
            ai_text = f"AI Algorithm: {algorithm_names[self.ai_algorithm]}"
            text = self.font_small.render(ai_text, True, BLACK)
            self.screen.blit(text, (BOARD_OFFSET_X + BOARD_SIZE * CELL_SIZE + 50, BOARD_OFFSET_Y + 35))

        # AI thinking overlay / indicator
        if self.game_mode == GameMode.PVA and self.ai_thinking:
            thinking_text = self.font_medium.render("AI thinking...", True, RED)
            self.screen.blit(thinking_text, (BOARD_OFFSET_X + BOARD_SIZE * CELL_SIZE + 50, BOARD_OFFSET_Y + 120))

        # Wall counts
        walls_p1 = f"Player 1 Walls: {self.game_state.walls_remaining[Player.ONE]}"
        walls_p2 = f"Player 2 Walls: {self.game_state.walls_remaining[Player.TWO]}"

        text1 = self.font_small.render(walls_p1, True, BLACK)
        text2 = self.font_small.render(walls_p2, True, BLACK)

        y_offset = 70 if self.game_mode == GameMode.PVA else 50
        self.screen.blit(text1, (BOARD_OFFSET_X + BOARD_SIZE * CELL_SIZE + 50, BOARD_OFFSET_Y + y_offset))
        self.screen.blit(text2, (BOARD_OFFSET_X + BOARD_SIZE * CELL_SIZE + 50, BOARD_OFFSET_Y + y_offset + 30))

        # Controls
        controls = [
            "Controls:",
            "WASD - Move pawn",
            "Mouse - Place walls",
            "Q - Rotate wall",
            "",
            f"Wall orientation: {'Horizontal' if self.wall_orientation == WallOrientation.HORIZONTAL else 'Vertical'}"
        ]

        for i, control in enumerate(controls):
            text = self.font_small.render(control, True, BLACK)
            self.screen.blit(text, (BOARD_OFFSET_X + BOARD_SIZE * CELL_SIZE + 50, BOARD_OFFSET_Y + 170 + i * 25))

    def draw_game_over(self):
        # Semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))

        # Winner text
        winner_text = f"{'Player 1' if self.game_state.winner == Player.ONE else 'Player 2'} Wins!"
        text = self.font_large.render(winner_text, True, WHITE)
        text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
        self.screen.blit(text, text_rect)

        # Options
        options = ["R - Restart", "M - Main Menu"]
        for i, option in enumerate(options):
            text = self.font_medium.render(option, True, WHITE)
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 20 + i * 40))
            self.screen.blit(text, text_rect)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if self.game_mode == GameMode.MENU:
                    self.handle_menu_events(event)
                else:
                    self.handle_game_events(event)

            # Draw
            if self.game_mode == GameMode.MENU:
                self.draw_menu()
            else:
                self.draw_game()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = QuoridorGame()
    game.run()
