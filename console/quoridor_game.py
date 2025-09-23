import copy
import math
import random
import sys
import threading
from dataclasses import dataclass
from enum import Enum
from heapq import heappush, heappop
from typing import List, Optional
from constants import *

import pygame

pygame.init()

pygame.mixer.init()

sound_place_wall = pygame.mixer.Sound(SOUND_PLACE_WALL)
sound_move_pawn = pygame.mixer.Sound(SOUND_MOVE_PAWN)
sound_winner = pygame.mixer.Sound(SOUND_WINNER)


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
        self.player_positions = {
            Player.ONE: [8, 4],
            Player.TWO: [0, 4]
        }
        self.walls_remaining = {
            Player.ONE: 10,
            Player.TWO: 10
        }
        self.walls = []
        self.current_player = Player.ONE
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

    def get_opponent(self, player: Player) -> Player:
        return Player.TWO if player == Player.ONE else Player.ONE

    def is_within_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def is_wall_between_coords(self, walls_list: List[Wall], row1: int, col1: int, row2: int, col2: int) -> bool:
        for wall in walls_list:
            if wall.orientation == WallOrientation.HORIZONTAL:
                if col1 == col2 and abs(row1 - row2) == 1:
                    wall_row = max(row1, row2)
                    if wall.row == wall_row and wall.col <= col1 <= wall.col + 1:
                        return True
            else:
                if row1 == row2 and abs(col1 - col2) == 1:
                    wall_col = max(col1, col2)
                    if wall.col == wall_col and wall.row <= row1 <= wall.row + 1:
                        return True
        return False

    def get_pawn_neighbors(self, player: Player, walls_list: Optional[List[Wall]] = None) -> List[tuple]:
        if walls_list is None:
            walls_list = self.walls
        cur_row, cur_col = self.player_positions[player]
        opp = self.get_opponent(player)
        opp_row, opp_col = self.player_positions[opp]
        neighbors = []
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in dirs:
            adj_row = cur_row + dr
            adj_col = cur_col + dc
            if not self.is_within_bounds(adj_row, adj_col):
                continue
            if self.is_wall_between_coords(walls_list, cur_row, cur_col, adj_row, adj_col):
                continue
            if [adj_row, adj_col] != [opp_row, opp_col]:
                neighbors.append((adj_row, adj_col))
            else:
                jump_row = adj_row + dr
                jump_col = adj_col + dc
                can_jump = False
                if self.is_within_bounds(jump_row, jump_col):
                    if not self.is_wall_between_coords(walls_list, adj_row, adj_col, jump_row, jump_col):
                        if [jump_row, jump_col] != [cur_row, cur_col]:
                            can_jump = True
                if can_jump:
                    neighbors.append((jump_row, jump_col))
                else:
                    if dr != 0:
                        perps = [(0, -1), (0, 1)]
                    else:
                        perps = [(-1, 0), (1, 0)]
                    for pdr, pdc in perps:
                        diag_row = adj_row + pdr
                        diag_col = adj_col + pdc
                        if not self.is_within_bounds(diag_row, diag_col):
                            continue
                        if self.is_wall_between_coords(walls_list, adj_row, adj_col, diag_row, diag_col):
                            continue
                        if [diag_row, diag_col] != [cur_row, cur_col] and [diag_row, diag_col] != [opp_row, opp_col]:
                            neighbors.append((diag_row, diag_col))
        unique_neighbors = []
        seen = set()
        for r, c in neighbors:
            if (r, c) not in seen:
                unique_neighbors.append((r, c))
                seen.add((r, c))
        return unique_neighbors

    def is_valid_move(self, player: Player, new_row: int, new_col: int) -> bool:
        if not (0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE):
            return False
        current_pos = self.player_positions[player]
        row_diff = abs(new_row - current_pos[0])
        col_diff = abs(new_col - current_pos[1])
        if not ((row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)):
            return False
        if self.is_wall_between_coords(self.walls, current_pos[0], current_pos[1], new_row, new_col):
            return False
        other_player = self.get_opponent(player)
        if self.player_positions[other_player] == [new_row, new_col]:
            return False
        return True

    def is_valid_wall_placement(self, wall: Wall) -> bool:
        if wall.orientation == WallOrientation.HORIZONTAL:
            if not (0 <= wall.row <= BOARD_SIZE and 0 <= wall.col <= BOARD_SIZE - 2):
                return False
        else:
            if not (0 <= wall.row <= BOARD_SIZE - 2 and 0 <= wall.col <= BOARD_SIZE):
                return False
        for existing_wall in self.walls:
            if self.walls_overlap(wall, existing_wall):
                return False
            if wall.orientation != existing_wall.orientation:
                if wall.orientation == WallOrientation.HORIZONTAL and existing_wall.orientation == WallOrientation.VERTICAL:
                    if existing_wall.row == wall.row - 1 and existing_wall.col == wall.col + 1:
                        return False
                if wall.orientation == WallOrientation.VERTICAL and existing_wall.orientation == WallOrientation.HORIZONTAL:
                    if existing_wall.row == wall.row + 1 and existing_wall.col == wall.col - 1:
                        return False
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
        else:
            if wall1.col != wall2.col:
                return False
            return not (wall1.row + 1 < wall2.row or wall2.row + 1 < wall1.row)

    def path_exists_for_player(self, player: Player, walls_list: List[Wall]) -> bool:
        start_pos = self.player_positions[player]
        goal_row = 0 if player == Player.ONE else BOARD_SIZE - 1
        visited = set()
        open_list = []
        heappush(open_list, (0, start_pos[0], start_pos[1], 0))
        while open_list:
            _, row, col, cost = heappop(open_list)
            if (row, col) in visited:
                continue
            visited.add((row, col))
            if row == goal_row:
                return True
            original_pos = self.player_positions[player]
            self.player_positions[player] = [row, col]
            neighbors = self.get_pawn_neighbors(player, walls_list)
            self.player_positions[player] = original_pos
            for new_row, new_col in neighbors:
                if (new_row, new_col) not in visited:
                    heuristic = abs(new_row - goal_row)
                    heappush(open_list, (cost + 1 + heuristic, new_row, new_col, cost + 1))
        return False

    def get_shortest_path_length(self, player: Player) -> int:
        start_pos = self.player_positions[player]
        goal_row = 0 if player == Player.ONE else BOARD_SIZE - 1
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
            original_pos = self.player_positions[player]
            self.player_positions[player] = [row, col]
            neighbors = self.get_pawn_neighbors(player, self.walls)
            self.player_positions[player] = original_pos
            for new_row, new_col in neighbors:
                if (new_row, new_col) not in visited:
                    heuristic = abs(new_row - goal_row)
                    heappush(open_list, (cost + 1 + heuristic, new_row, new_col, cost + 1))
        return 999

    def get_all_possible_moves(self, player: Player) -> List['GameState']:
        moves = []
        pawn_moves = self.get_pawn_neighbors(player, self.walls)
        for new_row, new_col in pawn_moves:
            new_state = self.copy()
            new_state.player_positions[player] = [new_row, new_col]
            new_state.check_win_condition()
            moves.append(new_state)
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
        if self.game_over:
            if self.winner == maximizing_player:
                return 1000
            else:
                return -1000
        player1_distance = self.get_shortest_path_length(Player.ONE)
        player2_distance = self.get_shortest_path_length(Player.TWO)
        if maximizing_player == Player.ONE:
            score = player2_distance - player1_distance
            score += (self.walls_remaining[Player.ONE] - self.walls_remaining[Player.TWO]) * 2
            return score
        else:
            score = player1_distance - player2_distance
            score += (self.walls_remaining[Player.TWO] - self.walls_remaining[Player.ONE]) * 2
            return score

    def make_move(self, player: Player, new_row: int, new_col: int):
        valid = self.get_pawn_neighbors(player, self.walls)
        if (new_row, new_col) in valid:
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
        if self.player_positions[Player.ONE][0] == 0:
            self.game_over = True
            self.winner = Player.ONE
        elif self.player_positions[Player.TWO][0] == BOARD_SIZE - 1:
            self.game_over = True
            self.winner = Player.TWO

    def switch_player(self):
        self.current_player = Player.TWO if self.current_player == Player.ONE else Player.ONE


class AIPlayer:
    def __init__(self, algorithm: AIAlgorithm):
        self.algorithm = algorithm

    def get_best_move(self, game_state: GameState) -> Optional[GameState]:
        if self.algorithm == AIAlgorithm.MINIMAX:
            return self.minimax_move(game_state)
        elif self.algorithm == AIAlgorithm.EXPECTIMAX:
            return self.expectimax_move(game_state)
        elif self.algorithm == AIAlgorithm.MONTE_CARLO:
            return self.monte_carlo_move(game_state)
        return None

    def minimax_move(self, game_state: GameState, depth=2) -> Optional[GameState]:
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        possible_moves = game_state.get_all_possible_moves(game_state.current_player)
        possible_moves.sort(key=lambda move: self._move_priority(game_state, move))
        for move in possible_moves:
            value = self._minimax(move, depth - 1, alpha, beta, False, game_state.current_player)
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return best_move

    def _minimax(self, state: GameState, depth: int, alpha: float, beta: float,
                 maximizing_player: bool, ai_player: Player) -> float:
        if depth == 0 or state.game_over:
            return self._evaluate_position(state, ai_player)
        current_player = state.current_player
        possible_moves = state.get_all_possible_moves(current_player)
        if not possible_moves:
            return self._evaluate_position(state, ai_player)
        if maximizing_player:
            max_eval = float('-inf')
            for move in possible_moves:
                move.switch_player()
                eval_score = self._minimax(move, depth - 1, alpha, beta, False, ai_player)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in possible_moves:
                move.switch_player()
                eval_score = self._minimax(move, depth - 1, alpha, beta, True, ai_player)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def expectimax_move(self, game_state: GameState, depth=2) -> Optional[GameState]:
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
        if depth == 0 or state.game_over:
            return self._evaluate_position(state, ai_player)
        possible_moves = state.get_all_possible_moves(state.current_player)
        if not possible_moves:
            return self._evaluate_position(state, ai_player)
        if maximizing_player:
            values = []
            for move in possible_moves:
                move.switch_player()
                values.append(self._expectimax(move, depth - 1, False, ai_player))
            return max(values)
        else:
            values = []
            for move in possible_moves:
                move.switch_player()
                values.append(self._expectimax(move, depth - 1, True, ai_player))
            return sum(values) / len(values) if values else 0

    def monte_carlo_move(self, game_state: GameState) -> Optional[GameState]:
        if game_state.game_over:
            return None
        root = MCTSNode(game_state, ai_player=game_state.current_player)
        if game_state.walls_remaining[game_state.current_player] > 0:
            simulations = 20
        else:
            simulations = 30
        for _ in range(simulations):
            node = root.select_and_expand()
            result = node.rollout()
            node.backpropagate(result)
        if root.children:
            best_child = root.best_child(c_param=0.0)
            return best_child.state
        return None

    def _move_priority(self, original_state: GameState, move_state: GameState) -> int:
        original_pos = original_state.player_positions[original_state.current_player]
        new_pos = move_state.player_positions[original_state.current_player]
        if original_pos != new_pos:
            return 0
        else:
            return 1

    def _evaluate_position(self, state: GameState, ai_player: Player) -> float:
        if state.game_over:
            if state.winner == ai_player:
                return 1000
            elif state.winner is not None:
                return -1000
            else:
                return 0
        ai_distance = state.get_shortest_path_length(ai_player)
        opponent = Player.TWO if ai_player == Player.ONE else Player.ONE
        opponent_distance = state.get_shortest_path_length(opponent)
        score = opponent_distance - ai_distance
        wall_advantage = (state.walls_remaining[ai_player] - state.walls_remaining[opponent]) * 2
        score += wall_advantage
        return score


class MCTSNode:
    def __init__(self, state: GameState, parent=None, ai_player=None):
        self.state = state
        self.parent = parent
        self.ai_player = ai_player if ai_player else state.current_player
        self.children = []
        self.visits = 0
        self.wins = 0
        self._untried_moves = None

    def untried_moves(self):
        if self._untried_moves is None:
            self._untried_moves = self.state.get_all_possible_moves(self.state.current_player)
        return self._untried_moves

    def is_fully_expanded(self):
        return len(self.untried_moves()) == 0

    def is_terminal(self):
        return self.state.game_over

    def select_and_expand(self):
        node = self
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()
        if not node.is_terminal():
            untried = node.untried_moves()
            if untried:
                move_state = untried.pop()
                child = MCTSNode(move_state, parent=node, ai_player=self.ai_player)
                node.children.append(child)
                return child
        return node

    def fast_rollout(self):
        current_state = self.state.copy()
        max_moves = 50
        moves = 0
        while not current_state.game_over and moves < max_moves:
            possible_moves = current_state.get_all_possible_moves(current_state.current_player)
            if not possible_moves:
                break
            if len(possible_moves) > 1:
                pawn_moves = []
                wall_moves = []
                for move in possible_moves:
                    original_pos = current_state.player_positions[current_state.current_player]
                    new_pos = move.player_positions[current_state.current_player]
                    if original_pos != new_pos:
                        pawn_moves.append(move)
                    else:
                        wall_moves.append(move)
                if pawn_moves and (not wall_moves or random.random() < 0.7):
                    move = random.choice(pawn_moves)
                else:
                    move = random.choice(possible_moves)
            else:
                move = possible_moves[0]
            current_state = move
            current_state.switch_player()
            moves += 1
        if current_state.game_over:
            if current_state.winner == self.ai_player:
                return 1.0
            elif current_state.winner is not None:
                return 0.0
            else:
                return 0.5
        else:
            ai_dist = current_state.get_shortest_path_length(self.ai_player)
            opponent = Player.TWO if self.ai_player == Player.ONE else Player.ONE
            opp_dist = current_state.get_shortest_path_length(opponent)
            if ai_dist == 999:
                return 0.0
            elif opp_dist == 999:
                return 1.0
            elif ai_dist < opp_dist:
                return 0.8
            elif opp_dist < ai_dist:
                return 0.2
            else:
                return 0.5

    def rollout(self):
        return self.fast_rollout()

    def backpropagate(self, result):
        node = self
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

    def best_child(self, c_param=1.4):
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
        self.played_winner_sound = False
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Quoridor")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.game_mode = GameMode.MENU
        self.ai_algorithm = AIAlgorithm.MINIMAX
        self.game_state = None
        self.ai_player = None
        self.wall_orientation = WallOrientation.HORIZONTAL
        self.hover_wall = None
        self.ai_thinking = False
        self.ai_lock = threading.Lock()
        self.menu_buttons = []
        self.ai_buttons = []
        self.show_ai_selection = False
        self.back_button = pygame.Rect(WINDOW_WIDTH - 180, WINDOW_HEIGHT - 50, 170, 40)
        self.restart_button = pygame.Rect(WINDOW_WIDTH - 300, WINDOW_HEIGHT - 50, 100, 40)
        self.setup_menu_buttons()

    def setup_menu_buttons(self):
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
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.show_ai_selection = False
                self.game_mode = GameMode.MENU
                return
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                if self.show_ai_selection:
                    if self.back_button.collidepoint(mouse_pos):
                        self.show_ai_selection = False
                        self.game_mode = GameMode.MENU
                        return
                if self.show_ai_selection:
                    for button in self.ai_buttons:
                        if button['rect'].collidepoint(mouse_pos):
                            self.ai_algorithm = button['algorithm']
                            self.ai_player = AIPlayer(self.ai_algorithm)
                            self.game_mode = GameMode.PVA
                            self.game_state = GameState()
                            self.show_ai_selection = False
                            break
                else:
                    for button in self.menu_buttons:
                        if button['rect'].collidepoint(mouse_pos):
                            if button['action'] == 'pvp':
                                self.game_mode = GameMode.PVP
                                self.game_state = GameState()
                            elif button['action'] == 'pva':
                                self.show_ai_selection = True
                            break

    def handle_game_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.game_mode = GameMode.MENU
                self.show_ai_selection = False
                return
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                if self.restart_button.collidepoint(mouse_pos):
                    self.game_state = GameState()
                    self.ai_thinking = False
                    return
                if self.back_button.collidepoint(mouse_pos):
                    self.game_mode = GameMode.MENU
                    self.show_ai_selection = False
                    return
        if self.game_state and self.game_state.game_over:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.game_state = GameState()
                self.played_winner_sound = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                self.game_mode = GameMode.MENU
                self.played_winner_sound = False
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                self.wall_orientation = (WallOrientation.VERTICAL
                                         if self.wall_orientation == WallOrientation.HORIZONTAL
                                         else WallOrientation.HORIZONTAL)
            elif event.key == pygame.K_r:
                self.game_state = GameState()
                self.ai_thinking = False
                return
            elif event.key in [pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d]:
                if self.game_mode == GameMode.PVP or self.game_state.current_player == Player.ONE:
                    self.handle_player_movement(event.key)
            elif event.key == pygame.K_j:
                if self.game_mode == GameMode.PVP or self.game_state.current_player == Player.ONE:
                    self.handle_diagonal_move(left=True)
            elif event.key == pygame.K_k:
                if self.game_mode == GameMode.PVP or self.game_state.current_player == Player.ONE:
                    self.handle_diagonal_move(left=False)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                if not self.back_button.collidepoint(mouse_pos) and not self.restart_button.collidepoint(mouse_pos):
                    self.handle_wall_placement(mouse_pos)
        elif event.type == pygame.MOUSEMOTION:
            self.update_wall_hover(pygame.mouse.get_pos())

    def handle_player_movement(self, key):
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
        opponent = self.game_state.get_opponent(self.game_state.current_player)
        opp_pos = self.game_state.player_positions[opponent]
        if [new_row, new_col] == opp_pos:
            dr = new_row - current_pos[0]
            dc = new_col - current_pos[1]
            jump_row = new_row + dr
            jump_col = new_col + dc
            can_jump = False
            if 0 <= jump_row < BOARD_SIZE and 0 <= jump_col < BOARD_SIZE:
                if not self.game_state.is_wall_between_coords(self.game_state.walls, new_row, new_col, jump_row, jump_col):
                    if self.game_state.player_positions[opponent] != [jump_row, jump_col] and self.game_state.player_positions[self.game_state.current_player] != [jump_row, jump_col]:
                        can_jump = True
            if can_jump:
                if self.game_state.make_move(self.game_state.current_player, jump_row, jump_col):
                    sound_move_pawn.play()
                    if not self.game_state.game_over:
                        self.game_state.switch_player()
                        if (self.game_mode == GameMode.PVA and
                                self.game_state.current_player == Player.TWO):
                            self.make_ai_move()
                return
            else:
                return
        if self.game_state.make_move(self.game_state.current_player, new_row, new_col):
            sound_move_pawn.play()
            if not self.game_state.game_over:
                self.game_state.switch_player()
                if (self.game_mode == GameMode.PVA and
                        self.game_state.current_player == Player.TWO):
                    self.make_ai_move()

    def handle_diagonal_move(self, left: bool):
        if self.game_mode == GameMode.PVA and self.ai_thinking:
            return
        player = self.game_state.current_player
        cur_row, cur_col = self.game_state.player_positions[player]
        opp = self.game_state.get_opponent(player)
        opp_row, opp_col = self.game_state.player_positions[opp]
        row_diff = opp_row - cur_row
        col_diff = opp_col - cur_col
        if not ((abs(row_diff) == 1 and col_diff == 0) or (abs(col_diff) == 1 and row_diff == 0)):
            return
        dr = row_diff
        dc = col_diff
        jump_row = opp_row + dr
        jump_col = opp_col + dc
        if 0 <= jump_row < BOARD_SIZE and 0 <= jump_col < BOARD_SIZE:
            if not self.game_state.is_wall_between_coords(self.game_state.walls, opp_row, opp_col, jump_row, jump_col):
                return
        if dr != 0:
            perps = [(0, -1), (0, 1)]
        else:
            perps = [(-1, 0), (1, 0)]
        idx = 0 if left else 1
        pdr, pdc = perps[idx]
        diag_row = opp_row + pdr
        diag_col = opp_col + pdc
        if not (0 <= diag_row < BOARD_SIZE and 0 <= diag_col < BOARD_SIZE):
            return
        if self.game_state.is_wall_between_coords(self.game_state.walls, opp_row, opp_col, diag_row, diag_col):
            return
        if [diag_row, diag_col] == self.game_state.player_positions[opp] or [diag_row, diag_col] == [cur_row, cur_col]:
            return
        if self.game_state.make_move(player, diag_row, diag_col):
            sound_move_pawn.play()
            if not self.game_state.game_over:
                self.game_state.switch_player()
                if (self.game_mode == GameMode.PVA and
                        self.game_state.current_player == Player.TWO):
                    self.make_ai_move()

    def handle_wall_placement(self, mouse_pos):
        if self.game_mode == GameMode.PVA and self.ai_thinking:
            return
        if self.game_state.walls_remaining[self.game_state.current_player] == 0:
            return
        wall = self.get_wall_from_mouse(mouse_pos)
        if wall and self.game_state.place_wall(self.game_state.current_player, wall):
            sound_place_wall.play()
            if not self.game_state.game_over:
                self.game_state.switch_player()
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
        grid_x = (x - BOARD_OFFSET_X) / CELL_SIZE
        grid_y = (y - BOARD_OFFSET_Y) / CELL_SIZE
        if self.wall_orientation == WallOrientation.HORIZONTAL:
            row = int(grid_y + 0.5)
            col = int(grid_x)
            if 0 <= row <= BOARD_SIZE and 0 <= col <= BOARD_SIZE - 2:
                return Wall(row, col, WallOrientation.HORIZONTAL)
        else:
            row = int(grid_y)
            col = int(grid_x + 0.5)
            if 0 <= row <= BOARD_SIZE - 2 and 0 <= col <= BOARD_SIZE:
                return Wall(row, col, WallOrientation.VERTICAL)
        return None

    def make_ai_move(self):
        if (self.ai_player and
                self.game_state.current_player == Player.TWO and
                not self.ai_thinking):
            threading.Thread(target=self._do_ai_move, daemon=True).start()

    def _do_ai_move(self):
        self.ai_thinking = True
        try:
            state_copy = self.game_state.copy()
            best_move = self.ai_player.get_best_move(state_copy)
            if best_move:
                with self.ai_lock:
                    if self.game_state.current_player == Player.TWO and not self.game_state.game_over:
                        self.game_state = best_move
                        if not self.game_state.game_over:
                            self.game_state.switch_player()
        finally:
            self.ai_thinking = False

    def draw_menu(self):
        self.screen.fill(PALE_GREEN)
        title = self.font_large.render("QUORIDOR", True, BLACK)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 150))
        self.screen.blit(title, title_rect)
        if self.show_ai_selection:
            subtitle = self.font_medium.render("Select AI Algorithm:", True, BLACK)
            subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 220))
            self.screen.blit(subtitle, subtitle_rect)
            for button in self.ai_buttons:
                color = LIGHT_GRAY if self.is_mouse_over_button(button) else GRAY
                pygame.draw.rect(self.screen, color, button['rect'])
                pygame.draw.rect(self.screen, BLACK, button['rect'], 2)
                text = self.font_medium.render(button['text'], True, BLACK)
                text_rect = text.get_rect(center=button['rect'].center)
                self.screen.blit(text, text_rect)
            mouse_pos = pygame.mouse.get_pos()
            btn_color = LIGHT_GRAY if self.back_button.collidepoint(mouse_pos) else YELLOW
            pygame.draw.rect(self.screen, btn_color, self.back_button)
            pygame.draw.rect(self.screen, BLACK, self.back_button, 2)
            text = self.font_small.render("Back to Menu (ESC)", True, BLACK)
            text_rect = text.get_rect(center=self.back_button.center)
            self.screen.blit(text, text_rect)
        else:
            for button in self.menu_buttons:
                color = LIGHT_GRAY if self.is_mouse_over_button(button) else GRAY
                pygame.draw.rect(self.screen, color, button['rect'])
                pygame.draw.rect(self.screen, BLACK, button['rect'], 2)
                text = self.font_medium.render(button['text'], True, BLACK)
                text_rect = text.get_rect(center=button['rect'].center)
                self.screen.blit(text, text_rect)

    def is_mouse_over_button(self, button):
        mouse_pos = pygame.mouse.get_pos()
        return button['rect'].collidepoint(mouse_pos)

    def draw_game(self):
        self.screen.fill(PALE_GREEN)
        self.draw_board()
        self.draw_walls()
        self.draw_hover_wall()
        self.draw_players()
        self.draw_ui()
        self.draw_restart_button()
        self.draw_back_button()
        if self.game_state.game_over:
            self.draw_game_over()

    def draw_back_button(self):
        mouse_pos = pygame.mouse.get_pos()
        button_color = LIGHT_GRAY if self.back_button.collidepoint(mouse_pos) else YELLOW
        pygame.draw.rect(self.screen, button_color, self.back_button)
        pygame.draw.rect(self.screen, BLACK, self.back_button, 2)
        text = self.font_small.render("Back to Menu (ESC)", True, BLACK)
        text_rect = text.get_rect(center=self.back_button.center)
        self.screen.blit(text, text_rect)

    def draw_restart_button(self):
        mouse_pos = pygame.mouse.get_pos()
        button_color = LIGHT_GRAY if self.restart_button.collidepoint(mouse_pos) else YELLOW
        pygame.draw.rect(self.screen, button_color, self.restart_button)
        pygame.draw.rect(self.screen, BLACK, self.restart_button, 2)
        text = self.font_small.render("Restart (R)", True, BLACK)
        text_rect = text.get_rect(center=self.restart_button.center)
        self.screen.blit(text, text_rect)

    def draw_board(self):
        board_rect = pygame.Rect(BOARD_OFFSET_X - 10, BOARD_OFFSET_Y - 10,
                                 BOARD_SIZE * CELL_SIZE + 20, BOARD_SIZE * CELL_SIZE + 20)
        pygame.draw.rect(self.screen, DARK_GRAY, board_rect)
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
            hover_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            self.draw_wall_on_surface(hover_surface, self.hover_wall, (255, 255, 0, 100))
            self.screen.blit(hover_surface, (0, 0))

    def draw_wall(self, wall: Wall, color):
        if wall.orientation == WallOrientation.HORIZONTAL:
            x = BOARD_OFFSET_X + wall.col * CELL_SIZE
            y = BOARD_OFFSET_Y + wall.row * CELL_SIZE - WALL_THICKNESS // 2
            width = CELL_SIZE * 2
            height = WALL_THICKNESS
        else:
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
        else:
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
        if self.game_mode == GameMode.PVP:
            player_text = f"Current Player: {'Player 1 (Blue)' if self.game_state.current_player == Player.ONE else 'Player 2 (Red)'}"
        else:
            player_text = f"Current Player: {'Human (Blue)' if self.game_state.current_player == Player.ONE else 'AI (Red)'}"
        text = self.font_medium.render(player_text, True, BLACK)
        self.screen.blit(text, (BOARD_OFFSET_X + BOARD_SIZE * CELL_SIZE + 50, BOARD_OFFSET_Y))
        if self.game_mode == GameMode.PVA:
            algorithm_names = {
                AIAlgorithm.MINIMAX: "Minimax",
                AIAlgorithm.EXPECTIMAX: "Expectimax",
                AIAlgorithm.MONTE_CARLO: "Monte Carlo"
            }
            ai_text = f"AI Algorithm: {algorithm_names[self.ai_algorithm]}"
            text = self.font_small.render(ai_text, True, BLACK)
            self.screen.blit(text, (BOARD_OFFSET_X + BOARD_SIZE * CELL_SIZE + 50, BOARD_OFFSET_Y + 35))
        if self.game_mode == GameMode.PVA and self.ai_thinking:
            thinking_text = self.font_medium.render("AI thinking...", True, RED)
            self.screen.blit(thinking_text, (BOARD_OFFSET_X + BOARD_SIZE * CELL_SIZE + 50, BOARD_OFFSET_Y + 120))
        walls_p1 = f"Player 1 Walls: {self.game_state.walls_remaining[Player.ONE]}"
        walls_p2 = f"Player 2 Walls: {self.game_state.walls_remaining[Player.TWO]}"
        text1 = self.font_small.render(walls_p1, True, BLUE)
        text2 = self.font_small.render(walls_p2, True, RED)
        y_offset = 70 if self.game_mode == GameMode.PVA else 50
        self.screen.blit(text1, (BOARD_OFFSET_X + BOARD_SIZE * CELL_SIZE + 50, BOARD_OFFSET_Y + y_offset))
        self.screen.blit(text2, (BOARD_OFFSET_X + BOARD_SIZE * CELL_SIZE + 50, BOARD_OFFSET_Y + y_offset + 30))
        controls = [
            "Controls:",
            "WASD - Move pawn",
            "J - Diagonal left",
            "K - Diagonal right",
            "Mouse - Place walls",
            "Q - Rotate wall",
            "",
            f"Wall orientation: {'Horizontal' if self.wall_orientation == WallOrientation.HORIZONTAL else 'Vertical'}"
        ]
        for i, control in enumerate(controls):
            text = self.font_small.render(control, True, BLACK)
            self.screen.blit(text, (BOARD_OFFSET_X + BOARD_SIZE * CELL_SIZE + 50, BOARD_OFFSET_Y + 170 + i * 25))

    def draw_game_over(self):
        if not self.played_winner_sound:
            sound_winner.play()
            self.played_winner_sound = True

        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        winner_text = f"{'Player 1' if self.game_state.winner == Player.ONE else 'Player 2'} Wins!"
        text = self.font_large.render(winner_text, True, WHITE)
        text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
        self.screen.blit(text, text_rect)
        options = ["R - Restart", "M - Main Menu"]
        for i, option in enumerate(options):
            text = self.font_medium.render(option, True, WHITE)
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 20 + i * 40))
            self.screen.blit(text, text_rect)

    def run(self):
        running = True
        self.game_mode = GameMode.MENU
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if self.game_mode == GameMode.MENU:
                    self.handle_menu_events(event)
                else:
                    if self.game_state is None:
                        self.game_state = GameState()
                    self.handle_game_events(event)
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
