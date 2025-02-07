import math
from game_state import GameState
from heuristics.evaluate_position import evaluate_position


def minimax(game_state: GameState, depth, alpha, beta, maximizing_player, turn):
    if depth == 0:
        return evaluate_position(game_state, turn)
    if maximizing_player:
        max_eval = -math.inf
        for child in game_state.all_possible_moves(turn):
            ev = minimax(
                child[0], depth - 1, alpha, beta, False, turn)
            max_eval = max(max_eval, ev)
            alpha = max(alpha, ev)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for child in game_state.all_possible_moves(turn):
            ev = minimax(
                child[0], depth - 1, alpha, beta, True, turn)
            min_eval = min(min_eval, ev)
            beta = min(beta, ev)
            if beta <= alpha:
                break
        return min_eval
