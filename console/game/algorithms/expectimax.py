from heuristics import evaluate_position
from game_state import GameState


def expectimax(game_state: GameState, depth, maximizing_player):
    if depth == 0:
        return evaluate_position(game_state, maximizing_player, True)
    if maximizing_player:
        return max([expectimax(child[1], False, depth - 1, maximizing_player) for child in
                    game_state.all_possible_moves(maximizing_player, True)])
    else:
        values = [expectimax(child[1], True, depth - 1, maximizing_player) for child in
                  game_state.all_possible_moves(maximizing_player, True)]
        return sum(values) / len(values)
