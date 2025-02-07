from heuristics import evaluate_position
from game_state import GameState


def expectimax(game_state: GameState, depth, maximizing_player, turn):
    if depth == 0:
        return evaluate_position.evaluate_position(game_state, turn, True)
    if maximizing_player:
        return max([expectimax(child[0], False, depth - 1, turn) for child in
                    game_state.all_possible_moves(turn)])
    else:
        values = [expectimax(child[0], True, depth - 1, turn) for child in
                  game_state.all_possible_moves(turn)]
        return sum(values) / len(values)