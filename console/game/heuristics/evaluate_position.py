from game_state import GameState
from searching import astar


# def evaluate_position(game_state: GameState, is_maximizing):
#     """
#     Evaluates the game state from the perspective of the current player (maximizing or minimizing).
#     A positive score benefits the maximizing player, and a negative score favors the opponent.
#     """
#     _, p1_distance = astar.path_exists(game_state, game_state.player_one_pos, True)
#     _, p2_distance = astar.path_exists(game_state, game_state.player_two_pos, False)

#     p1_walls = game_state.player_one_walls
#     p2_walls = game_state.player_two_walls

#     score = 0
#     score += (p2_distance - p1_distance)
#     score += (p1_walls - p2_walls) * 10

#     if game_state.player_one_pos[0] == 0:
#         score += 100
#     if game_state.player_two_pos[0] == 16:
#         score -= 100

#     if p1_distance == 0 and game_state.player_one_pos[0] != 0:
#         score -= 500
#     if p2_distance == 0 and game_state.player_two_pos[0] != 16:
#         score += 500

#     return score if is_maximizing else -score

def evaluate_position(game_state: GameState, is_maximizing, is_expectimax=False):
    player_one_distance = game_state.player_one_pos[0] // 2
    player_two_distance = (16 - game_state.player_two_pos[0]) // 2
    result = 0

    if is_maximizing:
        opponent_path_len, player_path_len = player_two_distance, player_one_distance
        if game_state.player_one_walls != 10 and game_state.player_two_walls != 10:
            previous = game_state.turn
            game_state.turn = True
            player_path_len = astar.path_exists(
                game_state, game_state.player_one_pos, True)[1]
            game_state.turn = previous

        result += opponent_path_len
        result -= player_one_distance
        num = 100
        if player_path_len != 0:
            num = player_path_len
        result += round(100 / num, 2)

        num_1 = 50
        if player_two_distance != 0:
            num_1 = player_two_distance
        result -= round(50 / num_1, 2)

        result += (game_state.player_one_walls -
                   game_state.player_two_walls)  # mozda ovo promijeni
        if game_state.player_one_pos[0] == 0:
            result += 100
        if player_path_len == 0 and game_state.player_one_pos[0] != 0:
            result -= 500
        return result

    else:
        opponent_path_len, player_path_len = player_one_distance, player_two_distance
        if game_state.player_one_walls != 10 and game_state.player_two_walls != 10:
            previous = game_state.turn
            game_state.turn = False
            player_path_len = astar.path_exists(
                game_state, game_state.player_two_pos, False)[1]
            game_state.turn = previous

        if not is_expectimax:
            result += opponent_path_len
        else:
            result += 17 * opponent_path_len
        result -= player_two_distance
        num = 100
        if player_path_len != 0:
            num = player_path_len
        result += round(100 / num, 2)

        num_1 = 50
        if player_one_distance != 0:
            num_1 = player_one_distance
        result -= round(50 / num_1, 2)

        result += (game_state.player_two_walls -
                   game_state.player_one_walls)
        if game_state.player_two_pos[0] == 16:
            result += 100
        if player_path_len == 0 and game_state.player_two_pos[0] != 16:
            result -= 500
        return result
