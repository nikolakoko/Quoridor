from searching import astar
from game_state import GameState


def evaluate_position(game_state: GameState, is_maximizing):
    """
    Evaluates the game state from player one's perspective.
    Higher scores are better for player one, lower scores favor player two.
    """

    p1_distance = game_state.player_one_pos[0] // 2
    p2_distance = (16 - game_state.player_two_pos[0]) // 2
    result = 0
    
    if game_state.player_one_walls != 10 and game_state.player_two_walls != 10:
        _, p1_distance = astar.path_exists(
            game_state, game_state.player_one_pos, True)
        _, p2_distance = astar.path_exists(
            game_state, game_state.player_two_pos, True)

    if is_maximizing:
        result += p2_distance
        result -= p1_distance
        
        num = 100
        if p1_distance != 0:
            num = p1_distance
        result += round(100 / num, 2)

        num = 50
        if p2_distance != 0:
            num = p2_distance
        result -= round(50 / num, 2)

        result += game_state.player_one_walls - game_state.player_two_walls
        if game_state.player_one_pos[0] == 0:
            result += 100
        if p1_distance == 0 and game_state.player_one_pos[0] != 0:
            result -= 500
        return result

    else:
        result += p1_distance * 17
        result -= p2_distance
        
        num = 100
        if p2_distance != 0:
            num = p2_distance
        result += round(100 / num, 2)

        num = 50
        if p1_distance != 0:
            num = p1_distance
        result -= round(50 / num, 2)

        result += game_state.player_two_walls - game_state.player_one_walls
        if game_state.player_two_pos[0] == 16:
            result += 100
        if p2_distance == 0 and game_state.player_two_pos[0] != 16:
            result -= 500
        return result


# def minimax_alpha_beta_pruning(game_state: GameState, depth, alpha, beta, maximizing_player, player_one_minimax):
#     if depth == 0:
#         # Return the evaluation score and a None move (no move at leaf)
#         return evaluate_position(game_state, player_one_minimax), None
    
#     if maximizing_player:
#         max_eval = -float('-inf')
#         best_move = None  # To track the best move
#         for child in game_state.get_all_child_states(player_one_minimax):
#             ev, _ = minimax_alpha_beta_pruning(child[0], depth - 1, alpha, beta, False, player_one_minimax)
#             if ev > max_eval:
#                 max_eval = ev
#                 best_move = child[1]  # Assume child[1] is the move that led to child[0]
#             alpha = max(alpha, ev)
#             if beta <= alpha:
#                 break
#         return max_eval, best_move  # Return both the score and the best move
    
#     else:
#         min_eval = float('-inf')
#         best_move = None  # To track the best move
#         for child in game_state.get_all_child_states(player_one_minimax):
#             ev, _ = minimax_alpha_beta_pruning(child[0], depth - 1, alpha, beta, True, player_one_minimax)
#             if ev < min_eval:
#                 min_eval = ev
#                 best_move = child[1]  # Assume child[1] is the move that led to child[0]
#             beta = min(beta, ev)
#             if beta <= alpha:
#                 break
#         return min_eval, best_move  # Return both the score and the best move


def minimax(game_state, depth, is_maximizing):
    stack = [(game_state, depth, float('-inf'), float('inf'), is_maximizing, None)]
    best_score = float('-inf')
    best_move = None

    while stack:
        current_state, current_depth, alpha, beta, maximizing_player, move_to_undo = stack.pop()

        if move_to_undo is not None:
            current_state.undo_last_move(move_to_undo)

        if current_depth == 0 or current_state.is_goal():
            score = evaluate_position(current_state, maximizing_player)
            if maximizing_player:
                if score > best_score:
                    best_score = score
                    best_move = move_to_undo
            else:
                if score < best_score:
                    best_score = score
                    best_move = move_to_undo
            continue

        move_found = False
        possible_moves = current_state.possible_moves_pawn() + current_state.possible_moves_wall()

        for move in possible_moves:
            if current_state.move_pawn(move) or current_state.place_wall(move):
                move_found = True
                stack.append((current_state, current_depth - 1, alpha, beta, not maximizing_player, move))
                if maximizing_player:
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        break
                else:
                    beta = min(beta, best_score)
                    if beta <= alpha:
                        break

        if not move_found:
            best_score = -1000 if maximizing_player else 1000

    return best_score, best_move


def get_best_move(game_state, depth=2):
    """
    Get the best move for the current player.
    Returns (move_type, move) where move_type is 1 for pawn or 2 for wall.
    """
    is_maximizing = game_state.turn  # True for player one, False for player two
    _, best_move = minimax(game_state, depth, is_maximizing)
    return best_move
