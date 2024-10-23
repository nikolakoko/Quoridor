def heuristic(state):
    if state.turn:
        return state.player_one_pos[0] * 100
    return state.player_two_pos[0] * 100 - 16

def astar(state, goal_state):
    visited = set()
    
    