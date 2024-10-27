from game_state import GameState
import os
import time


def clear_screen():
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For macOS and Linux
    else:
        _ = os.system('clear')


def player_v_player():
    stateGame = GameState()
    while not stateGame.is_goal():
        stateGame.player_stats()
        stateGame.print_board()
        print(f'Player: {stateGame.turn}')
        print(f'Possible moves: {stateGame.possible_moves_pawn()}')
        print(f'Possible walls: {stateGame.possible_moves_wall()}')
        ipt = input("Pawn(1) or wall(2)?")
        if ipt != "1" or ipt != "2":
            while ipt != "1" and ipt != "2":
                print("Must pick 1 or 2!")
                ipt = input("Pawn(1) or wall(2)?")
        if ipt == "1":
            move = input("Move: ")
            while not stateGame.move_pawn(stateGame.parse_move(move)):
                move = input("Move: ")
        elif ipt == "2":
            move = input("Wall: ")
            while not stateGame.place_wall(stateGame.parse_move(move)):
                move = input("Wall: ")
        clear_screen()

    stateGame.player_stats()
    stateGame.print_board()
    print(f'{stateGame.get_winner()} is the winner!')


def random():
    from algorithms.random import random_move
    stateGame = GameState()
    walls = []
    while not stateGame.is_goal():
        stateGame.player_stats()
        stateGame.print_board()
        print(len(stateGame.possible_moves_wall()))
        print(walls)
        print(f'Player: {stateGame.turn}')
        ipt, pawn, wall = random_move(
            stateGame.possible_moves_pawn(), stateGame.possible_moves_wall())
        if ipt == 1:
            print("Move: ")
            while True:
                if not stateGame.move_pawn(stateGame.possible_moves_pawn()[pawn]):
                    print("Move: ")
                else:
                    break
        elif ipt == 2:
            print("Wall: ")
            while True:
                wall_ = stateGame.possible_moves_wall()[wall]
                if not stateGame.place_wall(wall_):
                    print("Move: ")
                else:
                    walls.append(wall_)
                    break
        time.sleep(0.5)
        clear_screen()

    stateGame.player_stats()
    stateGame.print_board()
    print(f'{stateGame.get_winner()} is the winner!')


def other(alg=3):
    if alg ==  1:
        from  algorithms.minimax import get_best_move
        print("peder")
    elif alg == 2:
        from algorithms.monte_carlo_tree_search import get_best_move
    from algorithms.expectimax import get_best_move

    stateGame = GameState()
    ans = input("Do you want to play? (y/n)")
    while ans != "y" and ans != "n":
        ans = input("Do you want to play? (y/n)")
    if ans == "y":
        human_turn = True
        ans = input("Choose player: (1/2)")
        while ans != "1" and ans != "2":
            ans = input("Choose player: (1/2)")
        if ans == "2":
            human_turn = False
        while not stateGame.is_goal():
            if human_turn:
                stateGame.player_stats()
                stateGame.print_board()
                ipt = input("Pawn(1) or wall(2)?")
                if ipt != "1" or ipt != "2":
                    while ipt != "1" and ipt != "2":
                        print("Must pick 1 or 2!")
                        ipt = input("Pawn(1) or wall(2)?")
                if ipt == "1":
                    move = input("Move: ")
                    while not stateGame.move_pawn(stateGame.parse_move(move)):
                        move = input("Move: ")
                elif ipt == "2":
                    move = input("Wall: ")
                    while not stateGame.place_wall(stateGame.parse_move(move)):
                        move = input("Wall: ")

                human_turn = False
                clear_screen()

            else:
                stateGame.player_stats()
                stateGame.print_board()
                print("\nAI is thinking...")

                move_type, move = get_best_move(stateGame, depth=2)

                if move_type == 1:
                    print(f"AI moves pawn to: {move}")
                    stateGame.move_pawn(move)
                else:
                    wall_coords = stateGame.get_wall_coords(move)
                    print(f"AI places wall at: {wall_coords}")
                    stateGame.place_wall(move)

                human_turn = True
                time.sleep(0.5)
                clear_screen()

    else:
        while not stateGame.is_goal():
            stateGame.player_stats()
            stateGame.print_board()
            print(f'Player: {stateGame.turn}')

            # Get AI move
            move_type, move = get_best_move(stateGame, depth=2)

            if move_type == 1:  # Pawn move
                print(f"AI moves pawn to: {move}")
                stateGame.move_pawn(move)
            else:  # Wall placement
                print(f"AI places wall at: {move}")
                stateGame.place_wall(move)

            time.sleep(0.5)  # To make the game visible
            clear_screen()

    stateGame.player_stats()
    stateGame.print_board()
    print(f'{stateGame.get_winner()} is the winner!')


if __name__ == '__main__':
    other(1)
