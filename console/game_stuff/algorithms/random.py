import random


def random_move(pawns, walls):
    choice = random.randint(1, 2)
    if choice == 2 and len(walls) > 0:
        wall = random.randint(0, len(walls) - 1)
        pawn = None
    else:
        pawn = random.randint(0, len(pawns) - 1)
        wall = None
        choice = 1
    
    return choice, pawn, wall
