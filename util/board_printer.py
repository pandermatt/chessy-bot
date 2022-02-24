import numpy as np


def fancy_print_board(board):
    arr = np.array(board).flatten()
    for index, item in enumerate(arr):
        icon = '⬜'
        if item == 1:
            icon = '♚'
        elif item == 2:
            icon = '♛'
        elif item == 3:
            icon = '♔'
        print(icon, end='')
        if index % board.shape[0] == 3:
            print()
