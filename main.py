from chess_env import *
from chessy_agent import ChessyAgent

size_board = 4


def print_first_five(env):
    # PRINT 5 STEPS OF AN EPISODE CONSIDERING A RANDOM AGENT
    S, X, allowed_a = env.initialise_game()

    print(S)  # PRINT CHESS BOARD (SEE THE DESCRIPTION ABOVE)

    # PRINT VARIABLE THAT TELLS IF ENEMY KING IS IN CHECK (1) OR NOT (0)
    print("check? ", env.check)
    print("dofk2 ",
          np.sum(env.dfk2_constrain).astype(int)
          )  # PRINT THE NUMBER OF LOCATIONS THAT THE ENEMY KING CAN MOVE TO

    for i in range(5):
        a, _ = np.where(allowed_a == 1)  # FIND WHAT THE ALLOWED ACTIONS ARE
        a_agent = np.random.permutation(a)[0]  # MAKE A RANDOM ACTION

        S, X, allowed_a, R, Done = env.one_step(
            a_agent)  # UPDATE THE ENVIRONMENT

        # PRINT CHESS BOARD AND VARIABLES
        print("")
        print(S)
        print(R, "", Done)
        print("check? ", env.check)
        print("dofk2 ", np.sum(env.dfk2_constrain).astype(int))

        # TERMINATE THE EPISODE IF Done=True (DRAW OR CHECKMATE)
        if Done:
            break


def perform_random_agent(env, N_episodes=1000):
    """
    Perform N Episodes Making Random Actions
    And Compute The Average Reward And Number Of Moves
    """

    R_save_random = np.zeros([N_episodes, 1])
    N_moves_save_random = np.zeros([N_episodes, 1])
    checkmate_save = np.zeros(N_episodes)

    for n in range(N_episodes):
        board_state, X, allowed_actions = env.initialise_game()
        done = 0
        move_counter = 1

        while done == 0:
            a, _ = np.where(allowed_actions == 1)
            current_action = np.random.permutation(a)[0]

            board_state, X, allowed_actions, R, done = env.one_step(
                current_action)

            if done:
                R_save_random[n] = np.copy(R)
                N_moves_save_random[n] = np.copy(move_counter)
                break

            move_counter += 1

            # AS YOU SEE, THE PERFORMANCE OF A RANDOM AGENT ARE NOT GREAT,
    # SINCE THE MAJORITY OF THE POSITIONS END WITH A DRAW
    # (THE ENEMY KING IS NOT IN CHECK AND CAN'T MOVE)

    print(
        "Random Agent, Average reward:",
        np.mean(R_save_random),
        "Number of steps: ",
        np.mean(N_moves_save_random),
        "Number of checkmates: ",
        np.count_nonzero(checkmate_save == 1),
    )


if __name__ == '__main__':
    board_size = 4

    print(f"===== First 5 Steps =====")
    print_first_five(ChessEnv(board_size))

    print(f"===== Random Agent =====")
    perform_random_agent(ChessEnv(board_size))

    print(f"===== Neural Network =====")
    ChessyAgent(board_size).run(lambda *args: None)
