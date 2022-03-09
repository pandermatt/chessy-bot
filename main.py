from chess_env import *
from Neural_net import Neural_net

size_board = 4


def print_first_five(env):
    # PRINT 5 STEPS OF AN EPISODE CONSIDERING A RANDOM AGENT
    S, X, allowed_a = env.initialise_game()

    print(S)  # PRINT CHESS BOARD (SEE THE DESCRIPTION ABOVE)

    print('check? ', env.check)  # PRINT VARIABLE THAT TELLS IF ENEMY KING IS IN CHECK (1) OR NOT (0)
    print('dofk2 ',
          np.sum(env.dfk2_constrain).astype(int))  # PRINT THE NUMBER OF LOCATIONS THAT THE ENEMY KING CAN MOVE TO

    for i in range(5):
        a, _ = np.where(allowed_a == 1)  # FIND WHAT THE ALLOWED ACTIONS ARE
        a_agent = np.random.permutation(a)[0]  # MAKE A RANDOM ACTION

        S, X, allowed_a, R, Done = env.one_step(a_agent)  # UPDATE THE ENVIRONMENT

        # PRINT CHESS BOARD AND VARIABLES
        print('')
        print(S)
        print(R, '', Done)
        print('check? ', env.check)
        print('dofk2 ', np.sum(env.dfk2_constrain).astype(int))

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

    for n in range(N_episodes):
        board_state, X, allowed_actions = env.initialise_game()
        done = 0
        move_counter = 1

        while done == 0:
            a, _ = np.where(allowed_actions == 1)
            current_action = np.random.permutation(a)[0]

            board_state, X, allowed_actions, R, done = env.one_step(current_action)

            if done:
                R_save_random[n] = np.copy(R)
                N_moves_save_random[n] = np.copy(move_counter)
                break

            move_counter += 1

            # AS YOU SEE, THE PERFORMANCE OF A RANDOM AGENT ARE NOT GREAT,
    # SINCE THE MAJORITY OF THE POSITIONS END WITH A DRAW
    # (THE ENEMY KING IS NOT IN CHECK AND CAN'T MOVE)

    print('Random Agent, Average reward:', np.mean(R_save_random), 'Number of steps: ', np.mean(N_moves_save_random))


def perform_nerual_network(env):
    # INITIALISE THE PARAMETERS OF YOUR NEURAL NETWORK AND...
    # PLEASE CONSIDER USING A MASK OF ONE FOR THE ACTION MADE
    # AND ZERO OTHERWISE IF YOU ARE NOT USING VANILLA GRADIENT DESCENT...
    # WE SUGGEST A NETWORK WITH ONE HIDDEN LAYER WITH SIZE 200.

    board_state, X, allowed_actions = env.initialise_game()
    N_a = np.shape(allowed_actions)[0]  # TOTAL NUMBER OF POSSIBLE ACTIONS
    N_in = np.shape(X)[0]  # INPUT SIZE
    N_h1 = 200  # NUMBER OF HIDDEN NODES

    # INITALISE YOUR NEURAL NETWORK...
    # HYPERPARAMETERS SUGGESTED (FOR A GRID SIZE OF 4)
    nn = Neural_net(env, [N_in, N_h1, N_a], xavier=True)

    N_episodes = 300000  # THE NUMBER OF GAMES TO BE PLAYED

    nn.train(N_episodes)


if __name__ == '__main__':
    chess_env = ChessEnv(size_board)

    print(f"===== First 5 Steps =====")
    print_first_five(chess_env)

    print(f"===== Random Agent =====")
    perform_random_agent(chess_env)

    print(f"===== Neural Network =====")
    perform_nerual_network(chess_env)
