from chess_env import *

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
    N_h = 200  # NUMBER OF HIDDEN NODES

    # INITALISE YOUR NEURAL NETWORK...
    # HYPERPARAMETERS SUGGESTED (FOR A GRID SIZE OF 4)

    epsilon_0 = 0.2  # STARTING VALUE OF EPSILON FOR THE EPSILON-GREEDY POLICY
    beta = 0.00005  # THE PARAMETER SETS HOW QUICKLY THE VALUE OF EPSILON IS DECAYING (SEE epsilon_f BELOW)
    gamma = 0.85  # THE DISCOUNT FACTOR
    eta = 0.0035  # THE LEARNING RATE

    N_episodes = 100000  # THE NUMBER OF GAMES TO BE PLAYED

    # SAVING VARIABLES
    R_save = np.zeros([N_episodes, 1])
    N_moves_save = np.zeros([N_episodes, 1])

    # TRAINING LOOP BONE STRUCTURE...
    # I WROTE FOR YOU A RANDOM AGENT (THE RANDOM AGENT WILL BE SLOWER TO GIVE CHECKMATE THAN AN OPTIMISED ONE,
    # SO DON'T GET CONCERNED BY THE TIME IT TAKES), CHANGE WITH YOURS ...

    for n in range(N_episodes):
        epsilon_f = epsilon_0 / (1 + beta * n)  # DECAYING EPSILON
        done = 0  # SET DONE TO ZERO (BEGINNING OF THE EPISODE)
        move_counter = 1  # COUNTER FOR NUMBER OF ACTIONS

        board_state, X, allowed_actions = env.initialise_game()
        if n % 1000 == 0:
            print(f"Epoche ({n}/{N_episodes})")

        while done == 0:  # START THE EPISODE
            # THIS IS A RANDOM AGENT, CHANGE IT...

            a, _ = np.where(allowed_actions == 1)
            current_action = np.random.permutation(a)[0]

            S_next, X_next, allowed_a_next, R, done = env.one_step(current_action)

            # THE EPISODE HAS ENDED, UPDATE...BE CAREFUL, THIS IS THE LAST STEP OF THE EPISODE
            if done == 1:
                R_save[n] = np.copy(R)
                N_moves_save[n] = np.copy(move_counter)
                break
            else:
                # IF THE EPISODE IS NOT OVER...
                pass

            # NEXT STATE AND CO. BECOME ACTUAL STATE...
            board_state = np.copy(S_next)
            X = np.copy(X_next)
            allowed_actions = np.copy(allowed_a_next)

            move_counter += 1  # UPDATE COUNTER FOR NUMBER OF ACTIONS

    print('Chessy Agent, Average reward:', np.mean(R_save), 'Number of steps: ', np.mean(N_moves_save))



if __name__ == '__main__':
    chess_env = ChessEnv(size_board)

    print(f"===== First 5 Steps =====")
    print_first_five(chess_env)

    print(f"===== Random Agent =====")
    perform_random_agent(chess_env)

    print(f"===== Neural Network =====")
    perform_nerual_network(chess_env)
