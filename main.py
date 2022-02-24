from chess_env import *

size_board = 4

if __name__ == '__main__':
    env = ChessEnv(size_board)

    # PRINT 5 STEPS OF AN EPISODE CONSIDERING A RANDOM AGENT
    S, X, allowed_a = env.initialise_game()  # INITIALISE GAME

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

    # PERFORM N_episodes=1000 EPISODES MAKING RANDOM ACTIONS AND COMPUTE THE AVERAGE REWARD AND NUMBER OF MOVES
    S, X, allowed_a = env.initialise_game()
    N_episodes = 1000

    # VARIABLES WHERE TO SAVE THE FINAL REWARD IN AN EPISODE AND THE NUMBER OF MOVES
    R_save_random = np.zeros([N_episodes, 1])
    N_moves_save_random = np.zeros([N_episodes, 1])

    for n in range(N_episodes):
        S, X, allowed_a = env.initialise_game()  # INITIALISE GAME
        Done = 0  # SET Done=0 AT THE BEGINNING
        i = 1  # COUNTER FOR THE NUMBER OF ACTIONS (MOVES) IN AN EPISODE

        # UNTIL THE EPISODE IS NOT OVER...(Done=0)
        while Done == 0:
            # SAME AS THE CELL BEFORE, BUT SAVING THE RESULTS WHEN THE EPISODE TERMINATES
            a, _ = np.where(allowed_a == 1)
            a_agent = np.random.permutation(a)[0]

            S, X, allowed_a, R, Done = env.one_step(a_agent)

            if Done:
                R_save_random[n] = np.copy(R)
                N_moves_save_random[n] = np.copy(i)

                break

            i = i + 1  # UPDATE THE COUNTER

    # AS YOU SEE, THE PERFORMANCE OF A RANDOM AGENT ARE NOT GREAT, SINCE THE MAJORITY OF THE POSITIONS END WITH A DRAW
    # (THE ENEMY KING IS NOT IN CHECK AND CAN'T MOVE)

    print('Random_Agent, Average reward:', np.mean(R_save_random), 'Number of steps: ', np.mean(N_moves_save_random))

    # INITIALISE THE PARAMETERS OF YOUR NEURAL NETWORK AND...
    # PLEASE CONSIDER TO USE A MASK OF ONE FOR THE ACTION MADE AND ZERO OTHERWISE IF YOU ARE NOT USING VANILLA GRADIENT DESCENT...
    # WE SUGGEST A NETWORK WITH ONE HIDDEN LAYER WITH SIZE 200.

    S, X, allowed_a = env.initialise_game()
    N_a = np.shape(allowed_a)[0]  # TOTAL NUMBER OF POSSIBLE ACTIONS

    N_in = np.shape(X)[0]  ## INPUT SIZE
    N_h = 200  ## NUMBER OF HIDDEN NODES

    ## INITALISE YOUR NEURAL NETWORK...
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
        epsilon_f = epsilon_0 / (1 + beta * n)  ## DECAYING EPSILON
        Done = 0  ## SET DONE TO ZERO (BEGINNING OF THE EPISODE)
        i = 1  ## COUNTER FOR NUMBER OF ACTIONS

        S, X, allowed_a = env.initialise_game()  ## INITIALISE GAME
        if n % 1000 == 0:
            print(f".({n})", end='')

        while Done == 0:  ## START THE EPISODE
            ## THIS IS A RANDOM AGENT, CHANGE IT...

            a, _ = np.where(allowed_a == 1)
            a_agent = np.random.permutation(a)[0]

            S_next, X_next, allowed_a_next, R, Done = env.one_step(a_agent)

            ## THE EPISODE HAS ENDED, UPDATE...BE CAREFUL, THIS IS THE LAST STEP OF THE EPISODE
            if Done == 1:
                break
            # IF THE EPISODE IS NOT OVER...
            else:
                ## ONLY TO PUT SUMETHING
                PIPPO = 1

            # NEXT STATE AND CO. BECOME ACTUAL STATE...
            S = np.copy(S_next)
            X = np.copy(X_next)
            allowed_a = np.copy(allowed_a_next)

            i += 1  # UPDATE COUNTER FOR NUMBER OF ACTIONS
