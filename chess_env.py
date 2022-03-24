from degree_freedom.degree_freedom_king2 import *
from generate_game import *


class ChessEnv:

    def __init__(self, n_grid):

        self.reward_step = 0
        self.reward_draw = 0
        self.reward_checkmate = 1

        self.N_grid = n_grid  # SIZE OF THE BOARD

        self.Board = np.zeros(
            [n_grid, n_grid]
        )  # THE BOARD, THIS WILL BE FILLED BY 0 (NO PIECE), 1 (AGENT'S KING), 2 (AGENT'S QUEEN), 3 (OPPONENT'S KING)

        # POSITION OF THE AGENT'S KING AS COORDINATES
        self.p_k1 = np.zeros([2, 1])
        # POSITION OF THE OPPOENT'S KING AS COORDINATES
        self.p_k2 = np.zeros([2, 1])
        # POSITION OF THE AGENT'S QUEEN AS COORDINATES
        self.p_q1 = np.zeros([2, 1])

        self.dfk1 = np.zeros(
            [n_grid, n_grid]
        )  # ALL POSSIBLE ACTIONS FOR THE AGENT'S KING (LOCATIONS WHERE IT CAN MOVE WITHOUT THE PRESENCE OF THE OTHER PIECES)
        self.dfk2 = np.zeros(
            [n_grid, n_grid]
        )  # ALL POSSIBLE ACTIONS FOR THE OPPONENT'S KING (LOCATIONS WHERE IT CAN MOVE WITHOUT THE PRESENCE OF THE OTHER PIECES)
        self.dfq1 = np.zeros(
            [n_grid, n_grid]
        )  # ALL POSSIBLE ACTIONS FOR THE AGENT'S QUEEN (LOCATIONS WHERE IT CAN MOVE WITHOUT THE PRESENCE OF THE OTHER PIECES)

        self.dfk1_constrain = np.zeros(
            [n_grid, n_grid]
        )  # ALLOWED ACTIONS FOR THE AGENT'S KING CONSIDERING ALSO THE OTHER PIECES
        self.dfk2_constrain = np.zeros(
            [n_grid, n_grid]
        )  # ALLOWED ACTIONS FOT THE OPPONENT'S KING CONSIDERING ALSO THE OTHER PIECES
        self.dfq1_constrain = np.zeros(
            [n_grid, n_grid]
        )  # ALLOWED ACTIONS FOT THE AGENT'S QUEEN CONSIDERING ALSO THE OTHER PIECES

        # ALLOWED ACTIONS OF THE AGENT'S KING (CONSIDERING OTHER PIECES), ONE-HOT ENCODED
        self.ak1 = np.zeros([8])
        # TOTAL NUMBER OF POSSIBLE ACTIONS FOR AGENT'S KING
        self.possible_king_a = np.shape(self.ak1)[0]

        self.aq1 = np.zeros(
            [8 * (self.N_grid - 1)]
        )  # ALLOWED ACTIONS OF THE AGENT'S QUEEN (CONSIDERING OTHER PIECES), ONE-HOT ENCODED
        # TOTAL NUMBER OF POSSIBLE ACTIONS FOR AGENT'S QUEEN
        self.possible_queen_a = np.shape(self.aq1)[0]

        self.check = 0  # 1 (0) IF ENEMY KING (NOT) IN CHECK

        # THIS MAP IS USEFUL FOR US TO UNDERSTAND THE DIRECTION OF MOVEMENT GIVEN THE ACTION MADE (SKIP...)
        self.map = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1],
                             [-1, 1], [-1, -1]])

    def initialise_game(self):
        """
        Initialise_game. The method initialises an episode by placing the three pieces considered (Agent's king and
        queen, enemy's king) in the chess board. The outputs of the method are described below in order.

        :return: Board
            A matrix representing the board locations filled with 4 numbers:
                0: no piece in that position
                1: location of the agent's king
                2: location of the queen
                3: location of the enemy king.

        :return: X
            features: input to the neural network

        :return: allowed_a
            The allowed actions that the agent can make. The agent is moving a king, with a total number of 8
            possible actions, and a queen, with a total number of (board_size - 1) * 8 actions.
            The total number of possible actions correspond to the sum of the two, but not all actions are allowed in a
            given position (movements to locations outside the borders or against chess rules).
            Thus, the variable allowed_a is a vector that is one (zero) for an action that the agent can (can't) make.
            Be careful, apply the policy considered on the actions that are allowed only.
        """

        # START THE GAME BY SETTING PIECIES

        self.Board, self.p_k2, self.p_k1, self.p_q1 = generate_game(
            self.N_grid)

        # Allowed actions for the agent's king
        self.dfk1_constrain, self.a_k1, self.dfk1 = degree_freedom_king1(
            self.p_k1, self.p_k2, self.p_q1, self.Board)

        # Allowed actions for the agent's queen
        self.dfq1_constrain, self.a_q1, self.dfq1 = degree_freedom_queen(
            self.p_k1, self.p_k2, self.p_q1, self.Board)

        # Allowed actions for the enemy's king
        self.dfk2_constrain, self.a_k2, self.check = degree_freedom_king2(
            self.dfk1, self.p_k2, self.dfq1, self.Board, self.p_k1)

        # ALLOWED ACTIONS FOR THE AGENT, ONE-HOT ENCODED
        allowed_a = np.concatenate([self.a_q1, self.a_k1], 0)

        # FEATURES (INPUT TO NN) AT THIS POSITION
        X = self.features()

        return self.Board, X, allowed_a

    def one_step(self, agent_action):
        """
        OneStep. The method performs a one step update of the system. Given as input the action selected by the agent,
        it updates the chess board by performing that action and the response of the enemy king (which is a random allowed
        action in the settings considered). The first three outputs are the same as for the Initialise_game method,
        but the variables are computed for the position reached after the update of the system. The fourth and fifth
        outputs are:

        :return: R
            The reward

        :return: Done
            The variable that is 1 if the episode has ended (checkmate or draw)
        """

        # SET REWARD TO ZERO IF GAME IS NOT ENDED
        R = self.reward_step
        # SET Done TO ZERO (GAME NOT ENDED)
        Done = 0

        # PERFORM THE AGENT'S ACTION ON THE CHESS BOARD

        if agent_action < self.possible_queen_a:  # THE AGENT MOVED ITS QUEEN

            # UPDATE QUEEN'S POSITION
            direction = int(np.ceil(
                (agent_action + 1) / (self.N_grid - 1))) - 1
            steps = agent_action - direction * (self.N_grid - 1) + 1

            self.Board[self.p_q1[0], self.p_q1[1]] = 0

            mov = self.map[direction, :] * steps
            self.Board[self.p_q1[0] + mov[0], self.p_q1[1] + mov[1]] = 2
            self.p_q1[0] = self.p_q1[0] + mov[0]
            self.p_q1[1] = self.p_q1[1] + mov[1]

        else:  # THE AGENT MOVED ITS KING

            # UPDATE KING'S POSITION
            direction = agent_action - self.possible_queen_a
            steps = 1

            self.Board[self.p_k1[0], self.p_k1[1]] = 0
            mov = self.map[direction, :] * steps
            self.Board[self.p_k1[0] + mov[0], self.p_k1[1] + mov[1]] = 1
            self.p_k1[0] = self.p_k1[0] + mov[0]
            self.p_k1[1] = self.p_k1[1] + mov[1]

        # COMPUTE THE ALLOWED ACTIONS AFTER AGENT'S ACTION
        # Allowed actions for the agent's king
        self.dfk1_constrain, self.a_k1, self.dfk1 = degree_freedom_king1(
            self.p_k1, self.p_k2, self.p_q1, self.Board)

        # Allowed actions for the agent's queen
        self.dfq1_constrain, self.a_q1, self.dfq1 = degree_freedom_queen(
            self.p_k1, self.p_k2, self.p_q1, self.Board)

        # Allowed actions for the enemy's king
        self.dfk2_constrain, self.a_k2, self.check = degree_freedom_king2(
            self.dfk1, self.p_k2, self.dfq1, self.Board, self.p_k1)

        # CHECK IF POSITION IS A CHECMATE, DRAW, OR THE GAME CONTINUES

        # CASE OF CHECKMATE
        if (np.sum(self.dfk2_constrain) == 0
                and self.dfq1[self.p_k2[0], self.p_k2[1]] == 1):

            # King 2 has no freedom and it is checked
            # Checkmate and collect reward
            Done = 1  # The epsiode ends
            R = self.reward_checkmate  # Reward for checkmate
            allowed_a = []  # Allowed_a set to nothing (end of the episode)
            X = []  # Features set to nothing (end of the episode)

        # CASE OF DRAW
        elif (np.sum(self.dfk2_constrain) == 0
              and self.dfq1[self.p_k2[0], self.p_k2[1]] == 0):

            # King 2 has no freedom but it is not checked
            Done = 1  # The epsiode ends
            R = self.reward_draw  # Reward for draw
            allowed_a = []  # Allowed_a set to nothing (end of the episode)
            X = []  # Features set to nothing (end of the episode)

        # THE GAME CONTINUES
        else:
            # THE OPPONENT MOVES THE KING IN A RANDOM SAFE LOCATION
            allowed_enemy_a = np.where(self.a_k2 > 0)[0]
            a_help = int(
                np.ceil(np.random.rand() * allowed_enemy_a.shape[0]) - 1)
            a_enemy = allowed_enemy_a[a_help]

            direction = a_enemy
            steps = 1

            self.Board[self.p_k2[0], self.p_k2[1]] = 0
            mov = self.map[direction, :] * steps
            self.Board[self.p_k2[0] + mov[0], self.p_k2[1] + mov[1]] = 3

            self.p_k2[0] = self.p_k2[0] + mov[0]
            self.p_k2[1] = self.p_k2[1] + mov[1]

            # COMPUTE THE ALLOWED ACTIONS AFTER THE OPPONENT'S ACTION
            # Possible actions of the King
            self.dfk1_constrain, self.a_k1, self.dfk1 = degree_freedom_king1(
                self.p_k1, self.p_k2, self.p_q1, self.Board)

            # Allowed actions for the agent's king
            self.dfq1_constrain, self.a_q1, self.dfq1 = degree_freedom_queen(
                self.p_k1, self.p_k2, self.p_q1, self.Board)

            # Allowed actions for the enemy's king
            self.dfk2_constrain, self.a_k2, self.check = degree_freedom_king2(
                self.dfk1, self.p_k2, self.dfq1, self.Board, self.p_k1)

            # ALLOWED ACTIONS FOR THE AGENT, ONE-HOT ENCODED
            allowed_a = np.concatenate([self.a_q1, self.a_k1], 0)
            # FEATURES
            X = self.features()

        return self.Board, X, allowed_a, R, Done

    def features(self):
        """
        Features. Given the chessboard position, the method computes the features.
        """
        s_k1 = (np.array(self.Board == 1).astype(float).reshape(-1)
                )  # FEATURES FOR KING POSITION
        s_q1 = (np.array(self.Board == 2).astype(float).reshape(-1)
                )  # FEATURES FOR QUEEN POSITION
        s_k2 = (np.array(self.Board == 3).astype(float).reshape(-1)
                )  # FEATURE FOR ENEMY'S KING POSITION

        check = np.zeros([2])  # CHECK? FEATURE
        check[self.check] = 1

        # NUMBER OF ALLOWED ACTIONS FOR ENEMY'S KING, ONE-HOT ENCODED
        K2dof = np.zeros([8])
        K2dof[np.sum(self.dfk2_constrain).astype(int)] = 1

        # ALL FEATURES...
        x = np.concatenate([s_k1, s_q1, s_k2, check, K2dof], 0)

        return x
