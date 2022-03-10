import numpy as np

from agents.agent import Agent


class FirstFiveAgent(Agent):
    NAME = 'First 5 Steps'

    def run(self, callback=lambda *args: None):
        # PRINT 5 STEPS OF AN EPISODE CONSIDERING A RANDOM AGENT
        S, X, allowed_a = self.env.initialise_game()

        print(S)
        # PRINT VARIABLE THAT TELLS IF ENEMY KING IS IN CHECK (1) OR NOT (0)
        print("check? ", self.env.check)
        # PRINT THE NUMBER OF LOCATIONS THAT THE ENEMY KING CAN MOVE TO
        print("dofk2 ", np.sum(self.env.dfk2_constrain).astype(int))
        callback(S, 0, 5, [0], [0])

        for i in range(6):
            a, _ = np.where(allowed_a == 1)  # FIND WHAT THE ALLOWED ACTIONS ARE
            a_agent = np.random.permutation(a)[0]  # MAKE A RANDOM ACTION

            S, X, allowed_a, R, Done = self.env.one_step(a_agent)  # UPDATE THE ENVIRONMENT

            # PRINT CHESS BOARD AND VARIABLES
            callback(S, i, 5, [R], [0])

            print("")
            print(S)
            print(R, "", Done)
            print("check? ", self.env.check)
            print("dofk2 ", np.sum(self.env.dfk2_constrain).astype(int))

            # TERMINATE THE EPISODE IF Done=True (DRAW OR CHECKMATE)
            if Done:
                break
