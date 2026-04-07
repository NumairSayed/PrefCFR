from algorithm.MCCFR import ES_MCCFR
import numpy as np

class ES_MCCFVFP(ES_MCCFR):
    def __init__(self, game):
        super().__init__(game)

    def _regret_matching(self, regrets, num_legal_actions):
        tmp_p = np.zeros(num_legal_actions, dtype=np.float64)
        argmax_regrets = np.argmax(regrets)
        tmp_p[argmax_regrets] = 1.0

        return tmp_p
