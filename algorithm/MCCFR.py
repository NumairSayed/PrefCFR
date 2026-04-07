from open_spiel.python.algorithms.external_sampling_mccfr import ExternalSamplingSolver
import numpy as np


class ES_MCCFR(ExternalSamplingSolver):
    def __init__(self, game):
        super().__init__(game)
        self.node_touched = 0

    def _update_regrets(self, state, player):
        self.node_touched += 1
        return super()._update_regrets(state, player)

    def show_policy(self, info_keys):
        for i_info_key in info_keys:
            print(self._infostates[i_info_key][1] / np.sum(self._infostates[i_info_key][1]))

    def get_policy(self, info_key):
        tmp_policy = self._infostates[info_key][1] / np.sum(self._infostates[info_key][1])
        return tmp_policy[0]
