from algorithm.MCCFR import ES_MCCFR
import numpy as np
from open_spiel.python.algorithms import mccfr
from open_spiel.python.algorithms.external_sampling_mccfr import AverageType


class ES_MCPrefCFR(ES_MCCFR):
    """
    The default version is PrefCFR(BR).
    """
    def __init__(self, game, pref_config=None):
        super().__init__(game)
        if pref_config is None:
            pref_config = {}
        self.info_touched_times = {}
        self.pref_config = pref_config

    def _regret_matching(self, regrets, num_legal_actions, info_key=None):
        # print(info_key)
        tmp_p = np.zeros(num_legal_actions, dtype=np.float64)

        if info_key is not None and info_key in self.pref_config.keys():
            delta = self.pref_config[info_key][0]
            beta = self.pref_config[info_key][1]

            regrets = regrets - beta
            regrets[regrets < 0] = 0

            regrets *= delta

            if np.sum(regrets) == 0:
                tmp_p[np.argmax(delta)] = 1.0
            else:
                tmp_p[np.argmax(regrets)] = 1.0

            return tmp_p
        else:
            regrets[regrets < 0] = 0
            if np.sum(regrets) == 0:
                tmp_p[np.random.randint(num_legal_actions)] = 1.0
            else:
                argmax_regrets = np.argmax(regrets)
                tmp_p[argmax_regrets] = 1.0

            return tmp_p

    def _full_update_average(self, state, reach_probs):
        """Performs a full update average.

        Args:
          state: the open spiel state to run from
          reach_probs: array containing the probability of reaching the state
            from the players point of view
        """
        self.node_touched += 1

        if state.is_terminal():
            return
        if state.is_chance_node():
            for action in state.legal_actions():
                self._full_update_average(state.child(action), reach_probs)
            return

        # If all the probs are zero, no need to keep going.
        sum_reach_probs = np.sum(reach_probs)
        if sum_reach_probs == 0:
            return

        cur_player = state.current_player()
        info_state_key = state.information_state_string(cur_player)
        legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)

        infostate_info = self._lookup_infostate_info(info_state_key,
                                                     num_legal_actions)

        # It's different from the previous CFR algorithm
        if info_state_key not in self.info_touched_times.keys():
            policy = self._regret_matching(infostate_info[mccfr.REGRET_INDEX],
                                           num_legal_actions, info_state_key)
        else:
            policy = self._regret_matching(infostate_info[mccfr.REGRET_INDEX] / self.info_touched_times[info_state_key],
                                           num_legal_actions, info_state_key)

        for action_idx in range(num_legal_actions):
            new_reach_probs = np.copy(reach_probs)
            new_reach_probs[cur_player] *= policy[action_idx]
            self._full_update_average(
                state.child(legal_actions[action_idx]), new_reach_probs)

        # Now update the cumulative policy
        for action_idx in range(num_legal_actions):
            self._add_avstrat(info_state_key, action_idx,
                              reach_probs[cur_player] * policy[action_idx])

        # It's different from the previous CFR algorithm
        if info_state_key not in self.info_touched_times.keys():
            self.info_touched_times[info_state_key] = 1
        else:
            self.info_touched_times[info_state_key] += 1

    def _update_regrets(self, state, player):
        """Runs an episode of external sampling.

        Args:
          state: the open spiel state to run from
          player: the player to update regrets for

        Returns:
          value: is the value of the state in the game
          obtained as the weighted average of the values
          of the children
        """
        self.node_touched += 1

        if state.is_terminal():
            return state.player_return(player)

        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            outcome = np.random.choice(outcomes, p=probs)
            return self._update_regrets(state.child(outcome), player)

        cur_player = state.current_player()
        info_state_key = state.information_state_string(cur_player)
        legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)

        infostate_info = self._lookup_infostate_info(info_state_key,
                                                     num_legal_actions)

        # It's different from the previous CFR algorithm
        if info_state_key not in self.info_touched_times.keys():
            policy = self._regret_matching(infostate_info[mccfr.REGRET_INDEX],
                                           num_legal_actions, info_state_key)
        else:
            policy = self._regret_matching(infostate_info[mccfr.REGRET_INDEX] / self.info_touched_times[info_state_key],
                                           num_legal_actions, info_state_key)

        value = 0
        child_values = np.zeros(num_legal_actions, dtype=np.float64)
        if cur_player != player:
            # Sample at opponent node
            action_idx = np.random.choice(np.arange(num_legal_actions), p=policy)
            value = self._update_regrets(
                state.child(legal_actions[action_idx]), player)
        else:
            # Walk over all actions at my node
            for action_idx in range(num_legal_actions):
                child_values[action_idx] = self._update_regrets(
                    state.child(legal_actions[action_idx]), player)
                value += policy[action_idx] * child_values[action_idx]

        if cur_player == player:
            # Update regrets.
            for action_idx in range(num_legal_actions):
                self._add_regret(info_state_key, action_idx,
                                 child_values[action_idx] - value)
        # Simple average does averaging on the opponent node. To do this in a game
        # with more than two players, we only update the player + 1 mod num_players,
        # which reduces to the standard rule in 2 players.
        if self._average_type == AverageType.SIMPLE and cur_player == (
                player + 1) % self._num_players:
            for action_idx in range(num_legal_actions):
                self._add_avstrat(info_state_key, action_idx, policy[action_idx])

            # It's different from the previous CFR algorithm
            if info_state_key not in self.info_touched_times.keys():
                self.info_touched_times[info_state_key] = 1
            else:
                self.info_touched_times[info_state_key] += 1

        return value
