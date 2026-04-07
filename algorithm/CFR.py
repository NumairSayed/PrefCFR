from open_spiel.python.algorithms.cfr import _CFRSolver
import numpy as np
import attr
import collections


def random_dict_factory():
    return collections.defaultdict(lambda: np.random.uniform(0, 1))


@attr.s
class _InfoStateNode(object):
    """An object wrapping values associated to an information state."""
    legal_actions = attr.ib()
    index_in_tabular_policy = attr.ib()
    cumulative_regret = attr.ib(factory=random_dict_factory)
    cumulative_policy = attr.ib(factory=random_dict_factory)


class CFR(_CFRSolver):
    """
    This CFR algorithm is only for demonstration and currently only used in Kuhn Poker.
    """
    def __init__(self, game):
        super().__init__(
            game,
            regret_matching_plus=False,
            alternating_updates=False,
            linear_averaging=False
        )
        self.node_touched = 0

    def _initialize_info_state_nodes(self, state):
        """Initializes info_state_nodes.

        Create one _InfoStateNode per infoset. We could also initialize the node
        when we try to access it and it does not exist.

        Args:
          state: The current state in the tree walk. This should be the root node
            when we call this function from a CFR solver.
        """
        if state.is_terminal():
            return

        if state.is_chance_node():
            for action, unused_action_prob in state.chance_outcomes():
                self._initialize_info_state_nodes(state.child(action))
            return

        current_player = state.current_player()
        info_state = state.information_state_string(current_player)

        info_state_node = self._info_state_nodes.get(info_state)
        if info_state_node is None:
            legal_actions = state.legal_actions(current_player)
            info_state_node = _InfoStateNode(
                legal_actions=legal_actions,
                index_in_tabular_policy=self._current_policy.state_lookup[info_state])
            self._info_state_nodes[info_state] = info_state_node

        for action in info_state_node.legal_actions:
            self._initialize_info_state_nodes(state.child(action))

    def _compute_counterfactual_regret_for_player(self, state, policies,
                                                  reach_probabilities, player):
        self.node_touched += 1
        return super()._compute_counterfactual_regret_for_player(state, policies, reach_probabilities, player)

    def iteration(self):
        self.evaluate_and_update_policy()

    def show_policy(self, info_keys):
        for i_info_key in info_keys:
            act0 = self._info_state_nodes[i_info_key].cumulative_policy[0]
            act1 = self._info_state_nodes[i_info_key].cumulative_policy[1]
            tmp_policy = np.array([act0, act1])
            tmp_policy = tmp_policy / np.sum(tmp_policy)
            print(tmp_policy)

    def get_policy(self, info_key):
        act0 = self._info_state_nodes[info_key].cumulative_policy[0]
        act1 = self._info_state_nodes[info_key].cumulative_policy[1]
        tmp_policy = np.array([act0, act1])
        tmp_policy = tmp_policy / np.sum(tmp_policy)
        return tmp_policy[1]


def _pref_regret_matching(cumulative_regrets, legal_actions, pref_config):
    """Returns an info state policy by applying regret-matching.

    Args:
      cumulative_regrets: A {action: cumulative_regret} dictionary.
      legal_actions: the list of legal actions at this state.

    Returns:
      A dict of action -> prob for all legal actions.
    """

    delta = pref_config[0]
    beta = pref_config[1]
    regrets = cumulative_regrets.values()
    regrets = [regret - beta for regret in regrets]
    sum_positive_regrets = 0
    for i_legal_action in legal_actions:
        if regrets[i_legal_action] > 0:
            sum_positive_regrets += regrets[i_legal_action] * delta[i_legal_action]

    info_state_policy = {}
    if sum_positive_regrets > 0:
        '''
        This is the formula for Pref-CFR(RM), corresponding to Equations 13 and 14 in the paper.
        '''
        # for action in legal_actions:
        #     positive_action_regret = max(0.0, cumulative_regrets[action])
        #     info_state_policy[action] = (
        #             positive_action_regret * delta[action] / sum_positive_regrets)
        '''
        This is the formula for Pref-CFR(RM), corresponding to Equations 15 and 16 in the paper.
        '''
        max_idx = 0
        for action in legal_actions:
            if cumulative_regrets[action] * delta[action] > cumulative_regrets[max_idx] * delta[max_idx]:
                max_idx = action
        for action in legal_actions:
            if action == max_idx:
                info_state_policy[action] = 1.0
            else:
                info_state_policy[action] = 0.0
    else:
        sum_delta = np.sum(delta) - len(legal_actions)
        for action in legal_actions:
            info_state_policy[action] = (delta[action] - 1.0) / sum_delta
    return info_state_policy


def update_current_policy_pref(current_policy, info_state_nodes, pref_config):
    """Updates in place `current_policy` from the cumulative regrets.

    This function is a module level function to be reused by both CFRSolver and
    CFRBRSolver.

    Args:
      current_policy: A `policy.TabularPolicy` to be updated in-place.
      info_state_nodes: A dictionary {`info_state_str` -> `_InfoStateNode`}.
    """
    for info_state, info_state_node in info_state_nodes.items():
        state_policy = current_policy.policy_for_key(info_state)

        if info_state not in pref_config.keys():
            info_state_pref_config = [
                np.array([1, 1]),
                0
            ]
        else:
            info_state_pref_config = pref_config[info_state]
        for action, value in _pref_regret_matching(
                info_state_node.cumulative_regret,
                info_state_node.legal_actions,
                info_state_pref_config).items():
            state_policy[action] = value


class PrefCFR(CFR):
    def __init__(self, game, pref_config=None):
        super().__init__(game)
        if pref_config is None:
            pref_config = {}
        self.pref_config = pref_config

    def evaluate_and_update_policy(self):
        """Performs a single step of policy evaluation and policy improvement."""
        self._iteration += 1
        if self._alternating_updates:
            for player in range(self._game.num_players()):
                self._compute_counterfactual_regret_for_player(
                    self._root_node,
                    policies=None,
                    reach_probabilities=np.ones(self._game.num_players() + 1),
                    player=player)
                update_current_policy_pref(self._current_policy, self._info_state_nodes, self.pref_config)
        else:
            self._compute_counterfactual_regret_for_player(
                self._root_node,
                policies=None,
                reach_probabilities=np.ones(self._game.num_players() + 1),
                player=None)
            update_current_policy_pref(self._current_policy, self._info_state_nodes, self.pref_config)
