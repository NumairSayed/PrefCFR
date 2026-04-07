import pyspiel
import os
import time
from open_spiel.python.algorithms import exploitability

from algorithm.MCCFR import ES_MCCFR
from algorithm.MCCFVFP import ES_MCCFVFP
from algorithm.PrefCFR import ES_MCPrefCFR
from algorithm.CFR import CFR
from algorithm.CFR import PrefCFR
import csv
from Game_config import game_config, pref_config

from draw.draw_martix import plt_perfect_game_convergence_inline
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def train_parallel(train_config):
    game_name = train_config['game_name']
    path = train_config["path"]
    solver = train_config["solver"]
    game = train_config["game"]

    counter = 10000000
    result_data = {
        "node_touch"    : [],
        "exploitability": [],
        "time"          : [],
        "policy1"       : []
    }
    start_time = time.time()
    print_node = 1000
    while True:
        if solver.node_touched >= print_node:
            print_node *= 1.5
            conv = exploitability.nash_conv(game, solver.average_policy())
            now_time = time.time()
            result_data["node_touch"].append(solver.node_touched)
            result_data["exploitability"].append(conv)
            result_data["time"].append(now_time - start_time)
            if game_name == "kuhn_poker":
                result_data["policy1"].append(solver.get_policy("0"))
            elif game_name == "leduc_poker":
                result_data["policy1"].append(solver.get_policy(
                    "[Observer: 0][Private: 5][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]"))
            print("node_touch", "time", now_time - start_time, solver.node_touched, "exploitability", conv)

        if solver.node_touched >= counter:
            conv = exploitability.nash_conv(game, solver.average_policy())
            now_time = time.time()
            result_data["node_touch"].append(solver.node_touched)
            result_data["exploitability"].append(conv)
            result_data["time"].append(now_time - start_time)
            if game_name == "kuhn_poker":
                result_data["policy1"].append(solver.get_policy("0"))
            elif game_name == "leduc_poker":
                result_data["policy1"].append(solver.get_policy(
                    "[Observer: 0][Private: 5][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]"))
            print("node_touch", "time", now_time - start_time, solver.node_touched, "exploitability", conv)
            break

        solver.iteration()
    if game_name == "kuhn_poker":
        show_info_list = ["0", "1", "2"]
    elif game_name == "leduc_poker":
        show_info_list = [
            "[Observer: 0][Private: 0][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]",
            "[Observer: 0][Private: 1][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]",
            "[Observer: 0][Private: 2][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]",
            "[Observer: 0][Private: 3][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]",
            "[Observer: 0][Private: 4][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]",
            "[Observer: 0][Private: 5][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]"
        ]

    solver.show_policy(show_info_list)
    os.makedirs(path)
    with open(path + '/epsilon.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        result_keys = list(result_data.keys())
        writer.writerow(result_keys)
        for i_eps in range(len(result_data['node_touch'])):
            tmp_write_data = []
            for result_key in result_keys:
                tmp_write_data.append(result_data[result_key][i_eps])
            writer.writerow(tmp_write_data)


def train_one_setting(mode, game_name, result_file_path, pref_mode=None):
    num_cpu = 10
    num_train_times = 10
    train_config_list = []
    for train_times in range(num_train_times):
        print(mode, pref_mode, train_times)
        train_config = {
            "game_name": game_name,
            "game": pyspiel.load_game(game_name, game_config[game_name]),
            "path": result_file_path + '/' + mode + '_' + pref_mode + '/' + str(train_times)
        }
        game = pyspiel.load_game(game_name, game_config[game_name])
        if mode == "ES-MCCFR":
            train_config["solver"] = ES_MCCFR(game)
        elif mode == "ES-MCCFVFP":
            train_config["solver"] = ES_MCCFVFP(game)
        elif mode == "ES-MCPrefCFR":
            tmp_pref_config = pref_config[game_name][pref_mode]
            train_config["solver"] = ES_MCPrefCFR(game, pref_config=tmp_pref_config)
        elif mode == "CFR":
            train_config["solver"] = CFR(game)
        elif mode == "PrefCFR":
            tmp_pref_config = pref_config[game_name][pref_mode]
            train_config["solver"] = PrefCFR(game, pref_config=tmp_pref_config)

        train_config_list.append(train_config)

    Parallel(n_jobs=num_cpu)(
        delayed(train_parallel)(i_train_config) for i_train_config in train_config_list
    )


def train(game_name):
    now_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    now_path_str = os.path.dirname(os.path.abspath(__file__))
    result_file_path = ''.join(
        [
            now_path_str,
            '/results',
            '/',
            game_name + '_' + now_time,

        ]
    )
    '''
    In the Kuhn experiment, only the PrefCFR(BR) form is demonstrated here.
    If you need other forms of PrefCFR,
    you can modify the code in algorithm.CFR Line 114.
    PrefCFR by yourself. This corresponds to Figures 1 and 2 in the paper.
    '''
    train_one_setting("PrefCFR", game_name, result_file_path, "defensive10")
    train_one_setting("PrefCFR", game_name, result_file_path, "defensive5")
    train_one_setting("CFR", game_name, result_file_path, "normal")
    train_one_setting("PrefCFR", game_name, result_file_path, "offensive5")
    train_one_setting("PrefCFR", game_name, result_file_path, "offensive10")

    '''
    The Leduc experiment corresponds to Figure 3 in the paper.
    '''
    # train_one_setting("ES-MCPrefCFR", game_name, result_file_path, "offensive_10_05")
    # train_one_setting("ES-MCPrefCFR", game_name, result_file_path, "offensive_10_00")
    # train_one_setting("ES-MCPrefCFR", game_name, result_file_path, "offensive_5_05")
    # train_one_setting("ES-MCPrefCFR", game_name, result_file_path, "normal")
    # train_one_setting("ES-MCPrefCFR", game_name, result_file_path, "defensive_10_05")
    # train_one_setting("ES-MCPrefCFR", game_name, result_file_path, "defensive_10_00")
    # train_one_setting("ES-MCPrefCFR", game_name, result_file_path, "defensive_5_05")

    '''
    The Leduc experiment corresponds to Figure 4 in the paper.
    '''
    # train_one_setting("ES-MCPrefCFR", game_name, result_file_path, "offensive_5_05")
    # train_one_setting("ES-MCPrefCFR", game_name, result_file_path, "offensive_5_10")
    # train_one_setting("ES-MCPrefCFR", game_name, result_file_path, "offensive_5_20")
    # train_one_setting("ES-MCPrefCFR", game_name, result_file_path, "offensive_5_50")
    # train_one_setting("ES-MCPrefCFR", game_name, result_file_path, "normal")

    return result_file_path


if __name__ == '__main__':
    game_name = "kuhn_poker"
    # game_name = "leduc_poker"
    test_name = train(game_name)

    plt.figure(figsize=(32, 10), dpi=240)

    plt.subplot(121)
    plt_perfect_game_convergence_inline(
        test_name,
        is_x_log=True,
        x_label_index=0,
        y_label_index=1,
    )

    plt.legend(edgecolor='red', fontsize=20)
    plt.xlabel('Node touched', fontsize=24)
    plt.ylabel('log10(NE gap)', fontsize=24)

    plt.subplot(122)
    plt_perfect_game_convergence_inline(
        test_name,
        is_x_log=True,
        is_y_log=False,
        x_label_index=0,
        y_label_index=3
    )
    plt.xlabel('Node touched', fontsize=24)
    plt.ylabel('policy', fontsize=24)

    plt.savefig(test_name + "/conv.png", dpi=240)
    # plt.show()
