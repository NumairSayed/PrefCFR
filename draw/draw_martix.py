import matplotlib.pyplot as plt
import numpy as np
import os
from draw.convergence_rate import plot_once


def plt_perfect_game_convergence_inline(logdir, is_x_log=True, is_y_log=True, y_label_index=3, x_label_index=0):
    file_list = [f for f in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, f))]
    file_list.sort()

    for i_file in range(len(file_list)):
        plot_once(
            logdir + '/' + file_list[i_file],
            i_file,
            file_list[i_file],
            is_x_log=is_x_log,
            is_y_log=is_y_log,
            y_label_index=y_label_index,
            x_label_index=x_label_index
        )
