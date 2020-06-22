#!/usr/bin/python

import os.path as path
from csv import DictReader

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, FormatStrFormatter, PercentFormatter

ROOT_DIR = '/home/mrapp/Dropbox/Promotion/Papers/Boomer/Experimente/results_round5/evaluation/'

DATASET = 'synthetic_num-examples=10000_num-labels=6_marginal-independence_p=0.1'
# DATASET = 'synthetic_num-examples=10000_num-labels=6_tau=0.0_p=0.1'
# DATASET = 'synthetic_num-examples=10000_num-labels=6_tau=1.0_p=0.1_dependent-error'
# DATASET = 'synthetic_num-examples=10000_num-labels=6_one-error'

FILE_NAME = 'evaluation_overall.csv'

HAMMING_LOSS = 'Hamm. Loss'

SUBSET_0_1_LOSS = 'Subs. 0/1 Loss'

THRESHOLD = 0.0

STEP_SIZE = 1

MIN_RULES = 1

MAX_RULES = 1000

OUTPUT_DIR = None
# OUTPUT_DIR = '/home/mrapp/Dropbox/Promotion/Papers/Boomer/Experimente/results_round5/'

if __name__ == '__main__':
    plt.rc('font', size=12)  # controls default text sizes

    dir_macro = path.join(path.join(ROOT_DIR, 'macro-logistic-loss'), DATASET)
    dir_macro_single = path.join(dir_macro, 'single-label')
    dir_macro_full = path.join(dir_macro, 'full')
    dir_example_based = path.join(path.join(ROOT_DIR, 'example-based-logistic-loss'), DATASET)
    dir_example_based_single = path.join(dir_example_based, 'single-label')
    dir_example_based_full = path.join(dir_example_based, 'full')
    directories = [
        (dir_macro_single, 's', 'l.w.-log. single', MAX_RULES),
        (dir_macro_full, 'o', 'l.w.-log. multi', MAX_RULES),
        (dir_example_based_single, '^', 'ex.w.-log. single', MAX_RULES),
        (dir_example_based_full, 'x', 'ex.w.-log. multi', MAX_RULES)
    ]

    x_min = 1.0
    x_max = 0.0
    y_min = 1.0
    y_max = 0.0

    for i in np.linspace(0, 1, 11):
        isometrics_line_width = 1
        isometrics_line_style = 'dashed'
        isometrics_color = '0.75'
        plt.plot([i, i + 1], [0, 1], linewidth=isometrics_line_width, linestyle=isometrics_line_style,
                 color=isometrics_color)
        plt.plot([0, 1], [i, i + 1], linewidth=isometrics_line_width, linestyle=isometrics_line_style,
                 color=isometrics_color)

    for directory, marker, label, max_rules in directories:
        x = []
        y = []
        cur_x_min = 1.0
        cur_x_max = 0.0
        cur_y_min = 1.0
        cur_y_max = 0.0

        with open(path.join(directory, FILE_NAME), mode='r', newline='') as csv_file:
            reader = DictReader(csv_file, delimiter=',', quotechar='"')

            for i, row in enumerate(reader):
                if (MIN_RULES - 1) <= i < max_rules and i % STEP_SIZE == 0:
                    hamming_loss = float(row[HAMMING_LOSS])
                    subset_0_1_loss = float(row[SUBSET_0_1_LOSS])

                    if len(x) == 0 or abs(hamming_loss - x[len(x) - 1]) > THRESHOLD or abs(
                            subset_0_1_loss - y[len(y) - 1]) > THRESHOLD:
                        x.append(hamming_loss)
                        y.append(subset_0_1_loss)
                        cur_x_min = min(hamming_loss, cur_x_min)
                        cur_x_max = max(hamming_loss, cur_x_max)
                        cur_y_min = min(subset_0_1_loss, cur_y_min)
                        cur_y_max = max(subset_0_1_loss, cur_y_max)

        x_min = min(cur_x_min, x_min)
        x_max = max(cur_x_max, x_max)
        y_min = min(cur_y_min, y_min)
        y_max = max(cur_y_max, y_max)

        marker_size = 7
        curve = plt.plot(x, y, label=label)
        color = curve[0].get_color()

        next_tick = 1

        for i in range(len(x)):
            if (i + 1) % next_tick == 0:
                point_x = x[i]
                point_y = y[i]
                plt.plot(point_x, point_y, marker=marker, markersize=marker_size, color=color)
                next_tick = next_tick * 2

        alpha = 0.4
        plt.plot([cur_x_min, cur_x_min], [0, 1], color=color, alpha=alpha)
        plt.plot([0, 1], [cur_y_min, cur_y_min], color=color, alpha=alpha)

    padding_factor = 0.05
    padding_x = padding_factor * (x_max - x_min)
    x_min = max(0, x_min - padding_x)
    x_max = min(1, x_max + padding_x)
    padding_y = padding_factor * (y_max - y_min)
    y_min = max(0, y_min - padding_y)
    y_max = min(1, y_max + padding_y)

    ax = plt.gca()
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect(1 / ax.get_data_ratio())
    formatter = PercentFormatter(xmax=1, decimals=0)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    plt.xlabel('Hamming loss')
    plt.ylabel('Subset 0/1 loss')
    plt.legend()

    if OUTPUT_DIR is None:
        plt.show()
    else:
        output_file = path.join(OUTPUT_DIR, DATASET + '.pdf')
        plt.savefig(output_file)
