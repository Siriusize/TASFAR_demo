from dist_est import gen_den_map, cal_den
from dataset import NYCTaxiDataset
from network import ANN
from const import TRAINED_MODEL_PATH, THRESHOLD, QFUNC
import torch
from torch.utils.data import DataLoader
from conf_classifier import cal_var
import json
import numpy as np
from scipy.stats import norm
import math
import time


# Generate density map
def gen_den(blosk_size):
    with open('../data/wkend_var_std.json', 'r') as fp:
        json_data = json.load(fp)
        mean_std_list = []
        conf_count = 0
        for data in list(json_data.values()):
            # Check how many data are confident data
            if data[0] < THRESHOLD:
                mean_std_list.append([data[1], abs(data[1]-data[2])])
                conf_count += 1
        print('%s pieces of data are confident' % conf_count)

        mean_std_list = np.array(mean_std_list)
        min_data = mean_std_list[:, 0] - 3 * mean_std_list[:, 1]
        max_data = mean_std_list[:, 0] + 3 * mean_std_list[:, 1]
        minimum = np.min(min_data)
        maximum = np.max(max_data)
        num_block = math.ceil((maximum - minimum) / block_size)

        # Generate density map
        den_map = np.zeros(num_block)
        start_t = time.time()
        for i, data in enumerate(mean_std_list):
            den_list = cal_den(data[0], data[1], minimum, num_block, block_size)
            for d in den_list:
                den_map[d[0]] += d[1]
            if i % 1000 == 0:
                print(i, time.time() - start_t)
        den_map /= np.sum(den_map)
        return den_map, minimum + block_size / 2 + np.arange(0, num_block) * block_size, minimum, num_block


def check_pseudo_label(den_map, est_map, minimum, num, block_size):
    total_msle = 0
    total_count = 0
    with open('../data/wkend_var_std.json', 'r') as fp:
        json_data = json.load(fp)
    mean_std_list = []
    for data in list(json_data.values()):
        if data[0] < THRESHOLD:
            total_msle += (np.log(data[1]+1) - np.log(data[2]+1)) ** 2
            total_count += 1
        else:
            mean_std_list.append([data[1], abs(data[1]-data[2]), data[2]])

    # Generate pseudo label
    for mean_std in mean_std_list:
        den_list = cal_den(mean_std[0], mean_std[1], minimum, num, block_size)
        pseudo_list = []  # To be used for interpolation
        for d in den_list:
            pseudo_list.append((est_map[d[0]], den_map[d[0]] * d[1], den_map[d[0]]))
        pseudo_array = np.array(pseudo_list)
        pseudo_label = np.average(pseudo_array[:, 0], weights=pseudo_array[:, 1]).item()
        if pseudo_label < 0:
            pseudo_label = 0
        total_msle += (np.log(pseudo_label+1) - np.log(mean_std[2]+1)) ** 2
        total_count += 1

    return total_msle, total_count


block_size=100
den_map, est_map, minimum, num = gen_den_map(block_size)
print('-'*20, block_size, '-'*20)
total_msle, total_count = check_pseudo_label(den_map, est_map, minimum, num, block_size)
print(total_count)
print(np.sqrt(total_msle / total_count))