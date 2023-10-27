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


def cal_cdf(x, mean, std):
    return norm.cdf((x-mean)/std)


def cal_den(mean, std, side_info):
    """
    Args:
        mean: mean of the prediction
        std: std of the predicton
        side_info: a tuple giving the information of x/y-dimension, (min, max, number_of_block+1, block_size)
    Returns:
        a list of (slot, density)
    """
    den_list = []
    min_3sigma = mean - 3*std
    max_3sigma = mean + 3*std
    partitions = np.linspace(*side_info[:3])
    in_range = []
    for p in partitions:
        if min_3sigma < p < max_3sigma:
            in_range.append(p.item())
    if in_range:
        for i in range(len(in_range)):
            slot = round((in_range[i] - side_info[3] - side_info[0]) / side_info[3])
            if i == 0:
                den = cal_cdf(in_range[i], mean, std)
            else:
                den = cal_cdf(in_range[i], mean, std) - cal_cdf(in_range[i-1], mean, std)
            den_list.append([slot, den])
        if in_range[-1] != partitions[-1]:
            den_list.append([round((in_range[-1] - side_info[0]) / side_info[3]), 1 - cal_cdf(in_range[-1], mean, std)])
    else:
        for p in partitions:
            if min_3sigma <= p:
                slot = round((p - side_info[3] - side_info[0]) / side_info[3])
                den_list.append([slot, 1])
                break
    return den_list


def check_test_result():
    with open('../data/test_var_std.json', 'r') as fp:
        json_data = json.load(fp)

    total_msle = 0
    total_count = 0
    for data in list(json_data.values()):
        if data[0] < THRESHOLD:
            total_msle += (np.log(data[1]+1) - np.log(data[2]+1)) ** 2
        # total_msle += (data[1] - data[2]) ** 2
        total_count += 1
    print(np.sqrt(total_msle/total_count))


def gen_den_map(block_size):
    """
    Generate density map
    Args:
        block_size: side length of the block in the density map
    Returns:
        den_map: density map
        est_map: estimation map, i.e., the corresponding value in each block
        minimum: minimum value of the density map
        num: number of block in the density map
    """

    with open('../data/test_var_std.json', 'r') as fp:
        json_data = json.load(fp)
    mean_std_list = []
    conf_count = 0
    for data in list(json_data.values()):
        # Check how many data are confident data
        if data[0] < THRESHOLD:
            mean_std_list.append([data[1], data[0]*QFUNC[1]+QFUNC[0]])
            conf_count += 1
    print('%s pieces of data (%.2f%%) are confident' % (conf_count, conf_count/len(json_data.values())))

    mean_std_list = np.array(mean_std_list)
    min_data = mean_std_list[:, 0] - 3 * mean_std_list[:, 1]
    max_data = mean_std_list[:, 0] + 3 * mean_std_list[:, 1]
    minimum = np.min(min_data)
    maximum = np.max(max_data)
    num_block = math.ceil((maximum - minimum) / block_size)
    side_info = (minimum, minimum+num_block*block_size, num_block+1, block_size)

    # Generate density map
    den_map = np.zeros(num_block)
    start_t = time.time()
    for i, data in enumerate(mean_std_list):
        den_list = cal_den(data[0], data[1], side_info)
        for d in den_list:
            den_map[d[0]] += d[1]
        if i % 1000 == 0:
            print(i, time.time() - start_t)
    den_map /= np.sum(den_map)
    return den_map, (np.linspace(*side_info[:3]) * 2 + block_size) / 2, side_info


if __name__ == "__main__":
    check_test_result()
    # col_var_std()
