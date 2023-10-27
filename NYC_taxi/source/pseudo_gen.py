from dist_est import gen_den_map, cal_den
from const import QFUNC, THRESHOLD
import json
import numpy as np


def check_pseudo_label(den_map, est_map, side_info):
    total_msle = 0
    total_count = 0

    with open('../data/test_var_std.json', 'r') as fp:
        json_data = json.load(fp)
    mean_std_list = []
    for data in list(json_data.values()):
        if data[0] < THRESHOLD:
            total_msle += (np.log(data[1]+1) - np.log(data[2]+1)) ** 2
            total_count += 1
        else:
            mean_std_list.append([data[1], data[0]*QFUNC[1]+QFUNC[0], data[2]])

    # Generate pseudo label
    count = [0, 0, 0]
    for mean_std in mean_std_list:
        den_list = cal_den(mean_std[0], mean_std[1], side_info)
        pseudo_list = []  # To be used for interpolation
        for d in den_list:
            pseudo_list.append((est_map[d[0]], den_map[d[0]] * d[1], den_map[d[0]]))
        if pseudo_list:
            pseudo_array = np.array(pseudo_list)
            if np.sum(pseudo_array[:, 1]) == 0:
                pseudo_label = mean_std[0]
            else:
                pseudo_label = np.average(pseudo_array[:, 0], weights=pseudo_array[:, 1]).item()
        else:
            pseudo_label = mean_std[0]
        if abs(pseudo_label - mean_std[2]) < abs(mean_std[0] - mean_std[2]):
            count[0] += 1
        elif abs(pseudo_label - mean_std[2]) > abs(mean_std[0] - mean_std[2]):
            count[1] += 1
        else:
            count[2] += 1
        total_msle += (np.log(pseudo_label+1) - np.log(mean_std[2]+1)) ** 2
        total_count += 1

    print(count)

    return total_msle, total_count


def check_loop():
    block_size_list = [100]
    for block_size in block_size_list:
        den_map, est_map, side_info = gen_den_map(block_size)
        print('-'*20, block_size, '-'*20)
        total_msle, total_count = check_pseudo_label(den_map, est_map, side_info)
        print(total_count)
        print(np.sqrt(total_msle / total_count))


def col_pseudo_label():
    pseudo_label_dict = {}

    block_size = 100
    den_map, est_map, side_info = gen_den_map(block_size)

    with open('../data/test_var_std.json', 'r') as fp:
        json_data = json.load(fp)

    mean_std_list = {}
    for key in json_data.keys():
        data = json_data[key]
        if data[0] >= THRESHOLD:
            mean_std_list[key] = [data[1], data[0] * QFUNC[1] + QFUNC[0], key, data[0], data[2]]
        else:
            pseudo_label_dict[key] = {
                'pseudo_label': data[1],
                'variance': data[0],
                'lmd': 1/den_map.shape[0],
                'gmd': 1/den_map.shape[0],
                'label': data[2]
            }

    # Generate pseudo label
    for key in mean_std_list.keys():
        mean_std = mean_std_list[key]

        den_list = cal_den(mean_std[0], mean_std[1], side_info)
        pseudo_list = []  # To be used for interpolation
        for d in den_list:
            pseudo_list.append((est_map[d[0]], den_map[d[0]] * d[1], den_map[d[0]]))
        pseudo_array = np.array(pseudo_list)
        pseudo_label = np.average(pseudo_array[:, 0], weights=pseudo_array[:, 1]).item()
        lmd = np.mean(pseudo_array[:, 2]).item()
        gmd = 1/den_map.shape[0]
        pseudo_label_dict[key] = {
            'pseudo_label': pseudo_label,
            'variance': mean_std[3],
            'lmd': lmd,
            'gmd': gmd,
            'label': mean_std[4]
        }

    with open('../data/poor_pseudo.json', 'w') as fp:
        json.dump(pseudo_label_dict, fp)


if __name__ == "__main__":
    col_pseudo_label()
    # check_loop()
