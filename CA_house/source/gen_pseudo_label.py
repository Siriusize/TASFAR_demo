from const import QFUNC, THRESHOLD
import json
import numpy as np


def cal_cdf(x, mean, std):
    return norm.cdf((x-mean)/std)


def cal_den(et_count, std, minimum, num, size):
    """
    Calculate densities given a point's et_count and std
    Args:
        et_count: estimation
        std: standard deviation
        minimum: minimum value of the density map
        num: number of block in the density map
        size: block size of the density map
    Returns:
        A list of (slot, density)
    """
    den_list = []  # to be returned
    sigma_range = [et_count-3*std, et_count+3*std]  # the range [mean-3*sigma, mean+3*sigma] of the point
    partitions = minimum + np.arange(0, num) * size  # left side of the block in the density map
    # partitions in the range
    pos = np.where((partitions >= sigma_range[0]) & (partitions < sigma_range[1]))[0]
    values = partitions[pos]
    if values.shape[0] != 0:
        for i, (p, v) in enumerate(zip(pos, values)):
            if p == 0:
                continue
            elif i == 0:
                den_list.append([p-1, cal_cdf(v, et_count, std)])
            else:
                den_list.append([p-1, cal_cdf(v, et_count, std) - cal_cdf(partitions[p-1], et_count, std)])
        den_list.append([pos[-1], 1-cal_cdf(values[-1], et_count, std)])
    else:
        for i, p in enumerate(partitions):
            if sigma_range[0] >= p:
                den_list.append([i, 1])
                break
    return den_list


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
    with open('../data/poor_var_std.json', 'r') as fp:
        json_data = json.load(fp)
    mean_std_list = []
    conf_count = 0
    for data in json_data:
        # Check how many data are confident data
        if data[0] < THRESHOLD:
            mean_std_list.append([data[1], data[0] * QFUNC[1] + QFUNC[0]])
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
    for data in mean_std_list:
        den_list = cal_den(data[0], data[1], minimum, num_block, block_size)
        for d in den_list:
            den_map[d[0]] += d[1]
    den_map /= np.sum(den_map)
    return den_map, minimum + block_size / 2 + np.arange(0, num_block) * block_size, minimum, num_block


def check_pseudo_label(den_map, est_map, minimum, num, block_size):
    """
    Generate pseudo label
    Args:
        den_map: density map generated from gen_den()
        est_map: estimation map generated from gen_den()
        minimum: minimum value of the density map
        num: number of block in the density map
        block_size: side length of the block in the density map
    Returns:
        mse: MSE using pseudo label
    """
    total_mse = 0
    total_count = 0

    with open('../data/poor_var_std.json', 'r') as fp:
        json_data = json.load(fp)
    mean_std_list = []
    for data in json_data:
        if data[0] < THRESHOLD:
            total_mse += (data[1] - data[2]) ** 2
            total_count += 1
        else:
            mean_std_list.append([data[1], data[0] * QFUNC[1] + QFUNC[0], data[2]])

    # Generate pseudo label
    for mean_std in mean_std_list:
        den_list = cal_den(mean_std[0], mean_std[1], minimum, num, block_size)
        pseudo_list = []  # To be used for interpolation
        for d in den_list:
            pseudo_list.append((est_map[d[0]], den_map[d[0]] * d[1], den_map[d[0]]))
        pseudo_array = np.array(pseudo_list)
        pseudo_label = np.average(pseudo_array[:, 0], weights=pseudo_array[:, 1]).item()
        total_mse += (pseudo_label - mean_std[2]) ** 2
        total_count += 1

    return total_mse, total_count


def col_pseudo_label():
    pseudo_label_dict = {}

    block_size = 1.0
    den_map, est_map, minimum, num = gen_den_map(block_size)

    with open('../data/poor_var_std.json', 'r') as fp:
        json_data = json.load(fp)
    mean_std_list = []
    for i, data in enumerate(json_data):
        if data[0] >= THRESHOLD:
            mean_std_list.append([data[1], data[0] * QFUNC[1] + QFUNC[0], i, data[0], data[2]])
        else:
            pseudo_label_dict[i] = {
                'pseudo_label': data[1],
                'variance': data[0],
                'lmd': 1/den_map.shape[0],
                'gmd': 1/den_map.shape[0],
                'label': data[2]
            }

    # Generate pseudo label
    for mean_std in mean_std_list:
        den_list = cal_den(mean_std[0], mean_std[1], minimum, num, block_size)
        pseudo_list = []  # To be used for interpolation
        for d in den_list:
            pseudo_list.append((est_map[d[0]], den_map[d[0]] * d[1], den_map[d[0]]))
        pseudo_array = np.array(pseudo_list)
        pseudo_label = np.average(pseudo_array[:, 0], weights=pseudo_array[:, 1]).item()
        lmd = np.mean(pseudo_array[:, 2]).item()
        gmd = 1/den_map.shape[0]
        pseudo_label_dict[mean_std[2]] = {
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
