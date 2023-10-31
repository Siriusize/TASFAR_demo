import pickle
import json
import numpy as np
from scipy.stats import norm
from dataset import NYCTaxiDataset
from torch.utils.data import DataLoader
import torch
from network import ANN
import math


# Given parameters
THRESHOLD = 12102.397583007812
Q_FUNC = (1.08138730e+02, 4.66947225e-02)
BLOCK_SIZE = 100


def cal_var_lst(lst):
    """
    Args:
        lst: a list of prediction
    Returns:
        variance of lst
    """
    lst = torch.concat(lst, dim=1).detach().cpu().numpy()  # (1024, 20)
    return np.var(lst, axis=1)


def cal_cdf(x, mean, std):
    return norm.cdf((x-mean)/std)


def cal_block_indices(mean, std, side_info):
    """
    Args:
        mean: mean of the prediction
        std: std of the predicton
        side_info: a tuple giving the information of density map, (min, max, number_of_block+1, block_size)
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


def gen_pseudo_label(model_path, q_func, block_size, device):
    """
    Args:
        model_path: path of pretrained model
        q_func: q function
        block_size: block size of density map
        device: device
    Returns:
        pseudo_label_dict: a dictionary containing the pseudo label
    """

    # Load dataset
    scaler = pickle.load(open('../data/scaler.pkl', 'rb'))
    test_dataset = NYCTaxiDataset(data_path='../data/nyc_test.csv', scaler=scaler)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    intercept, slope = q_func

    # Load network
    device = torch.device(device)
    net = ANN(input_size=56, output_size=1, hidden_sizes=[128, 128, 64], dropout=0.2)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)

    # Collect prediction data
    pred_data = []
    for i, data in enumerate(test_dataloader):
        x, label, data_index = data
        x = x.to(device)
        # calculating variance
        pred_list = []
        net.train()
        for j in range(20):
            pred = net(x)
            pred_list.append(pred)
        var = cal_var_lst(pred_list)

        # collecting prediction
        net.eval()
        prediction = np.squeeze(net(x).detach().cpu().numpy())
        label = np.squeeze(label)

        for k in range(data_index.shape[0]):
            pred_data.append([prediction[k].item(), slope*var[k].item()+intercept, var[k].item(), k, label[k].item()])

    # Generate density map
    pred_data = np.array(pred_data)
    min_data = pred_data[:, 0] - 3 * pred_data[:, 1]
    max_data = pred_data[:, 0] + 3 * pred_data[:, 1]
    minimum = np.min(min_data)
    maximum = np.max(max_data)
    num_block = math.ceil((maximum - minimum) / block_size)
    side_info = (minimum, minimum+num_block*block_size, num_block+1, block_size)

    # Generate density map
    den_map = np.zeros(num_block)
    for data in pred_data:
        if data[2] < THRESHOLD:
            den_list = cal_block_indices(data[0], data[1], side_info)
            for d in den_list:
                den_map[d[0]] += d[1]
    den_map /= np.sum(den_map)

    # Generate estimation map
    est_map = (np.linspace(*side_info[:3]) * 2 + block_size) / 2

    # Generate pseudo label dictionary
    pseudo_label_dict = {}
    for data in pred_data:
        if data[2] <= THRESHOLD:
            pseudo_label_dict[int(data[3])] = {
                'pseudo_label': data[0],
                'variance': data[2],
                'lmd': 1/den_map.shape[0],
                'gmd': 1/den_map.shape[0],
                'label': data[4]
            }
        else:
            den_list = cal_block_indices(data[0], data[1], side_info)
            pseudo_list = []  # To be used for interpolation
            for d in den_list:
                pseudo_list.append((est_map[d[0]], den_map[d[0]] * d[1], den_map[d[0]]))
            pseudo_array = np.array(pseudo_list)
            pseudo_label = np.average(pseudo_array[:, 0], weights=pseudo_array[:, 1]).item()
            lmd = np.mean(pseudo_array[:, 2]).item()
            pseudo_label_dict[int(data[3])] = {
                'pseudo_label': pseudo_label,
                'variance': data[2],
                'lmd': lmd,
                'gmd': 1 / den_map.shape[0],
                'label': data[4]
            }
    return pseudo_label_dict


def check_pseudo_label(pseudo_label_dict):
    total_mse = 0
    total_count = 0
    for k in pseudo_label_dict.keys():
        data = pseudo_label_dict[k]
        total_mse += (np.log(data['pseudo_label']+1) - np.log(data['label']+1)) ** 2
        total_count += 1
    mse_b = 0.5119
    mse_a = np.sqrt(total_mse/total_count)
    print('-' * 60)
    print('Duration RMSLE before adaptation: %.4f' % mse_b)
    print('Duration RMSLE after adaptation: %.4f' % mse_a)
    print('RMSLE reduction rate: %.2f%%' % ((mse_b-mse_a)/mse_b*100))
    print('-' * 60)


if __name__ == "__main__":
    model_path = '../model/pretrained_model.pt'
    block_size = 100
    pseudo_label_dict = gen_pseudo_label(model_path, Q_FUNC, block_size, device='cuda:0')
    save_path = '../data/pseudo_label_train.json'
    with open(save_path, 'w') as fp:
        json.dump(pseudo_label_dict, fp)
    check_pseudo_label(pseudo_label_dict)
