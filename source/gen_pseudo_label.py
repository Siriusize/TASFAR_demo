import json
import torch
import numpy as np
from scipy.stats import norm
from dataset import IMUDataset
from torch.utils.data import DataLoader
from model_temporal import TCNSeqNetwork
import math
import argparse
import random
from metric import compute_relative_trajectory_error

# Given parameters
THRESHOLD = 0.0020399182569235566
Q_FUNC = (9142.279915283665, 19.291012011076035)
BLOCK_SIZE = 10 * 2


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)


def cal_cdf(x, mean, std):
    return norm.cdf((x-mean)/std)


def cal_block_indices(mean, std, side_info):
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
        den_list.append([round((in_range[-1] - side_info[0]) / side_info[3]), 1 - cal_cdf(in_range[-1], mean, std)])
    else:
        for p in partitions:
            if min_3sigma <= p:
                slot = round((p - side_info[3] - side_info[0]) / side_info[3])
                den_list.append([slot, 1])
                break
    return den_list


def calculate_variance(lst):
    displacement_list = []
    for i in range(len(lst)):
        dts = 0.005
        dis = lst[i] * dts
        dis_sum = torch.sum(dis, axis=0)
        dis_sum = dis_sum.tolist()
        displacement_list.append(dis_sum)
    # Calculate Variance
    dis_tensor = torch.Tensor(displacement_list)
    mean = torch.mean(dis_tensor, axis=0)
    dis_tensor = dis_tensor - mean
    dis_tensor = torch.sqrt(torch.sum(dis_tensor ** 2, axis=1))
    return torch.var(dis_tensor).item()


def gen_pseudo_label(model_path, data_path, q_func, block_size, device):
    """
    Args:
        model_path: path of pretrained model
        data_path: data path of target domain
        q_func: q function
        block_size: block size of density map
        device: device
    Returns:
        pseudo_label_dict: a dictionary containing the pseudo label
    """

    # Load dataset
    adapt_dataset = IMUDataset(data_path)
    adapt_dataloader = DataLoader(adapt_dataset, shuffle=False)
    slope, intercept = q_func

    # Load network
    device = torch.device(device)
    network = TCNSeqNetwork(6, 2, 3, [32, 64, 128, 256, 72, 36])
    network.load_state_dict(torch.load(model_path).get('model_state_dict'))
    network.to(device)

    # Collect prediction data
    pred_data = []  # [x, y, var, frame_id]
    for i, data in enumerate(adapt_dataloader):
        feat, label, frame_id = data
        feat = feat.to(device)
        # calculating variance
        pred_list = []
        network.train()
        for j in range(20):
            pred = network(feat)
            pred = torch.squeeze(pred)
            pred_list.append(pred)
        var = calculate_variance(pred_list)

        # collecting prediction
        network.eval()
        pred = network(feat)
        pred = torch.squeeze(pred).cpu().detach().numpy()
        pred_sum = np.sum(np.array(pred), axis=0).tolist()

        pred_data.append([pred_sum[0], pred_sum[1], slope*var+intercept, var, frame_id.item()])

    # Generate density map
    pred_data = np.array(pred_data)
    min_data = np.transpose(np.transpose(pred_data[:, 0:2]) - 3 * pred_data[:, 2])
    max_data = np.transpose(np.transpose(pred_data[:, 0:2]) + 3 * pred_data[:, 2])
    min_xy = np.min(min_data, axis=0)
    max_xy = np.max(max_data, axis=0)
    x_num_block = math.ceil((max_xy[0] - min_xy[0]) / block_size)
    y_num_block = math.ceil((max_xy[1] - min_xy[1]) / block_size)
    x_info = (min_xy[0], min_xy[0]+x_num_block*block_size, x_num_block+1, block_size)
    y_info = (min_xy[1], min_xy[1]+y_num_block*block_size, y_num_block+1, block_size)

    den_map = np.zeros(shape=(x_num_block, y_num_block))
    for data in pred_data:
        if data[3] <= THRESHOLD:
            x_list = cal_block_indices(data[0], data[2], x_info)
            y_list = cal_block_indices(data[1], data[2], y_info)
            for x in x_list:
                for y in y_list:
                    den_map[x[0], y[0]] += x[1] * y[1]
    den_map /= np.sum(den_map)

    # Generate estimation map
    est_map = np.zeros(shape=(x_num_block, y_num_block, 2))
    est_map_x = (np.linspace(*x_info[:3]) * 2 + block_size) / 2
    est_map_y = (np.linspace(*y_info[:3]) * 2 + block_size) / 2
    for i in range(x_num_block):
        for j in range(y_num_block):
            est_map[i, j] = (est_map_x[i], est_map_y[j])

    # Generate pseudo label dictionary
    global_mean_density = np.mean(den_map).item()
    pseudo_label_dict = {}
    pseudo_label_dict['global_mean_density'] = global_mean_density
    for data in pred_data:
        if data[3] <= THRESHOLD:
            pseudo_label_dict[int(data[4])] = {
                'pred': [data[0], data[1]],
                'pseudo_label': [data[0], data[1]],
                'variance': data[3],
                'local_mean_density': global_mean_density,
                'is_pseudo_label': False
            }
        else:
            pseudo_list = []
            x_list = cal_block_indices(data[0], data[2], x_info)
            y_list = cal_block_indices(data[1], data[2], y_info)
            for x in x_list:
                for y in y_list:
                    pseudo_list.append((est_map[x[0], y[0]][0], est_map[x[0], y[0]][1], den_map[x[0], y[0]] * x[1] * y[1]))
            pseudo_list = np.array(pseudo_list)
            pseudo_label = np.average(pseudo_list[:, :2], axis=0, weights=pseudo_list[:, 2])
            pseudo_label_dict[int(data[4])] = {
                'pred': [data[0], data[1]],
                'pseudo_label': pseudo_label.tolist(),
                'variance': data[3],
                'local_mean_density': np.mean(pseudo_list[:, 2]).item(),
                'is_pseudo_label': True
            }
    return pseudo_label_dict


def cal_rte(pred, label, pred_per_min=30):
    dts = 0.005
    pos_pred = pred * dts
    pos_gt = label * dts
    pos_gt = np.cumsum(pos_gt, axis=0)
    pos_pred[0, :] = pos_gt[0, :]
    pos_pred = np.cumsum(pos_pred, axis=0)

    if pos_pred.shape[0] < pred_per_min:
        ratio = pred_per_min / pos_pred.shape[0]
        rte = compute_relative_trajectory_error(pos_pred, pos_gt, delta=pos_pred.shape[0] - 1) * ratio
    else:
        rte = compute_relative_trajectory_error(pos_pred, pos_gt, delta=pred_per_min)
    return rte


def test_pseudo_label_dict(pseudo_label_dict, data_path):
    """
    Args:
        pseudo_label_dict: a dictionary containing the pseudo label
        data_path: data path of target domain
    Returns:
        traj_length: length of trajectory
        step_count: how many steps in the trajectory
        changed_data_ratio: ratio of data with pseudo label 
        ste_before
        ste_after
    """
    # Load dataset
    adapt_dataset = IMUDataset(data_path)

    # Check pseudo label
    step_count = 0
    change_count = 0
    origin_ste = 0
    ours_ste = 0
    traj = []
    for i, data in enumerate(adapt_dataset):
        step_count += 1
        feat, label, frame_id = data
        traj.append(label)
        if pseudo_label_dict[frame_id]['is_pseudo_label']:
            pred = np.array(pseudo_label_dict[frame_id]['pred'])
            pseudo_label = np.array(pseudo_label_dict[frame_id]['pseudo_label'])
            change_count += 1
            origin_ste += np.sqrt(np.sum((label-pred) ** 2)).item() * 0.005
            ours_ste += np.sqrt(np.sum((label-pseudo_label) ** 2)).item() * 0.005
    traj = np.array(traj)
    traj_length = np.sum(np.sqrt(np.sum(traj ** 2, axis=1))) * 0.005
    changed_data_r = change_count/step_count
    return traj_length, step_count, changed_data_r, origin_ste/change_count, ours_ste/change_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_code_name', '-u', type=str, metavar='', required=True, help='enter a user code name')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='device')
    args = parser.parse_args()
    user_name = args.user_code_name

    model_path = '../model/pretrained_model.pt'
    data_path = '../data/%s.json' % user_name
    pseudo_label_dict = gen_pseudo_label(model_path, data_path, Q_FUNC, BLOCK_SIZE, args.device)
    with open('../data/%s_pseudo_label.json' % user_name, 'w') as fp:
        json.dump(pseudo_label_dict, fp)
    traj_l, step_count, changed_data_r, ste_before, ste_after = test_pseudo_label_dict(pseudo_label_dict, data_path)
    print('-' * 60)
    print('Information of %s:' % user_name)
    print('Trajectory length: %.2fm' % traj_l)
    print('Time period: %ss' % (step_count*2))
    print('Number of steps (2s): %s' % step_count)
    print('Uncertain data ratio: %.2f%%' % (changed_data_r*100))
    print('Average step error (STE) before adaptation: %.3fm' % ste_before)
    print('Average step error (STE) after adaptation: %.3fm' % ste_after)
    print('STE reduction rate: %.2f%%' % ((ste_before - ste_after ) / ste_before * 100))
    print('-' * 60)
