import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse

from dataset import IMUDataset
from model_temporal import TCNSeqNetwork
from metric import  compute_relative_trajectory_error


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


def test(model_path, data_path, pseudo_label_path, device):
    # Load data
    train_dataset = IMUDataset(data_path, pseudo_label_path, test=True)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1)
    test_dataset = train_dataset.get_test_dataset()
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)

    # Load network
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    network = TCNSeqNetwork(6, 2, 3, [32, 64, 128, 256, 72, 36])
    network.load_state_dict(torch.load(model_path).get('model_state_dict'))
    # network.load_state_dict(torch.load(model_path))
    network.to(device)

    # Test network on train dataset
    network.eval()
    pred_seq_train = []
    label_seq_train = []
    for data in train_dataloader:
        feat, label, _, _ = data
        feat, label = feat.to(device), label.to(device)
        pred = network(feat)
        pred = torch.sum(pred, dim=1)
        pred_seq_train.append(pred)
        label_seq_train.append(label)
    pred_seq_train = torch.concat(pred_seq_train, dim=0).cpu().detach().numpy()
    label_seq_train = torch.concat(label_seq_train, dim=0).cpu().detach().numpy()
    train_rte = cal_rte(pred_seq_train, label_seq_train)
    
    # Test network on test dataset
    network.eval()
    pred_seq_test = []
    label_seq_test = []
    for data in test_dataloader:
        feat, label, _ = data
        feat, label = feat.to(device), label.to(device)
        pred = network(feat)
        pred = torch.sum(pred, dim=1)
        pred_seq_test.append(pred)
        label_seq_test.append(label)
    pred_seq_test = torch.concat(pred_seq_test, dim=0).cpu().detach().numpy()
    label_seq_test = torch.concat(label_seq_test, dim=0).cpu().detach().numpy()
    test_rte = cal_rte(pred_seq_test, label_seq_test)

    return train_rte, test_rte, np.concatenate([pred_seq_train, pred_seq_test], axis=0) * 0.005, \
           np.concatenate([label_seq_train, label_seq_test], axis=0) * 0.005


def plot_trajectory(data_name, label_traj, origin_traj, pred_traj, fig_path):
    origin_point = np.array([[0, 0]])
    plt.grid(linestyle='-.')
    label_traj = np.cumsum(np.concatenate([origin_point, label_traj], axis=0), axis=0)
    origin_traj = np.cumsum(np.concatenate([origin_point, origin_traj], axis=0), axis=0)
    pred_traj = np.cumsum(np.concatenate([origin_point, pred_traj], axis=0), axis=0)
    plt.plot(label_traj[:, 0], label_traj[:, 1], label='Ground truth', color='blue', linestyle='--')
    plt.plot(origin_traj[:, 0], origin_traj[:, 1], label='Before adaptation', color='green', linestyle='-.')
    plt.plot(pred_traj[:, 0], pred_traj[:, 1], label='After adaptation', color='red', linestyle='-')
    plt.legend(fontsize=14)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_code_name', '-u', type=str, metavar='', required=True, help='enter a user code name')
    parser.add_argument('--device', '-d', type=str, default='cuda', help='device')
    args = parser.parse_args()
    user_name = args.user_code_name

    print('-' * 60)
    model_path = '../model/pretrained_model.pt'
    fig_path = '../figure/%s' % user_name
    data_path = '../data/%s.json' % user_name
    pseudo_label_path = '../data/%s_pseudo_label.json' % user_name
    train_rte_origin, test_rte_origin, origin_traj, label_traj = test(model_path, data_path, pseudo_label_path, args.device)
    model_path = '../model/%s_model.pt' % user_name
    data_path = '../data/%s.json' % user_name
    pseudo_label_path = '../data/%s_pseudo_label.json' % user_name
    train_rte_tasfar, test_rte_tasfar, tasfar_traj, label_traj = test(model_path, data_path, pseudo_label_path, args.device)
    print('Relative trajectory error (RTE) of adaptation set (origin): %.3f' % train_rte_origin)
    print('Relative trajectory error (RTE) of adaptation set (TASFAR): %.3f' % train_rte_tasfar)
    print('Relative trajectory error (RTE) of test set (origin): %.3f' % test_rte_origin)
    print('Relative trajectory error (RTE) of test set (TASFAR): %.3f' % test_rte_tasfar)
    plot_trajectory(user_name, label_traj, origin_traj, tasfar_traj, fig_path)
    print('Trajectory visualization has been saved to \'%s\'' % fig_path)
    print('-' * 60)