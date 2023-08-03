import os
import torch
from torch.utils.data import DataLoader
import argparse
from dataset import IMUDataset
from model_temporal import TCNSeqNetwork
from test import test, plot_trajectory

THRESHOLD = 0.0020399182569235566


def weighted_mse(pred, label, lmd, gmd, var, thre):
    loss = torch.sum((pred-label)**2, axis=1) * lmd / gmd * var / thre
    return torch.mean(loss)


def train(save_path, model_path, data_path, pseodo_label_path, device):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    user_name = save_path.split('_')[-1]

    # Hyperparameters
    learning_rate = 1e-6
    num_epochs = 300  # for simplicity

    # Load data
    train_dataset = IMUDataset(data_path, pseodo_label_path)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=128, drop_last=False)
    gmd = train_dataset.get_gmd()

    # Load network
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    network = TCNSeqNetwork(6, 2, 3, [32, 64, 128, 256, 72, 36])
    network.load_state_dict(torch.load(model_path).get('model_state_dict'))
    network.to(device)

    # Set loss and optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    
    network.train()
    print('-' * 60)
    print('Training starts.')
    for epoch in range(1, num_epochs+1):
        train_loss = 0
        count = 0
        for data in train_dataloader:
            feat, pseudo_label, variance, lmd, _ = data
            feat, pseudo_label, variance, lmd = feat.to(device), pseudo_label.to(device), variance.to(device), lmd.to(device)
            optimizer.zero_grad()
            pred = network(feat)
            pred_sum = torch.sum(pred, axis=1)
            loss = weighted_mse(pred_sum, pseudo_label, lmd, gmd, variance, THRESHOLD)
            train_loss += loss.cpu().detach().numpy().item()
            count += feat.size()[0]
            loss.backward()
            optimizer.step()
        
        if epoch % 50 == 0:
            print('Training loss of epoch %s: %s' % (epoch, train_loss / count))

    # Save model
    print('Training finished.')
    model_saved_path = os.path.join(save_path, '%s_training.pt' % user_name)
    torch.save({'model_state_dict': network.state_dict()}, model_saved_path)
    print('The model has been saved to \'%s\'' % model_saved_path)
    print('-' * 60)


def test_model(user_name, saved_model_path, device):
    model_path = '../model/pretrained_model.pt'
    fig_path = '../figure/%s' % user_name
    data_path = '../data/%s.json' % user_name
    pseudo_label_path = '../data/%s_pseudo_label.json' % user_name
    train_rte_origin, test_rte_origin, origin_traj, label_traj = test(model_path, data_path, pseudo_label_path, device)
    model_path = saved_model_path
    data_path = '../data/%s.json' % user_name
    pseudo_label_path = '../data/%s_pseudo_label.json' % user_name
    train_rte_tasfar, test_rte_tasfar, tasfar_traj, label_traj = test(model_path, data_path, pseudo_label_path, device)
    print('Relative trajectory error (RTE) of adaptation set (origin): %.3f' % train_rte_origin)
    print('Relative trajectory error (RTE) of adaptation set (TASFAR): %.3f' % train_rte_tasfar)
    print('Relative trajectory error (RTE) of test set (origin): %.3f' % test_rte_origin)
    print('Relative trajectory error (RTE) of test set (TASFAR): %.3f' % test_rte_tasfar)
    plot_trajectory(user_name, label_traj, origin_traj, tasfar_traj, fig_path)
    print('Trajectory visualization has been saved to \'%s\'' % fig_path)
    print('-' * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_code_name', '-u', type=str, metavar='', required=True, help='enter a user code name')
    parser.add_argument('--device', '-d', type=str, default='cuda', help='device')
    args = parser.parse_args()
    user_name = args.user_code_name
    
    save_path = '../model/tasfar_%s' % user_name
    model_path = '../model/pretrained_model.pt'
    data_path = '../data/%s.json' % user_name
    pseodo_label_path = '../data/%s_pseudo_label.json' % user_name
    train(save_path, model_path, data_path, pseodo_label_path, args.device)

    fig_path = '../figure/%s' % user_name
    test_model(user_name, os.path.join(save_path, '%s_training.pt' % user_name), args.device)
