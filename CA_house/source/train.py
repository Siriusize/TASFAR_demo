from dataset import CalHouseDataset
from network import ANN
import os
import torch
from torch.utils.data import DataLoader
from test import test_model

THRESHOLD = 0.04821875973500476


def weighted_mse(pred, gt, lmd, gmd, var, thre):
    loss = (pred-gt) ** 2 * lmd / gmd * var / thre
    return torch.sqrt(torch.mean(loss))


def train(save_model_path, batch_size, pseudo_label_path, device):
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    txt_path = os.path.join(save_model_path, 'log.txt')
    f = open(txt_path, 'w')
    f.close()

    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    learning_rate = 1e-6
    num_epochs = 5000

    # Load data
    train_dataset = CalHouseDataset(domain_index='poor', pseudo_label_path=pseudo_label_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataset = CalHouseDataset(domain_index='poor', pseudo_label_path=None)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize network
    net = ANN(input_size=8, output_size=1, hidden_sizes=[128, 128, 64], dropout=0.2)
    net.load_state_dict(torch.load('../model/pretrained_model.pt'))
    net.to(device)

    # Set loss and optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    # Train network
    net.train()
    for ep in range(1, num_epochs + 1):
        train_loss = 0
        for i, data in enumerate(train_dataloader):
            x, pseudo_label, var, lmd, gmd = data
            x, pseudo_label, var, lmd, gmd = x.to(device), pseudo_label.to(device), var.to(device), lmd.to(device), gmd.to(device)
            pred = net(x)
            loss = weighted_mse(pred, pseudo_label, lmd, gmd, var, THRESHOLD)
            train_loss += loss.cpu().detach().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if ep % 100 == 0:
            train_loss /= len(train_dataloader)
            print('Training loss of epoch %s: %s' % (ep, train_loss))

        if ep % 1000 == 0:
            model_path = os.path.join(save_model_path, '%s.pt' % ep)
            torch.save(net.state_dict(), model_path)
            # Test network
            mse = test_model(model_path, test_dataloader, device)
            print('-' * 60)
            print('Price MSE before adaptation: %.4f' % 0.2421)
            print('Price MSE after adaptation (%s epochs): %.4f' % (ep, mse))
            print('MSE reduction rate (%s epochs): %.2f%%' % (ep, (0.2421 - mse) / 0.2421 * 100))
            print('The model has been saved to \'%s\'' % model_path)
            print('-' * 60)


if __name__ == "__main__":
    save_model_path = '../model/training'
    pseudo_label_path = '../data/pseudo_label.json'
    batch_size = 256
    device = 'cuda:0'
    train(save_model_path, batch_size, pseudo_label_path, device)
