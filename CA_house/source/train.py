from dataset import CalHouseDataset, CalHouseDatasetAdapt
from network import ANN
import time
import os
import torch
from torch.utils.data import DataLoader
from test import test_model
from const import TRAINED_MODEL_PATH, THRESHOLD


def weighted_mse(pred, gt, lmd, gmd, var, thre):
    loss = (pred-gt) ** 2 * lmd / gmd * var / thre
    return torch.sqrt(torch.mean(loss))


def train(save_model_path, batch_size, device):
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
    train_dataset = CalHouseDatasetAdapt(domain_index='poor', pseudo_label_path='../data/poor_pseudo.json')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataset = CalHouseDataset(domain_index='poor')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize network
    net = ANN(input_size=8, output_size=1, hidden_sizes=[128, 128, 64], dropout=0.2)
    net.load_state_dict(torch.load(TRAINED_MODEL_PATH))
    net.to(device)
    print(net)

    # Set loss and optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    # Train network
    net.train()
    for ep in range(1, num_epochs + 1):
        train_loss = 0
        start_t = time.time()
        for i, data in enumerate(train_dataloader):
            x, pseudo_label, var, lmd, gmd = data
            x, pseudo_label, var, lmd, gmd = x.to(device), pseudo_label.to(device), var.to(device), lmd.to(device), gmd.to(device)
            pred = net(x)
            loss = weighted_mse(pred, pseudo_label, lmd, gmd, var, THRESHOLD)
            train_loss += loss.cpu().detach().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)
        end_t = time.time()

        log_string = 'Epoch %s (%.2fs): training loss: %.4f' % (ep, end_t - start_t, train_loss)
        print(log_string)
        with open(txt_path, 'a') as file:
            file.write(log_string + '\n')

        if ep % 1000 == 0:
            model_path = os.path.join(save_model_path, '%s.pt' % ep)
            torch.save(net.state_dict(), model_path)
            # Test network
            mse = test_model(model_path, test_dataloader, device)
            log_string = 'Epoch %s: mean squared error: %.4f' % (ep, mse)
            print(log_string)
            with open(txt_path, 'a') as file:
                file.write(log_string + '\n')


if __name__ == "__main__":
    save_model_path = '../model/training'
    if os.path.exists(save_model_path):
        print('EXIST!, do you want to continue?')
        exit()
    batch_size = 256
    device = 'cuda:0'
    train(save_model_path, batch_size, device)
