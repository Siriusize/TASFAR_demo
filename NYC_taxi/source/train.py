from dataset import NYCTaxiDataset, NYCTaxiDatasetAdapt
from network import ANN
import time
import os
import torch
from torch.utils.data import DataLoader
from const import TRAINED_MODEL_PATH, THRESHOLD
import numpy as np
import pickle


def test_model(model_path, test_dataloader, device):
    net = ANN(input_size=56, output_size=1, hidden_sizes=[128, 128, 64], dropout=0.2)
    net.load_state_dict(torch.load(model_path))
    net.to(device)
    net.eval()
    msle_total = 0
    count = 0
    with torch.no_grad():
        for data in test_dataloader:
            x, label, index = data
            x = x.to(device)
            pred = net(x).cpu().detach().numpy()
            label = label.numpy()

            # Root Mean Squared Logarithmic Error
            # msle_total += np.sum((pred - label) ** 2).item()
            msle_total += np.sum((np.log(pred+1) - np.log(label+1)) ** 2).item()

            count += label.shape[0]
    return np.sqrt(msle_total / count)


def weighted_rmsle(pred, gt, lmd, gmd, var, thre):
    loss = (torch.log(pred+1)-torch.log(gt+1)) ** 2 * lmd / gmd * var / thre
    return torch.sqrt(torch.mean(loss))


class RMSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


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
    scaler = pickle.load(open('../data/scaler.pkl', 'rb'))
    train_dataset = NYCTaxiDatasetAdapt(data_path='../data/nyc_test.csv', data_type='trip_duration', pseudo_path='../data/pseudo.json', scaler=scaler)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataset = NYCTaxiDataset(data_path='../data/nyc_test.csv', data_type='trip_duration', scaler=scaler)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Initialize network
    net = ANN(input_size=56, output_size=1, hidden_sizes=[128, 128, 64], dropout=0.2)
    net.load_state_dict(torch.load(TRAINED_MODEL_PATH))
    net.to(device)
    print(net)

    # Set loss and optimizer
    # criterion = RMSLELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Train network
    net.train()
    for ep in range(1, num_epochs + 1):
        train_loss = 0
        start_t = time.time()
        for i, data in enumerate(train_dataloader):
            x, pseudo_label, row_index, var, lmd, gmd = data
            x, pseudo_label, var, lmd, gmd = x.to(device), pseudo_label.to(device), var.to(device), lmd.to(device), gmd.to(device)
            pred = net(x)
            # loss = weighted_rmsle(pred, pseudo_label, lmd, gmd, var, THRESHOLD)
            loss = weighted_rmsle(pred, pseudo_label, lmd, gmd, var, THRESHOLD)
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

        if ep % 10 == 0:
            model_path = os.path.join(save_model_path, '%s.pt' % ep)
            torch.save(net.state_dict(), model_path)
            # Test network
            mse = test_model(model_path, test_dataloader, device)
            log_string = 'Epoch %s: mean squared error: %.4f' % (ep, mse)
            print(log_string)
            with open(txt_path, 'a') as file:
                file.write(log_string + '\n')


if __name__ == "__main__":
    model_type = 'trainontest_origin'
    save_model_path = '../model/%s' % model_type
    # if os.path.exists(save_model_path):
    #     print('EXIST!, do you want to continue?')
    #     exit()
    batch_size = 1024
    device = 'cuda:0'
    train(save_model_path, batch_size, device)
