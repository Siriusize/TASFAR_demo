from dataset import NYCTaxiDataset
from network import ANN
import os
import torch
from torch.utils.data import DataLoader
import pickle
from test import test_model

THRESHOLD = 12102.397583007812

def weighted_rmsle(pred, gt, lmd, gmd, var, thre):
    loss = (torch.log(pred+1)-torch.log(gt+1)) ** 2 * lmd / gmd * var / thre
    return torch.sqrt(torch.mean(loss))


class RMSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


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
    num_epochs = 100

    # Load data
    scaler = pickle.load(open('../data/scaler.pkl', 'rb'))
    train_dataset = NYCTaxiDataset(data_path='../data/nyc_test.csv', scaler=scaler, pseudo_label_path=pseudo_label_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataset = NYCTaxiDataset(data_path='../data/nyc_test.csv', scaler=scaler)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Initialize network
    net = ANN(input_size=56, output_size=1, hidden_sizes=[128, 128, 64], dropout=0.2)
    net.load_state_dict(torch.load('../model/pretrained_model.pt', map_location=device))
    net.to(device)
    print(net)

    # Set loss and optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Train network
    net.train()
    for ep in range(1, num_epochs + 1):
        train_loss = 0
        for i, data in enumerate(train_dataloader):
            x, pseudo_label, row_index, var, lmd, gmd = data
            x, pseudo_label, var, lmd, gmd = x.to(device), pseudo_label.to(device), var.to(device), lmd.to(device), gmd.to(device)
            pred = net(x)
            loss = weighted_rmsle(pred, pseudo_label, lmd, gmd, var, THRESHOLD)
            train_loss += loss.cpu().detach().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)
        print('Training loss of epoch %s: %s' % (ep, train_loss))

    # Save model
    print('Training finished.')
    model_path = os.path.join(save_model_path, '%s.pt' % ep)
    torch.save(net.state_dict(), model_path)

    # Test network
    rmsle = test_model(model_path, test_dataloader, device)
    print('-' * 60)
    print('Duration RMSLE before adaptation: %.4f' % 0.5119)
    print('Duration RMSLE after adaptation (%s epochs): %.4f' % (ep, rmsle))
    print('RMSLE reduction rate (%s epochs): %.2f%%' % (ep, (0.5119 - rmsle) / 0.5119 * 100))
    print('The model has been saved to \'%s\'' % model_path)
    print('-' * 60)


if __name__ == "__main__":
    save_model_path = '../model/training'
    pseudo_label_path = '../data/pseudo_label.json'
    batch_size = 1024
    device = 'cuda:0'
    train(save_model_path, batch_size, pseudo_label_path, device)
