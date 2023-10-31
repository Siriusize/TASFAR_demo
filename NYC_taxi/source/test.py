import torch
import numpy as np
from dataset import NYCTaxiDataset
from network import ANN
from torch.utils.data import DataLoader
import pickle


def test_model(model_path, test_dataloader, device):
    net = ANN(input_size=56, output_size=1, hidden_sizes=[128, 128, 64], dropout=0.2)
    net.load_state_dict(torch.load(model_path, map_location=device))
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
            msle_total += np.sum((np.log(pred+1) - np.log(label+1)) ** 2).item()
            count += label.shape[0]

    return np.sqrt(msle_total / count)


if __name__ == "__main__":
    scaler = pickle.load(open('../data/scaler.pkl', 'rb'))
    test_dataset = NYCTaxiDataset(data_path='../data/nyc_test.csv', scaler=scaler)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    device = 'cuda:0'
    pretrained_model_path = '../model/pretrained_model.pt'
    pretrained_mse = test_model(pretrained_model_path, test_dataloader, device)
    adapted_model_path = '../model/adapted_model.pt'
    adapted_mse = test_model(adapted_model_path, test_dataloader, device)
    print('-' * 60)
    print('Duration RMSLE before adaptation: %.4f' % pretrained_mse)
    print('Duration RMSLE after adaptation: %.4f' % adapted_mse)
    print('RMSLE reduction rate: %.2f%%' % ((pretrained_mse - adapted_mse) / pretrained_mse * 100))
    print('-' * 60)

