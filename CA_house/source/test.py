import torch
from dataset import CalHouseDataset
from network import ANN
from torch.utils.data import DataLoader


def test_model(model_path, test_dataloader, device):
    net = ANN(input_size=8, output_size=1, hidden_sizes=[128, 128, 64], dropout=0.2)
    net.load_state_dict(torch.load(model_path))
    net.to(device)
    net.eval()
    mse_total = 0
    count = 0
    with torch.no_grad():
        for data in test_dataloader:
            x, label = data
            x = x.to(device)
            pred = net(x).item()
            mse_total += (pred - label) ** 2
            count += 1
    return mse_total / count


if __name__ == "__main__":
    test_dataset = CalHouseDataset(domain_index='poor')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = 'cuda:0'
    pretrained_model_path = '../model/pretrained_model.pt'
    pretrained_mse = test_model(pretrained_model_path, test_dataloader, device)
    adapted_model_path = '../model/adapted_model.pt'
    adapted_mse = test_model(adapted_model_path, test_dataloader, device)
    print('MSE of testing set (origin): %.4f' % pretrained_mse)
    print('MSE of testing set (TASFAR): %.4f' % adapted_mse)
