import torch
import numpy as np
from dataset import NYCTaxiDataset
from network import ANN
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


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
            msle_total += np.sum((pred - label) ** 2).item()
            # msle_total += np.sum((np.log(pred+1) - np.log(label+1)) ** 2).item()

            count += label.shape[0]
    return np.sqrt(msle_total / count)


def check_model():
    model_type = 'richonrich4'
    test_dataset = NYCTaxiDataset(domain_index='rich')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = 'cuda:0'
    model_path = '../model/%s/%s.pt' % (model_type, 3000)
    mse = test_model(model_path, test_dataloader, device)
    print(mse)


def check_log():
    model_type = 'richonrich4'
    log_path = '../model/%s/log.txt' % model_type
    f = open(log_path, 'r')
    lines = f.readlines()
    mse_list = []
    loss_list = []
    for i, line in enumerate(lines):
        if i % 110 == 109:
            mse = line.rstrip().split(' ')[-1]
            mse_list.append(float(mse))
        if i % 110 == 108:
            loss = line.rstrip().split(' ')[-1]
            loss_list.append(float(loss))

    # Plotting
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    lns1 = ax1.plot(range(100, 5001, 100), loss_list, c='blue', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.set_xlim(100, 5000)
    ax1.set_ylim(0, 1)

    lns2 = ax2.plot(range(100, 5001, 100), mse_list, c='green', label='MSE')
    ax2.set_ylabel('MSE on Test Dataset', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=18)

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, fontsize=18)
    ax1.grid(linestyle='-.')
    plt.savefig('../figure/%s_training_100' % model_type, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    check_model()
