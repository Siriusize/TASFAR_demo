import numpy as np
from const import TRAINED_MODEL_PATH
import json
from network import ANN
import torch
from dataset import CalHouseDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import statsmodels.api as sm


def cal_var(lst):
    """
    Args:
        lst: a list of prediction
    Returns:
        variance of lst
    """
    lst = np.array(lst)
    return np.var(lst)


def col_var_std():
    net = ANN(input_size=8, output_size=1, hidden_sizes=[128, 128, 64], dropout=0.2)
    print(net)
    net.load_state_dict(torch.load(TRAINED_MODEL_PATH))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    train_dataset = CalHouseDataset(domain_index='rich')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    json_data = []

    for j, data in enumerate(train_dataloader):
        x, label = data
        x = x.to(device)

        # Calculate Variance
        net.train()
        pred_list = []
        for i in range(20):
            pred = net(x)
            pred_list.append(pred.item())
        var = cal_var(pred_list)

        # Calculate prediction for std (|pred-label|)
        net.eval()
        prediction = net(x).item()
        json_data.append([var, prediction, label.item()])

        print(j)
    print(len(json_data))
    # with open('../data/rich_var_std.json', 'w') as fp:
    #     json.dump(json_data, fp)


def gen_q():
    with open('../data/rich_var_std.json', 'r') as fp:
        json_data = json.load(fp)
    var_std_list = []
    for data in json_data:
        var_std_list.append([data[0], abs(data[1]-data[2])])
    var_std_list = [row for row in var_std_list if row[0] < 0.2]
    var_std_list = np.array(var_std_list)

    X, Y = var_std_list[:, 0], var_std_list[:, 1]
    X = sm.add_constant(X[:, np.newaxis])
    quantreg = sm.QuantReg(Y, X)
    model = quantreg.fit(q=0.6827)
    predicted_Y = model.predict(X[:, np.newaxis])

    plt.scatter(var_std_list[:, 0], Y, s=2)
    plt.plot(var_std_list[:, 0], predicted_Y, color='red')
    print('Threshold is %s' % np.quantile(var_std_list[:, 0], q=0.85))
    print('The coefficients for the line is %s' % model.params)

    plt.grid(linestyle='-.')
    plt.xlabel('Variance')
    plt.ylabel('STD')
    plt.xlim(0, 0.2)
    plt.ylim(0, 4.5)
    plt.savefig('../figure/rich_q', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # col_var_std()
    gen_q()
