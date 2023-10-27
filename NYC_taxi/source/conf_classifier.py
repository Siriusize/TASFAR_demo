import numpy as np
from const import TRAINED_MODEL_PATH
import json
from network import ANN
import torch
from dataset import NYCTaxiDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import curve_fit


def cal_var(lst):
    """
    Args:
        lst: a list of prediction
    Returns:
        variance of lst
    """
    lst = torch.concat(lst, dim=1).detach().cpu().numpy()  # (1024, 20)
    return np.var(lst, axis=1)


def col_var_std():
    net = ANN(input_size=56, output_size=1, hidden_sizes=[128, 128, 64], dropout=0.2)
    print(net)
    net.load_state_dict(torch.load(TRAINED_MODEL_PATH))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    train_dataset = NYCTaxiDataset(data_path='../data/nyc_train.csv', data_type='trip_duration')
    # train_dataset = NYCTaxiDataset(data_path='../data/nyc_test.csv', data_type='trip_duration', scaler=temp_dataset.get_scaler())
    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=False, drop_last=False)

    json_data = {}

    for j, data in enumerate(train_dataloader):
        x, label, index = data  # torch.Size([1024, 56]) torch.Size([1024, 1]) torch.Size([1024])
        x = x.to(device)

        # Calculate Variance
        net.train()
        pred_list = []
        for i in range(20):
            pred = net(x)  # torch.Size([1024, 1])
            pred_list.append(pred)
        var = cal_var(pred_list)  # (1024, )

        # Calculate prediction for std (|pred-label|)
        net.eval()
        prediction = np.squeeze(net(x).detach().cpu().numpy())
        label = np.squeeze(label)

        for k in range(index.shape[0]):
            json_data[index[k].item()] = [var[k].item(), prediction[k].item(), label[k].item()]

    print(len(json_data))
    with open('../data/train_var_std.json', 'w') as fp:
        json.dump(json_data, fp)


def cal_linear(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return intercept, slope


def log_func(x, a, b, c):
    return a * np.log(x + c) + b


max_var = 0.4
seg = 0.01


def gen_q():
    with open('../data/train_var_std.json', 'r') as fp:
        json_data = json.load(fp)
    var_std_list = []
    for data in list(json_data.values()):
        var_std_list.append([data[0], abs(data[1]-data[2])])

    # var_std_list = [row for row in var_std_list if row[0] <= max_var]

    # point_list = []
    # for i in np.arange(0, max_var, seg):
    #     point_list.append(np.quantile(np.array([row for row in var_std_list if i <= row[0] < i + seg]), q=0.6827))
    # point_list2 = []
    # for i in range(len(point_list)-1):
    #     point_list2.append((point_list[i]+point_list[i+1])/2)

    # params = []
    # for i in range(len(point_list2)-1):
    #     params.append(cal_linear(i*seg+seg, point_list2[i], (i+1)*seg+seg, point_list2[i+1]))
    # params.insert(0, params[0])

    # with open('../data/params.json', 'w') as fp:
    #     json.dump(params, fp)

    var_std_list = np.array(var_std_list)
    X, Y = var_std_list[:, 0], var_std_list[:, 1]
    X = sm.add_constant(X[:, np.newaxis])
    quantreg = sm.QuantReg(Y, X)
    model = quantreg.fit(q=0.6827)
    print(model.params)
    predicted_Y = model.predict(X[:, np.newaxis])
    plt.plot(var_std_list[:, 0], predicted_Y, color='green')

    print('Threshold is %s' % np.quantile(var_std_list[:, 0], q=0.997))
    exit()
    plt.grid(linestyle='-.')
    plt.scatter(var_std_list[:, 0], var_std_list[:, 1], s=2)
    plt.plot([0, 20000], [3.99881886e+02, 3.99881886e+02+20000*3.56911669e-02], color='orange')

    # plt.scatter(np.arange(seg/2, max_var, seg), point_list, c='red')
    # plt.plot(np.arange(seg, max_var, seg), point_list2, color='purple')
    plt.xlabel('Variance')
    plt.ylabel('STD')
    # plt.savefig('../figure/train_q', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    gen_q()
    # col_var_std()
