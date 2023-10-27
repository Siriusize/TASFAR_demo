import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

nyc = pd.read_csv('../data/all.csv', index_col=0)
print(nyc.head)


def cal_linear(list):
    x1, y1, x2, y2 = list
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return intercept, slope


def train_condition(longitude, latitude):
    A = (-74.042, 40.710)
    B = (-73.995, 40.692)
    C = (-73.918, 40.800)
    D = (-73.9818, 40.7684)
    AB = cal_linear(A+B)
    BC = cal_linear(B+C)
    # CD = cal_linear(C+D)

    below = (longitude * AB[1] + AB[0] <= latitude)
    right = (longitude * BC[1] + BC[0] <= latitude)
    # upper = (longitude * CD[1] + CD[0] >= latitude)
    # print(below)

    return (below & right)

nyc_train = nyc[train_condition(nyc['pickup_longitude'], nyc['pickup_latitude'])]
nyc_test = nyc[~train_condition(nyc['pickup_longitude'], nyc['pickup_latitude'])]
# nyc_train = nyc[nyc['pickup_weekday'] < 4.5]
# nyc_test = nyc[nyc['pickup_weekday'] > 4.5]

print(nyc_train.shape)
print(nyc_test.shape)

fig, axes = plt.subplots(2, 1)

axes[0].hist(nyc_train['trip_duration'], bins=np.linspace(0, 3600, 100), density=True)
print(np.mean(nyc_train['trip_duration']))
axes[1].hist(nyc_test['trip_duration'], bins=np.linspace(0, 3600, 100), density=True)
print(np.mean(nyc_test['trip_duration']))
plt.savefig('../figure/temp', bbox_inches='tight')

fig, axes = plt.subplots(2, 1)

axes[0].hist(np.log(nyc_train['trip_duration']+1), bins=np.linspace(1, 9, 100), density=True)
axes[1].hist(np.log(nyc_test['trip_duration']+1), bins=np.linspace(1, 9, 100), density=True)
plt.savefig('../figure/temp_log', bbox_inches='tight')

nyc_train.to_csv('../data/nyc_train.csv')
nyc_test.to_csv('../data/nyc_test.csv')