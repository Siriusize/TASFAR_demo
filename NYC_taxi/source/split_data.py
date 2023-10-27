import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

DOMAIN_INDEX_DICT = {'rich': 0, 'poor': 1}


def check_data():
    nyc_taxi = pd.read_csv('../data/train.csv')
    rng = np.random.RandomState(0)
    indices = rng.choice(
        np.arange(nyc_taxi.shape[0]), size=1000, replace=False
    )
    valid_indices = np.where(nyc_taxi['trip_duration'] <= 3600)[0]
    indices = np.intersect1d(indices, valid_indices)

    # Time
    nyc_taxi['time'] = pd.to_datetime(nyc_taxi['pickup_datetime'])
    nyc_taxi['hour'] = nyc_taxi['time'].dt.hour.astype(int)
    nyc_taxi['minute'] = nyc_taxi['time'].dt.minute.astype(int)
    nyc_taxi['second'] = nyc_taxi['time'].dt.second.astype(int)
    nyc_taxi['seconds'] = nyc_taxi['hour'] * 3600 + nyc_taxi['minute'] * 60 + nyc_taxi['second']

    sns.scatterplot(
        data=nyc_taxi.iloc[indices],
        x="seconds",
        y="trip_duration",
        size="trip_duration",
        hue="trip_duration",
        palette="viridis",
        alpha=0.5,
    )
    _ = plt.title("Trip duration depending of\n their pickup location")
    plt.grid(linestyle='-.')
    plt.xlim(0, 3600*24)
    plt.ylim(0, 2000)
    plt.xlabel('Time')
    plt.xticks([0, 3600*8, 3600*16, 3600*24], ['00:00', '08:00', '16:00', '24:00'])
    plt.savefig('../figure/data_map_time', bbox_inches='tight')
    plt.show()


def check_data_mean():
    nyc_taxi = pd.read_csv('../data/train.csv')
    print(nyc_taxi.shape)

    # Time
    nyc_taxi['time'] = pd.to_datetime(nyc_taxi['pickup_datetime'])
    nyc_taxi['month'] = nyc_taxi['time'].dt.month.astype(int)
    nyc_taxi['weekday'] = nyc_taxi['time'].dt.weekday.astype(int)
    nyc_taxi['hour'] = nyc_taxi['time'].dt.hour.astype(int)
    nyc_taxi['minute'] = nyc_taxi['time'].dt.minute.astype(int)
    nyc_taxi['second'] = nyc_taxi['time'].dt.second.astype(int)
    nyc_taxi['seconds'] = nyc_taxi['hour'] * 3600 + nyc_taxi['minute'] * 60 + nyc_taxi['second']

    total_array = np.zeros(12)
    total_count = np.zeros(12)
    for row_index, row in nyc_taxi.iterrows():
        if row['trip_duration'] <= 3600:
            total_array[row['month']] += row['trip_duration']
            total_count[row['month']] += 1

    for i in range(len(total_count)):
        if total_count[i] == 0:
            total_count[i] += 1
    total_mean = total_array / total_count
    plt.plot(total_mean)
    plt.grid(linestyle='-.')
    plt.xlim(0, 11)
    # ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])
    plt.savefig('../figure/data_map_month_mean', bbox_inches='tight')
    plt.show()


def split_fun(longitude, latitude):
    if longitude <= -120:
        y = - longitude * 6 / 5 - 108
        if y >= latitude:
            domain_index = DOMAIN_INDEX_DICT['rich']
        else:
            domain_index = DOMAIN_INDEX_DICT['poor']
    else:
        y = - longitude * 6 / 8 - 54
        if y >= latitude:
            domain_index = DOMAIN_INDEX_DICT['rich']
        else:
            domain_index = DOMAIN_INDEX_DICT['poor']
    return domain_index


def check_split():
    nyc_taxi = pd.read_csv('../data/train.csv')
    x = nyc_taxi['Longitude']
    y = nyc_taxi['Latitude']
    di_list = []
    for xx, yy in zip(x, y):
        di_list.append(split_fun(xx, yy))
    palette = {0: 'blue', 1: 'green'}
    sns.scatterplot(x=x, y=y, hue=di_list, palette=palette)
    plt.grid(linestyle='-.')
    plt.plot([-125, -120], [42, 36], c='red')
    plt.plot([-120, -112], [36, 30], c='red')
    plt.xlim(-125, -114)
    plt.ylim(32, 42)
    plt.savefig('../figure/check_split', bbox_inches='tight')
    plt.show()


def stats_data():
    nyc_taxi = pd.read_csv('../data/train.csv')
    x = nyc_taxi['Longitude']
    y = nyc_taxi['Latitude']
    rich_count = 0
    poor_count = 0
    for xx, yy in zip(x, y):
        di = split_fun(xx, yy)
        if di == 0:
            rich_count += 1
        else:
            poor_count += 1
    print(rich_count, poor_count)  # 15528, 5112


if __name__ == "__main__":
    check_data_mean()
