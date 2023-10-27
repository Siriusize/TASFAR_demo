from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json


class NYCTaxiDataset(Dataset):
    def __init__(self, data_path, data_type, scaler=None):
        """
        Dataset for California Dataset
        Args:
            data_path: path of the csv file
            scaler: given scaler. If none, we fit and transform the same data
        """
        self.data_path = data_path
        self.data_type = data_type
        self.data = []
        nyc_taxi = pd.read_csv(data_path, index_col=0)
        if scaler:
            nyc_taxi.iloc[:, :-1] = scaler.transform(nyc_taxi.iloc[:, :-1])
        else:
            self.scaler = StandardScaler()
            nyc_taxi.iloc[:, :-1] = self.scaler.fit_transform(nyc_taxi.iloc[:, :-1])
        for row_index, row in nyc_taxi.iterrows():
            self.data.append([np.array(row)[:-1], np.array([row['trip_duration']]), row_index])
        print('Dataset is constructed with %s pieces of data' % len(self.data))

    def __getitem__(self, item):
        if self.data_type == 'trip_duration':
            return self.data[item][0].astype(np.float32), self.data[item][1].astype(np.float32), self.data[item][2]
        elif self.data_type == 'log_trip_duration':
            return self.data[item][0].astype(np.float32), np.log(self.data[item][1].astype(np.float32)+1), self.data[item][2]

    def __len__(self):
        return len(self.data)

    def get_scaler(self):
        return self.scaler


class NYCTaxiDatasetAdapt(Dataset):
    def __init__(self, data_path, data_type, pseudo_path, scaler=None):
        """
        Dataset for California Dataset
        Args:
            data_path: path of the csv file
            scaler: given scaler. If none, we fit and transform the same data
        """
        self.data_path = data_path
        self.data_type = data_type
        self.data = []
        self.pseudo_path = pseudo_path
        with open('../data/pseudo.json', 'r') as fp:
            self.pseudo = json.load(fp)
        nyc_taxi = pd.read_csv(data_path, index_col=0)
        if scaler:
            nyc_taxi.iloc[:, :-1] = scaler.transform(nyc_taxi.iloc[:, :-1])
        else:
            self.scaler = StandardScaler()
            nyc_taxi.iloc[:, :-1] = self.scaler.fit_transform(nyc_taxi.iloc[:, :-1])
        for row_index, row in nyc_taxi.iterrows():
            self.data.append([np.array(row)[:-1], np.array([self.pseudo[str(row_index)]['pseudo_label']]), str(row_index)])
        print('Dataset is constructed with %s pieces of data' % len(self.data))

    def __getitem__(self, item):
        if self.data_type == 'trip_duration':
            return self.data[item][0].astype(np.float32), self.data[item][1].astype(np.float32), int(self.data[item][2]), np.array([self.pseudo[self.data[item][2]]['variance']]).astype(np.float32), \
                    np.array([self.pseudo[self.data[item][2]]['lmd']]).astype(np.float32), np.array([self.pseudo[self.data[item][2]]['gmd']]).astype(np.float32),

        elif self.data_type == 'log_trip_duration':
            return self.data[item][0].astype(np.float32), np.log(self.data[item][1].astype(np.float32)+1), self.data[item][2]

    def __len__(self):
        return len(self.data)

    def get_scaler(self):
        return self.scaler