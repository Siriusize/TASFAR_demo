from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json


class NYCTaxiDataset(Dataset):
    def __init__(self, data_path, scaler=None, pseudo_label_path=None):
        """
        Dataset for California Dataset
        Args:
            data_path: path of the csv file
            scaler: given scaler. If none, we fit and transform the same data
            pseudo_label_path: path of pseudo label
        """
        self.data_path = data_path
        self.data = []
        self.pseudo_label_path = pseudo_label_path
        if self.pseudo_label_path:
            with open(self.pseudo_label_path, 'r') as fp:
                self.pseudo = json.load(fp)
        nyc_taxi = pd.read_csv(data_path, index_col=0)
        if scaler:
            nyc_taxi.iloc[:, :-1] = scaler.transform(nyc_taxi.iloc[:, :-1])
        else:
            self.scaler = StandardScaler()
            nyc_taxi.iloc[:, :-1] = self.scaler.fit_transform(nyc_taxi.iloc[:, :-1])

        for row_index, row in nyc_taxi.iterrows():
            if self.pseudo_label_path:
                self.data.append([np.array(row)[:-1], np.array([self.pseudo[str(row_index)]['pseudo_label']]), str(row_index)])
            else:
                self.data.append([np.array(row)[:-1], np.array([row['trip_duration']]), row_index])
        print('Dataset is constructed with %s pieces of data' % len(self.data))

    def __getitem__(self, item):
        if self.pseudo_label_path:
            return self.data[item][0].astype(np.float32), self.data[item][1].astype(np.float32), int(self.data[item][2]), np.array([self.pseudo[self.data[item][2]]['variance']]).astype(np.float32), \
                    np.array([self.pseudo[self.data[item][2]]['lmd']]).astype(np.float32), np.array([self.pseudo[self.data[item][2]]['gmd']]).astype(np.float32)
        else:
            return self.data[item][0].astype(np.float32), self.data[item][1].astype(np.float32), self.data[item][2]

    def __len__(self):
        return len(self.data)

    def get_scaler(self):
        return self.scaler