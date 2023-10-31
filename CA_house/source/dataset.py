from split_data import split_fun, DOMAIN_INDEX_DICT
from torch.utils.data import Dataset
from sklearn.datasets import fetch_california_housing
import numpy as np
import json


class CalHouseDataset(Dataset):
    def __init__(self, domain_index, pseudo_label_path=None):
        """
        Dataset for California Dataset
        Args:
            domain_index: 'rich' or 'poor'
            pseudo_label_path: path of pseudo label
        """
        self.pseudo_label_path = pseudo_label_path
        if self.pseudo_label_path:
            with open(pseudo_label_path, 'r') as fp:
                self.pseudo = json.load(fp)

        cal_house = fetch_california_housing(as_frame=True).frame
        self.data = []
        self.p_index = 0
        for row_index, row in cal_house.iterrows():
            temp_di = split_fun(row['Longitude'], row['Latitude'])
            if temp_di == DOMAIN_INDEX_DICT[domain_index]:
                temp_list = [np.array(row)[:-1], np.array([row['MedHouseVal']]), self.p_index]
                if self.pseudo_label_path:
                    str_index = str(self.p_index)
                    temp_list.append(np.array([self.pseudo[str_index]['pseudo_label']]))
                    temp_list.append(np.array([self.pseudo[str_index]['variance']]))
                    temp_list.append(np.array([self.pseudo[str_index]['lmd']]))
                    temp_list.append(np.array([self.pseudo[str_index]['gmd']]))
                self.p_index += 1
                self.data.append(temp_list)
        print('Dataset is constructed with %s pieces of data' % len(self.data))

    def __getitem__(self, item):
        if self.pseudo_label_path:
            return self.data[item][0].astype(np.float32), self.data[item][3].astype(np.float32), self.data[item][4].astype(np.float32), \
                   self.data[item][5].astype(np.float32), self.data[item][6].astype(np.float32)
        else:
            return self.data[item][0].astype(np.float32), self.data[item][1].astype(np.float32), self.data[item][2]

    def __len__(self):
        return len(self.data)
