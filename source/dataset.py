from torch.utils.data import Dataset
import numpy as np
import json


class IMUDataset(Dataset):
    def __init__(self, data_path, pseudo_label_path=None, test=False):
        """
        IMU dataset
        Args:
            data_path: path of json data
            pseudo_label_path: path of json data saving pseudo labels
            test: if dataset is used for test
        """
        super().__init__()
        self.pseudo_label = None
        self.test = test
        with open(data_path, 'r') as fp:
            self.data = json.load(fp)
        if pseudo_label_path:
            with open(pseudo_label_path, 'r') as fp:
                self.pseudo_label = json.load(fp)
            self.gmd = self.pseudo_label['global_mean_density']

        self.train_data = []
        self.test_data = []
        train_length = round(len(self.data) * 0.8)
        for i, frame_data in enumerate(self.data):
            if i < train_length:
                self.train_data.append(frame_data)
            else:
                self.test_data.append(frame_data)

    def __getitem__(self, item):
        if self.test:
            return np.array(self.train_data[item]['imu_data']).astype(np.float32), \
                   np.array(self.train_data[item]['label']).astype(np.float32), \
                   self.train_data[item]['frame_id'], \
                   self.pseudo_label[str(self.train_data[item]['frame_id'])]['is_pseudo_label']
        elif self.pseudo_label:
            pseudo_label_key = str(self.train_data[item]['frame_id'])
            return np.array(self.train_data[item]['imu_data']).astype(np.float32), \
                   np.array(self.pseudo_label[pseudo_label_key]['pseudo_label']).astype(np.float32), \
                   self.pseudo_label[pseudo_label_key]['variance'], \
                   self.pseudo_label[pseudo_label_key]['local_mean_density'], \
                   self.train_data[item]['frame_id']
        else:
            return np.array(self.train_data[item]['imu_data']).astype(np.float32), \
                   np.array(self.train_data[item]['label']).astype(np.float32), \
                   self.train_data[item]['frame_id']

    def __len__(self):
        return len(self.train_data)
    
    def get_gmd(self):
        return self.gmd

    class TestDataset(Dataset):
        def __init__(self, data):
            super().__init__()
            self.data = data

        def __getitem__(self, item):
            return np.array(self.data[item]['imu_data']).astype(np.float32), \
                   np.array(self.data[item]['label']).astype(np.float32), \
                   self.data[item]['frame_id']

        def __len__(self):
            return len(self.data)

    def get_test_dataset(self):
        return IMUDataset.TestDataset(self.test_data)
