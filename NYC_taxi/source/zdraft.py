from dataset import NYCTaxiDataset
import pickle

temp_dataset = NYCTaxiDataset(data_path='../data/nyc_train.csv', data_type='trip_duration')

s = temp_dataset.get_scaler()

pickle.dump(s, open('../data/scaler.pkl','wb'))