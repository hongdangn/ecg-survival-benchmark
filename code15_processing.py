import numpy as np
import h5py
import os
import csv
import time
from torch import manual_seed
from torch.utils.data import random_split
from scipy.signal import resample
import pandas as pd

data_dir = "./HDF5_DATA/Code15/"
code15_ecg_dir = os.path.join(data_dir,'raw/')
code15_exams_dir = os.path.join(data_dir,'exams.csv')  
new_data_path = os.path.join(data_dir, 'Labels_Code15_mort_032025_pd_8020.csv') 

labels = pd.read_csv(code15_exams_dir)
list_ecgs = os.listdir(code15_ecg_dir)
list_ecgs = ["exams_part0.hdf5", "exams_part1.hdf5", "exams_part2.hdf5"] # added
labels = labels.loc[labels["trace_file"].isin(list_ecgs)].copy().reset_index() # added

# rename columns
labels.rename(columns={'exam_id': 'SID'}, inplace=True)
labels.rename(columns={'is_male': 'Is_Male'}, inplace=True)
labels.rename(columns={'age': 'Age'}, inplace=True)
labels.rename(columns={'death': 'Mort_Event'}, inplace=True)
labels.rename(columns={'timey': 'Mort_TTE'}, inplace=True)
labels.rename(columns={'patient_id': 'PID'}, inplace=True)

labels.loc[labels['Mort_TTE']<1/365,'Mort_TTE'] = 1/365
legitPIDs = labels.loc[np.isnan(labels['Mort_TTE'])==False]['PID'].to_numpy()

cnt_test = int(len(legitPIDs) * 0.2)
cnt_train = len(legitPIDs) - cnt_test
pids_train, pids_test = random_split(range(len(legitPIDs)), [cnt_train, cnt_test])   

pid_labels = {}
for i in pids_train:
    temp_PID = legitPIDs[i]
    pid_labels[temp_PID] = 'train'
    
for i in pids_test:
    temp_PID = legitPIDs[i]
    pid_labels[temp_PID] = 'test'
    
splits = []
for PID, TTE in zip(labels['PID'],labels['Mort_TTE']):
    if (np.isnan(TTE)):
        splits.append('NegTTE')
    else:
        splits.append(pid_labels[PID])
        
labels['train_test_split'] = splits

EIDs = []
ECGs = []
for f in list_ecgs: 
    print(f)
    with h5py.File(os.path.join(code15_ecg_dir, f), "r") as h:
        print("Keys: %s" % h.keys())
        EIDs.append(h['exam_id'][:-1])
        ECGs.append(resample(h['tracings'][:-1, 48:-48], 1000, axis=1).astype(np.float32)) # keep the middle 7 seconds @ 400Hz = 2800 samples
        
EIDs = np.hstack([k for k in EIDs])
ECGs = np.vstack([k for k in ECGs])

ECG_SID_label_dict = {}
for i,SID in enumerate(EIDs):
    ECG_SID_label_dict[SID] = i
    
reorder_arr = []
for SID in labels["SID"]:
    if (SID in ECG_SID_label_dict.keys()): 
        reorder_arr.append(ECG_SID_label_dict[SID])
    
EIDs = EIDs[reorder_arr]
ECGs = ECGs[reorder_arr]

with h5py.File(os.path.join(data_dir, 'Code15_ECG.hdf5'), "w") as f:
    f.create_dataset('ECG',       data = ECGs, chunks = (1,1000,12),  compression="gzip")
    f.create_dataset('SID',       data = EIDs)
    f.create_dataset('PID',       data = labels['PID'])
    
labels.to_csv (new_data_path)